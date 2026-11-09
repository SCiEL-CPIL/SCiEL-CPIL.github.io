import time
import sys

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    n, m, p = DIMENSIONS
    # n=3, m=4, p=6
    
    # We will build the decomposition column by column (rank term by term)
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # Initialize vectors with zeros
    # U has size n*m = 12
    # V has size m*p = 24
    # W has size p*n = 18
    
    # Strassen decomposition for 2x2 * 2x2
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = AB
    # Terms (u, v, w) where u is index in A, v in B, w in C
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # We need to map local indices 0..3 for A and B, and 0..3 for C
    # to global indices.
    
    # For a specific column block c (0, 1, 2):
    # B columns are 2c, 2c+1.
    # B local row 0,1 correspond to global rows 0,1 (top block of B)
    # B local row 2,3 correspond to global rows 2,3 (bottom block of B)
    # Wait, B is 4x6.
    # B[0:2, :] is top, B[2:4, :] is bottom.
    # So for column block c:
    # B_top = B[0:2, 2c:2c+2] -> local indices 0,1 for rows, 0,1 for cols
    # B_bot = B[2:4, 2c:2c+2] -> local indices 2,3 for rows, 0,1 for cols
    
    # A is 3x4.
    # A_top = A[0:2, :] -> rows 0,1
    # A_bot = A[2:3, :] -> row 2
    
    # For top part C[0:2, 2c:2c+2]:
    # C = A_top * B_col_c
    # A_top = [[A00, A01, A02, A03], [A10, A11, A12, A13]]
    # We split A_top into A11 (cols 0,1) and A12 (cols 2,3).
    # A11 = A[0:2, 0:2]
    # A12 = A[0:2, 2:4]
    
    # B_col_c = [[B00, B01], [B10, B11], [B20, B21], [B30, B31]]
    # B1c = B[0:2, 2c:2c+2] (rows 0,1)
    # B2c = B[2:4, 2c:2c+2] (rows 2,3)
    
    # C_top = A11 * B1c + A12 * B2c
    
    # We use Strassen for A11 * B1c and A12 * B2c.
    
    # For bottom part C[2:3, 2c:2c+2]:
    # C_bot = A_bot * B_col_c
    # A_bot = A[2:3, :] -> [A20, A21, A22, A23]
    # B_col_c is 4x2.
    # C_bot[0,0] = A20*B[0,2c] + A21*B[1,2c] + A22*B[2,2c] + A23*B[3,2c]
    # C_bot[0,1] = A20*B[0,2c+1] + A21*B[1,2c+1] + A22*B[2,2c+1] + A23*B[3,2c+1]
    # This is 8 independent multiplications.

    def get_strassen_terms(a_base, b_base, c_base):
        """
        Returns list of (u_idx, v_idx, w_idx, coeff) for a 2x2x2 Strassen block.
        a_base: tuple (row_offset, col_offset) for A
        b_base: tuple (row_offset, col_offset) for B
        c_base: tuple (row_offset, col_offset) for C
        """
        terms = []
        # M1
        # (a00+a11)(b00+b11) -> C00 +1, C11 +1
        # a00: (0,0), a11: (1,1)
        # b00: (0,0), b11: (1,1)
        # We expand the product of sums into individual terms?
        # No, Strassen terms are bilinear forms.
        # M1 = (A00+A11) * (B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # But we want to represent it as 1 rank-1 term? No, Strassen uses 7 products.
        # Each product is a scalar multiplication of a linear combination of A entries and B entries.
        # So U, V, W entries can be non-zero in multiple places.
        
        # The task asks for U, V, W matrices.
        # The decomposition is C = sum U_c V_c W_c^T (conceptually).
        # Actually, the tensor is T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}?
        # No, the standard form is:
        # C_{ik} = sum_{j,c} A_{ij} B_{jk} = sum_c (sum_j U_{ij,c} A_{ij}) (sum_j V_{jk,c} B_{jk}) W_{ik,c}?
        # The interface says:
        # U (n*m, rank), V (m*p, rank), W (p*n, rank).
        # This implies:
        # C = sum_c U_c \otimes V_c \otimes W_c
        # Where U_c is a vector of length n*m (flattened A),
        # V_c is a vector of length m*p (flattened B),
        # W_c is a vector of length p*n (flattened C).
        
        # So for a term like A00 * B00 contributing to C00:
        # U has 1 at index (0,0)
        # V has 1 at index (0,0)
        # W has 1 at index (0,0) (since C00 is at k*n+i = 0*3+0=0)
        
        # For M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # This single "multiplication" M1 contributes to C00 and C11.
        # C00 += M1
        # C11 += M1
        # So for the term corresponding to M1:
        # U_c has 1 at A00 and 1 at A11.
        # V_c has 1 at B00 and 1 at B11.
        # W_c has 1 at C00 and 1 at C11.
        
        # This is valid.
        
        def make_vec(size, indices, val=1):
            v = [0] * size
            for idx in indices:
                v[idx] += val
            return v

        a_size = 4 # 2x2
        b_size = 4
        c_size = 4
        
        # Local indices for 2x2:
        # A: 00->0, 01->1, 10->2, 11->3
        # B: 00->0, 01->1, 10->2, 11->3
        # C: 00->0, 01->1, 10->2, 11->3
        
        # M1
        u = make_vec(a_size, [0, 3])
        v = make_vec(b_size, [0, 3])
        w = make_vec(c_size, [0, 3])
        terms.append((u, v, w))
        
        # M2 = (A10+A11)B00
        u = make_vec(a_size, [2, 3])
        v = make_vec(b_size, [0])
        w = make_vec(c_size, [2]) # C10
        terms.append((u, v, w))
        
        # M3 = A00(B01-B11)
        u = make_vec(a_size, [0])
        v = make_vec(b_size, [1, 3], -1) # B01 - B11
        w = make_vec(c_size, [1]) # C01
        terms.append((u, v, w))
        
        # M4 = A11(B10-B00)
        u = make_vec(a_size, [3])
        v = make_vec(b_size, [2, 0], -1) # B10 - B00
        w = make_vec(c_size, [0, 2]) # C00 + C10
        terms.append((u, v, w))
        
        # M5 = (A00+A01)B11
        u = make_vec(a_size, [0, 1])
        v = make_vec(b_size, [3])
        w = make_vec(c_size, [1, 3]) # C01 + C11? 
        # Wait. C01 = M3 + M5. C11 = M1 - M2 + M3 + M6.
        # M5 contributes to C01 and C11?
        # Formula: C01 = M3 + M5. C11 = M1 - M2 + M3 + M6.
        # So M5 only contributes to C01?
        # Let's check Strassen again.
        # C11 = M1 - M2 + M3 + M6.
        # C01 = M3 + M5.
        # So M5 contributes to C01 only?
        # My previous check:
        # M5 = (A00+A01)B11.
        # (A00+A01)B11 = A00B11 + A01B11.
        # In C = AB:
        # C01 = A00B01 + A01B11.
        # C11 = A10B01 + A11B11.
        # So M5 is NOT a direct term in the sum for C11.
        # The formula for C11 does not have M5.
        # So W for M5 should only have 1 at C01 (index 1).
        w = make_vec(c_size, [1])
        terms.append((u, v, w))
        
        # M6 = (A10-A00)(B00+B01)
        u = make_vec(a_size, [2, 0], -1) # A10 - A00
        v = make_vec(b_size, [0, 1])
        w = make_vec(c_size, [3]) # C11
        terms.append((u, v, w))
        
        # M7 = (A01-A11)(B10+B11)
        u = make_vec(a_size, [1, 3], -1) # A01 - A11
        v = make_vec(b_size, [2, 3])
        w = make_vec(c_size, [0]) # C00
        terms.append((u, v, w))
        
        return terms

    # Loop over column blocks
    for c in range(3):
        if time.time() > deadline:
            break
            
        # Top part: A[0:2, :] * B[:, 2c:2c+2]
        # A11: rows 0,1; cols 0,1. Base (0,0).
        # A12: rows 0,1; cols 2,3. Base (0,2).
        # B1c: rows 0,1; cols 2c,2c+1. Base (0, 2c).
        # B2c: rows 2,3; cols 2c,2c+1. Base (2, 2c).
        # C_top: rows 0,1; cols 2c,2c+1. Base (0, 2c).
        
        # Term 1: A11 * B1c
        terms1 = get_strassen_terms((0,0), (0, 2*c), (0, 2*c))
        for u, v, w in terms1:
            # Map to global indices
            U_g = [0] * (n*m)
            V_g = [0] * (m*p)
            W_g = [0] * (p*n)
            
            # u is local 2x2 (size 4). Map to A[0:2, 0:2]
            for li in range(4):
                r, col = divmod(li, 2)
                gi = r
                gj = col
                U_g[gi*m + gj] = u[li]
                
            # v is local 2x2 (size 4). Map to B[0:2, 2c:2c+2]
            for li in range(4):
                r, col = divmod(li, 2)
                gi = r
                gj = 2*c + col
                V_g[gi*p + gj] = v[li]
                
            # w is local 2x2 (size 4). Map to C[0:2, 2c:2c+2]
            for li in range(4):
                r, col = divmod(li, 2)
                gi = r
                gj = 2*c + col
                # W index is k*n + i. Here k is col of C, i is row of C.
                W_g[gj*n + gi] = w[li]
                
            add_term(U_g, V_g, W_g)

        # Term 2: A12 * B2c
        terms2 = get_strassen_terms((0,2), (2, 2*c), (0, 2*c))
        for u, v, w in terms2:
            U_g = [0] * (n*m)
            V_g = [0] * (m*p)
            W_g = [0] * (p*n)
            
            # u local 2x2. Map to A[0:2, 2:4]
            for li in range(4):
                r, col = divmod(li, 2)
                gi = r
                gj = 2 + col
                U_g[gi*m + gj] = u[li]
                
            # v local 2x2. Map to B[2:4, 2c:2c+2]
            for li in range(4):
                r, col = divmod(li, 2)
                gi = 2 + r
                gj = 2*c + col
                V_g[gi*p + gj] = v[li]
                
            # w local 2x2. Map to C[0:2, 2c:2c+2]
            for li in range(4):
                r, col = divmod(li, 2)
                gi = r
                gj = 2*c + col
                W_g[gj*n + gi] = w[li]
                
            add_term(U_g, V_g, W_g)

        # Bottom part: A[2:3, :] * B[:, 2c:2c+2]
        # 8 terms
        for j in range(4):
            for k in range(2):
                U_g = [0] * (n*m)
                V_g = [0] * (m*p)
                W_g = [0] * (p*n)
                
                # A index: row 2, col j
                U_g[2*m + j] = 1
                # B index: row j, col 2c+k
                V_g[j*p + (2*c+k)] = 1
                # C index: row 2, col 2c+k
                W_g[(2*c+k)*n + 2] = 1
                
                add_term(U_g, V_g, W_g)

    # Convert lists to matrices
    rank = len(U_cols)
    U = [U_cols[i] for i in range(rank)]
    V = [V_cols[i] for i in range(rank)]
    W = [W_cols[i] for i in range(rank)]
    
    return {"U": U, "V": V, "W": W}
