import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    """Construct the factor matrices for matrix multiplication 3x4x6.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    """
    n, m, p = DIMENSIONS
    # n=3, m=4, p=6
    
    # We will build the decomposition term by term.
    # We use a list of tuples (u_idx, v_idx, w_idx, coeff) to build the matrices.
    # Then we convert to the required format.
    # Actually, it's easier to just append to lists.
    
    # Strassen's algorithm for 2x2x2
    # U, V, W are 4x7 matrices.
    # Indices for A (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for B (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for C (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # We define the 7 terms as (u_vec, v_vec, w_vec)
    # u_vec, v_vec, w_vec are length 4.
    
    def get_strassen_terms():
        terms = []
        # M1
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1])) # C00 += M1
        # M2
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0])) # C10 += M2
        # M3
        terms.append(([1, 0, 0, 0], [0, -1, 1, 0], [0, 1, 0, 0])) # C01 += M3
        # M4
        terms.append(([0, 0, 0, 1], [-1, 1, 0, 0], [1, 0, 0, 1])) # C00 += M4, C11 += M4
        # Wait, M4 contributes to C00 and C11?
        # Z00 = M1 + M4 - M5 + M7
        # Z11 = M1 - M2 + M3 + M6
        # My previous mapping was slightly off. Let's re-verify Strassen output.
        
        # Standard Strassen:
        # P1 = (A11 + A22)(B11 + B22)
        # P2 = (A21 + A22)B11
        # P3 = A11(B21 - B22)
        # P4 = A22(B12 - B11)
        # P5 = (A11 + A12)B22
        # P6 = (A21 - A11)(B11 + B12)
        # P7 = (A12 - A22)(B21 + B22)
        
        # C11 = P1 - P4 + P5 - P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # Let's use 0-based indexing for 2x2:
        # A00, A01, A10, A11
        # B00, B01, B10, B11
        # C00, C01, C10, C11
        
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B10-B11)
        # P4 = A11(B01-B00)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B10)
        # P7 = (A01-A11)(B11+B01)
        
        # C00 = P1 - P4 + P5 - P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # We need to express each Pk as a rank-1 term u_k v_k^T?
        # No, Pk is a scalar. The term is u_k * A * v_k * B * w_k?
        # In tensor decomposition, each term is u \otimes v \otimes w.
        # The contribution to C is (u^T A) (v^T B) w.
        # So for P1, u1 = [1,0,0,1], v1 = [1,0,0,1].
        # P1 contributes to C00 and C11.
        # So w1 must have 1 at C00 and 1 at C11?
        # Yes, w1 = [1, 0, 0, 1].
        
        # Let's list the 7 terms (u, v, w)
        # 1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1] (P1)
        # 2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0] (P2 -> C10)
        # 3: u=[1,0,0,0], v=[0,-1,1,0], w=[0,1,0,0] (P3 -> C01, C11)
        #    Wait, P3 contributes to C01 and C11.
        #    C01 = P3 + P5. C11 = P1 - P2 + P3 + P6.
        #    So w3 should have 1 at C01 and 1 at C11.
        #    w3 = [0, 1, 0, 1].
        # 4: u=[0,0,0,1], v=[-1,1,0,0], w=[1,0,0,1] (P4 -> C00, C10)
        #    C00 = ... - P4 ...
        #    C10 = P2 + P4.
        #    So w4 = [-1, 0, 1, 0]. (Coef -1 for C00, +1 for C10)
        # 5: u=[1,1,0,0], v=[0,0,0,1], w=[1,1,0,0] (P5 -> C00, C01)
        #    C00 = ... + P5 ...
        #    C01 = P3 + P5.
        #    So w5 = [1, 1, 0, 0].
        # 6: u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,0,1] (P6 -> C11)
        #    C11 = ... + P6.
        #    So w6 = [0, 0, 0, 1].
        # 7: u=[0,1,0,-1], v=[0,1,1,0], w=[-1,0,0,0] (P7 -> C00)
        #    C00 = ... - P7.
        #    So w7 = [-1, 0, 0, 0].
        
        return [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
            ([1, 0, 0, 0], [0, -1, 1, 0], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 1, 0, 0], [-1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 1, 1, 0], [-1, 0, 0, 0])
        ]

    strassen_terms = get_strassen_terms()
    
    # Global dimensions
    n, m, p = 3, 4, 6
    U = [[0] * 66 for _ in range(n * m)]
    V = [[0] * 66 for _ in range(m * p)]
    W = [[0] * 66 for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Row 0 of A (i=0)
    # A[0, j] for j=0..3
    # B[j, k] for k=0..5
    # C[0, k] for k=0..5
    # Split B into 3 blocks of 2 cols: (0,1), (2,3), (4,5)
    for s in range(3):
        k_start = 2 * s
        # Block is [1, 4, 2]
        # Schoolbook rank 8
        for j in range(4):
            for k_local in range(2):
                k_global = k_start + k_local
                U[0 * m + j][col] = 1
                V[j * p + k_global][col] = 1
                W[k_global * n + 0][col] = 1
                col += 1
                
    # Part 2: Rows 1,2 of A (i=1,2)
    # A[1..2, j] for j=0..3
    # Split A into Left (j=0,1) and Right (j=2,3)
    # Split B into 3 blocks of 2 cols: (0,1), (2,3), (4,5)
    # For each B block, split B into Top (j=0,1) and Bot (j=2,3)
    # Product C[1..2, k_start..k_start+1]
    
    for s in range(3):
        k_start = 2 * s
        
        # Left-Top Block: A[1..2, 0..1] * B[0..1, k_start..k_start+1]
        # A indices: (1,0)->4, (1,1)->5, (2,0)->6, (2,1)->7
        # B indices: (0,k_start)->0*6+k_start, (0,k_start+1)->1, (1,k_start)->6+k_start, (1,k_start+1)->2
        # C indices: (1,k_start)->k_start*3+1, (1,k_start+1)->(k_start+1)*3+1, (2,k_start)->k_start*3+2, (2,k_start+1)->(k_start+1)*3+2
        
        a_idx_lt = [1*m+0, 1*m+1, 2*m+0, 2*m+1] # 4, 5, 6, 7
        b_idx_lt = [0*p+k_start, 0*p+k_start+1, 1*p+k_start, 1*p+k_start+1]
        c_idx_lt = [k_start*n+1, (k_start+1)*n+1, k_start*n+2, (k_start+1)*n+2]
        
        for u, v, w in strassen_terms:
            for r in range(4):
                if u[r] != 0:
                    U[a_idx_lt[r]][col] += u[r]
                if v[r] != 0:
                    V[b_idx_lt[r]][col] += v[r]
                if w[r] != 0:
                    W[c_idx_lt[r]][col] += w[r]
            col += 1
            
        # Right-Bot Block: A[1..2, 2..3] * B[2..3, k_start..k_start+1]
        # A indices: (1,2)->6, (1,3)->7, (2,2)->8, (2,3)->9
        # B indices: (2,k_start)->12+k_start, (2,k_start+1)->13, (3,k_start)->18+k_start, (3,k_start+1)->19
        # C indices: same as above
        
        a_idx_rb = [1*m+2, 1*m+3, 2*m+2, 2*m+3] # 6, 7, 8, 9
        b_idx_rb = [2*p+k_start, 2*p+k_start+1, 3*p+k_start, 3*p+k_start+1]
        c_idx_rb = c_idx_lt
        
        for u, v, w in strassen_terms:
            for r in range(4):
                if u[r] != 0:
                    U[a_idx_rb[r]][col] += u[r]
                if v[r] != 0:
                    V[b_idx_rb[r]][col] += v[r]
                if w[r] != 0:
                    W[c_idx_rb[r]][col] += w[r]
            col += 1
            
    return {"U": U, "V": V, "W": W}
