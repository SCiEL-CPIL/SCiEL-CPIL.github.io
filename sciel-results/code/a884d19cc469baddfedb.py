"""Strassen-based decomposition for 4x4 * 4x5 matrix multiplication."""
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_222():
    """
    Return the 7 Strassen products for 2x2 * 2x2.
    A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    
    Strassen:
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # Returns list of (U_factors, V_factors, W_factors) for each of 7 terms
    # U: 4 rows (a,b,c,d indexed as 0,1,2,3), V: 4 rows (e,f,g,h as 0,1,2,3)
    # W: 4 rows (C11,C12,C21,C22 as 0,1,2,3)
    
    # Each term: (U_vec, V_vec, W_vec) where U_vec[i]*V_vec[j]*W_vec[k] contributes
    # to the tensor. Actually, let me think of it as: the tensor T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
    # where i indexes A elements, j indexes B elements, k indexes C elements.
    
    # A elements: a=0, b=1, c=2, d=3
    # B elements: e=0, f=1, g=2, h=3
    # C elements: C11=0, C12=1, C21=2, C22=3
    
    # M1 = (a+d)(e+h) contributes to C11 (+1), C22 (+1)
    # M2 = (c+d)e contributes to C21 (+1), C22 (-1)
    # M3 = a(f-h) contributes to C12 (+1), C22 (+1)
    # M4 = d(g-e) contributes to C11 (+1), C21 (+1)
    # M5 = (a+b)h contributes to C11 (-1), C12 (+1)
    # M6 = (c-a)(e+f) contributes to C22 (+1)
    # M7 = (b-d)(g+h) contributes to C11 (+1)
    
    terms = []
    
    # M1: (a+d)(e+h)
    # U: a=1, d=1 => [1,0,0,1]
    # V: e=1, h=1 => [1,0,0,1]
    # W: C11=1, C22=1 => [1,0,0,1]
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # M2: (c+d)e
    # U: c=1, d=1 => [0,0,1,1]
    # V: e=1 => [1,0,0,0]
    # W: C21=1, C22=-1 => [0,0,1,-1]
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    
    # M3: a(f-h)
    # U: a=1 => [1,0,0,0]
    # V: f=1, h=-1 => [0,1,0,-1]
    # W: C12=1, C22=1 => [0,1,0,1]
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,1]))
    
    # M4: d(g-e)
    # U: d=1 => [0,0,0,1]
    # V: g=1, e=-1 => [-1,0,1,0]
    # W: C11=1, C21=1 => [1,0,1,0]
    terms.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0]))
    
    # M5: (a+b)h
    # U: a=1, b=1 => [1,1,0,0]
    # V: h=1 => [0,0,0,1]
    # W: C11=-1, C12=1 => [-1,1,0,0]
    terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    
    # M6: (c-a)(e+f)
    # U: c=1, a=-1 => [-1,0,1,0]
    # V: e=1, f=1 => [1,1,0,0]
    # W: C22=1 => [0,0,0,1]
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    
    # M7: (b-d)(g+h)
    # U: b=1, d=-1 => [0,1,0,-1]
    # V: g=1, h=1 => [0,0,1,1]
    # W: C11=1 => [1,0,0,0]
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    
    return terms


def _schoolbook_221():
    """
    Return the 4 schoolbook products for 2x2 * 2x1.
    A = [[a,b],[c,d]], B = [[e],[f]], C = [[ae+bf],[ce+df]]
    """
    # A: a=0, b=1, c=2, d=3
    # B: e=0, f=1
    # C: C1=0, C2=1
    
    terms = []
    # ae -> C1
    terms.append(([1,0,0,0], [1,0], [1,0]))
    # bf -> C1
    terms.append(([0,1,0,0], [0,1], [1,0]))
    # ce -> C2
    terms.append(([0,0,1,0], [1,0], [0,1]))
    # df -> C2
    terms.append(([0,0,0,1], [0,1], [0,1]))
    
    return terms


def construct():
    deadline = time.time() + 140  # safety margin
    
    # For each of the 4 blocks (2x2 of 2x2 * 2x3):
    # Block (i,j) where i in {0,1} (row block of A), j in {0,1} (col block of B)
    # A_block[i,j] is 2x2, B_block[j,?] ... 
    
    # Let me define the block structure:
    # A is 4x4, split into 2x2 blocks of 2x2:
    #   A[0:2, 0:2] = A00, A[0:2, 2:4] = A01
    #   A[2:4, 0:2] = A10, A[2:4, 2:4] = A11
    #
    # B is 4x5, split into 2x2 blocks of 2x3:
    #   B[0:2, 0:3] = B00, B[0:2, 3:5] = B01
    #   B[2:4, 0:3] = B10, B[2:4, 3:5] = B11
    #
    # C is 4x5, split into 2x2 blocks of 2x3:
    #   C[0:2, 0:3] = C00, C[0:2, 3:5] = C01
    #   C[2:4, 0:3] = C10, C[2:4, 3:5] = C11
    #
    # C_ij = sum_k A_ik * B_kj
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # For each 2x2 * 2x3 multiplication, I'll use:
    #   Split B (2x3) into B_left (2x2) and B_right (2x1)
    #   A*B_left: Strassen rank 7
    #   A*B_right: schoolbook rank 4
    #   Total per block: 11
    
    # But I need to be careful with the W index convention: W[k*n+i, r]
    # where k indexes rows of B (0..p-1), i indexes rows of A (0..n-1)
    # Wait, let me re-read: W[k*n+i, r] where n=4, p=5
    # So W has p*n = 20 rows, indexed by (k, i) -> k*4 + i
    # k is the column index in B (which is the row index in the B matrix, 0..3)
    # i is the row index in A (0..3)
    
    # Actually, let me re-read the interface:
    # U: (n*m, rank) = (16, rank) - indexed by (i, j) where i is row of A, j is col of A / row of B
    # V: (m*p, rank) = (20, rank) - indexed by (j, k) where j is row of B, k is col of B
    # W: (p*n, rank) = (20, rank) - indexed by (k, i) where k is col of B, i is row of A
    #   W[k*n+i, r] means row index = k*4 + i
    
    # The tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal delta for matrix multiplication, i.e., T[i,j,k] = 1 if the 
    # (i,k) entry of C = A*B gets contribution from A[i,j]*B[j,k].
    
    # Actually T[i,j,k] = 1 for all valid (i,j,k) in schoolbook, since C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Let me build the full decomposition.
    
    # For each block combination, I'll add terms.
    
    # Let me define helper to add terms with offset.
    # U index: i*m + j = i*4 + j
    # V index: j*p + k = j*5 + k  
    # W index: k*n + i = k*4 + i
    
    U_dict = {}  # (u_idx, rank) -> value
    V_dict = {}
    W_dict = {}
    
    rank = 0
    
    def add_term(U_vec, V_vec, W_vec, u_offset, v_offset, w_offset):
        """
        Add a term. U_vec has length 4 (for 2x2 A block elements a,b,c,d)
        V_vec has length 4 (for 2x2 B block elements) or 2 (for 2x1)
        W_vec has length 4 (for 2x2 C block) or 2 (for 2x1 C block)
        
        u_offset: (i0, j0) - top-left of A block
        v_offset: (j0, k0) - top-left of B block  
        w_offset: (k0, i0) - top-left of C block
        
        U_vec elements correspond to A[i0+di, j0+dj] for di,dj in {0,1}
        Order: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1] relative to block
        So: (di,dj) = (0,0), (0,1), (1,0), (1,1)
        
        V_vec for 2x2: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1] relative
        V_vec for 2x1: e=B[0,0], f=B[1,0] relative
        
        W_vec for 2x2: C[0,0], C[0,1], C[1,0], C[1,1] relative
        W_vec for 2x1: C[0,0], C[1,0] relative
        """
        nonlocal rank
        col = rank
        rank += 1
        
        # Map U_vec indices to actual (i,j)
        # a=0->(0,0), b=1->(0,1), c=2->(1,0), d=3->(1,1)
        u_map = [(0,0), (0,1), (1,0), (1,1)]
        for idx, val in enumerate(U_vec):
            if val != 0:
                di, dj = u_map[idx]
                i = u_offset[0] + di
                j = u_offset[1] + dj
                u_idx = i * 4 + j
                U_dict[(u_idx, col)] = U_dict.get((u_idx, col), 0) + val
        
        # Map V_vec indices
        if len(V_vec) == 4:
            # 2x2 B block: e=0->(0,0), f=1->(0,1), g=2->(1,0), h=3->(1,1)
            v_map = [(0,0), (0,1), (1,0), (1,1)]
        else:
            # 2x1 B block: e=0->(0,0), f=1->(1,0)
            v_map = [(0,0), (1,0)]
        for idx, val in enumerate(V_vec):
            if val != 0:
                dj, dk = v_map[idx]
                j = v_offset[0] + dj
                k = v_offset[1] + dk
                v_idx = j * 5 + k
                V_dict[(v_idx, col)] = V_dict.get((v_idx, col), 0) + val
        
        # Map W_vec indices
        if len(W_vec) == 4:
            # 2x2 C block: C[0,0], C[0,1], C[1,0], C[1,1]
            w_map = [(0,0), (0,1), (1,0), (1,1)]
        else:
            # 2x1 C block: C[0,0], C[1,0]
            w_map = [(0,0), (1,0)]
        for idx, val in enumerate(W_vec):
            if val != 0:
                dk, di = w_map[idx]
                k = w_offset[0] + dk
                i = w_offset[1] + di
                w_idx = k * 4 + i
                W_dict[(w_idx, col)] = W_dict.get((w_idx, col), 0) + val
    
    strassen_terms = _strassen_222()
    schoolbook_terms = _schoolbook_221()
    
    # Block (i,j) where i is row-block of A (0 or 1), j is col-block of B (0 or 1)
    # A block: rows i*2 to i*2+1, cols j*2 to j*2+1 -> u_offset = (i*2, j*2)
    # B block: rows j*2 to j*2+1, cols depend on which C block
    # C block: rows i*2 to i*2+1, cols depend
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # For C_ij = A_ik * B_kj where k sums over block rows
    
    # Let me enumerate all 8 terms (4 blocks * 2 k-values):
    # (i, k, j) where i in {0,1}, k in {0,1}, j in {0,1}
    # A block: (i, k) -> u_offset = (i*2, k*2)
    # B block: (k, j) -> v_offset = (k*2, j*2) for left 2 cols, (k*2, j*2+2) for... 
    # Wait, B is 4x5. B[k*2:(k+1)*2, ?]
    # B block for C_ij: B rows k*2 to k*2+1, cols j*3 to j*3+2 (since C has 3-col blocks)
    # Hmm, but 5 doesn't split evenly into 2 blocks of 2.5.
    
    # Let me reconsider. B is 4x5. Split into 2 row-blocks of 2x3 and 2x2?
    # Or 2x2 and 2x3?
    
    # Actually, let me split B's columns as: first 3 cols, last 2 cols.
    # B[0:2, 0:3] = B00 (2x3), B[0:2, 3:5] = B01 (2x2)
    # B[2:4, 0:3] = B10 (2x3), B[2:4, 3:5] = B11 (2x2)
    
    # C[0:2, 0:3] = C00 (2x3), C[0:2, 3:5] = C01 (2x2)
    # C[2:4, 0:3] = C10 (2x3), C[2:4, 3:5] = C11 (2x2)
    
    # C00 = A00*B00 + A01*B10 (2x2 * 2x3 + 2x2 * 2x3)
    # C01 = A00*B01 + A01*B11 (2x2 * 2x2 + 2x2 * 2x2)
    # C10 = A10*B00 + A11*B10 (2x2 * 2x3 + 2x2 * 2x3)
    # C11 = A10*B01 + A11*B11 (2x2 * 2x2 + 2x2 * 2x2)
    
    # For 2x2 * 2x3: split B into 2x2 + 2x1, use Strassen(7) + schoolbook(4) = 11
    # For 2x2 * 2x2: use Strassen(7)
    
    # 2x2*2x3 terms: 4 such multiplications (C00 has 2, C10 has 2) -> 4 * 11 = 44
    # 2x2*2x2 terms: 4 such multiplications (C01 has 2, C11 has 2) -> 4 * 7 = 28
    # Total: 44 + 28 = 72
    
    # Hmm, that's only slightly better than 80. Let me reconsider.
    
    # Actually wait, I miscounted. Let me recount:
    # C00 = A00*B00 + A01*B10: two 2x2*2x3 multiplications
    # C10 = A10*B00 + A11*B10: two 2x2*2x3 multiplications
    # Total 2x2*2x3: 4
    # C01 = A00*B01 + A01*B11: two 2x2*2x2 multiplications
    # C11 = A10*B01 + A11*B11: two 2x2*2x2 multiplications
    # Total 2x2*2x2: 4
    
    # 4*11 + 4*7 = 44 + 28 = 72. Still not great.
    
    # Alternative: maybe I should use a different splitting. What if I don't split B's columns?
    # Instead, treat each 2x2 * 2x3 as a single tensor and find its minimal decomposition.
    # The 2x2*2x3 tensor has rank at most 12 (schoolbook), and possibly lower.
    # Actually, the rank of the 2x2*2x3 tensor is known to be 12 (it's the same as 2x2*2x2 
    # padded, but actually it might be less). Let me just use schoolbook for 2x2*2x3 (rank 12)
    # and Strassen for 2x2*2x2 (rank 7).
    
    # 4*12 + 4*7 = 48 + 28 = 76. Worse!
    
    # Let me go back to 4*11 + 4*7 = 72.
    
    # Actually, can I do better on the 2x2*2x3? The rank of 2x2*2x3 matrix multiplication 
    # tensor is actually 12 (I believe it's not known to be less). But my Strassen+schoolbook 
    # gives 11, which should be valid.
    
    # Let me just go with rank 72 for now. Actually wait, let me double-check my Strassen+schoolbook 
    # for 2x2*2x3.
    
    # A (2x2) * B (2x3) = C (2x3)
    # B = [B_left | B_right] where B_left is 2x2, B_right is 2x1
    # A * B = A * B_left + A * B_right
    # A * B_left: 2x2*2x2, Strassen rank 7
    # A * B_right: 2x2*2x1, schoolbook rank 4
    # Total: 11. Yes.
    
    # So total rank = 4*11 + 4*7 = 72.
    
    # Let me implement this.
    
    # Clear and restart
    U_dict = {}
    V_dict = {}
    W_dict = {}
    rank = 0
    
    def add_term(U_vec, V_vec, W_vec, u_offset, v_offset, w_offset):
        nonlocal rank
        col = rank
        rank += 1
        
        u_map = [(0,0), (0,1), (1,0), (1,1)]
        for idx, val in enumerate(U_vec):
            if val != 0:
                di, dj = u_map[idx]
                i = u_offset[0] + di
                j = u_offset[1] + dj
                u_idx = i * 4 + j
                U_dict[(u_idx, col)] = U_dict.get((u_idx, col), 0) + val
        
        if len(V_vec) == 4:
            v_map = [(0,0), (0,1), (1,0), (1,1)]
        else:
            v_map = [(0,0), (1,0)]
        for idx, val in enumerate(V_vec):
            if val != 0:
                dj, dk = v_map[idx]
                j = v_offset[0] + dj
                k = v_offset[1] + dk
                v_idx = j * 5 + k
                V_dict[(v_idx, col)] = V_dict.get((v_idx, col), 0) + val
        
        if len(W_vec) == 4:
            w_map = [(0,0), (0,1), (1,0), (1,1)]
        else:
            w_map = [(0,0), (1,0)]
        for idx, val in enumerate(W_vec):
            if val != 0:
                dk, di = w_map[idx]
                k = w_offset[0] + dk
                i = w_offset[1] + di
                w_idx = k * 4 + i
                W_dict[(w_idx, col)] = W_dict.get((w_idx, col), 0) + val
    
    # Process each (i, k, j) combination
    # i: A row block (0 or 1)
    # k: A col block / B row block (0 or 1)
    # j: B col block / C col block (0 or 1)
    
    for i in range(2):
        for k in range(2):
            for j in range(2):
                # A block at (i, k): u_offset = (i*2, k*2)
                u_off = (i*2, k*2)
                # B block at (k, j): 
                #   j=0: B[k*2:k*2+2, 0:3] -> v_offset = (k*2, 0), 2x3
                #   j=1: B[k*2:k*2+2, 3:5] -> v_offset = (k*2, 3), 2x2
                # C block at (i, j):
                #   j=0: C[i*2:i*2+2, 0:3] -> w_offset = (0, i*2) [k_offset=0, i_offset=i*2]
                #   j=1: C[i*2:i*2+2, 3:5] -> w_offset = (3, i*2) [k_offset=3, i_offset=i*2]
                
                if j == 0:
                    # 2x2 * 2x3
                    v_off = (k*2, 0)
                    w_off = (0, i*2)  # k_offset=0, i_offset=i*2
                    
                    # Strassen part: A * B_left where B_left = B[k*2:k*2+2, 0:2]
                    # v_off for B_left: (k*2, 0)
                    # w_off for C_left: C[i*2:i*2+2, 0:2] -> (0, i*2)
                    for U_vec, V_vec, W_vec in strassen_terms:
                        add_term(U_vec, V_vec, W_vec, u_off, (k*2, 0), (0, i*2))
                    
                    # Schoolbook part: A * B_right where B_right = B[k*2:k*2+2, 2:3]
                    # v_off for B_right: (k*2, 2)
                    # w_off for C_right: C[i*2:i*2+2, 2:3] -> (2, i*2)
                    for U_vec, V_vec, W_vec in schoolbook_terms:
                        add_term(U_vec, V_vec, W_vec, u_off, (k*2, 2), (2, i*2))
                else:
                    # 2x2 * 2x2
                    v_off = (k*2, 3)
                    w_off = (3, i*2)  # k_offset=3, i_offset=i*2
                    
                    for U_vec, V_vec, W_vec in strassen_terms:
                        add_term(U_vec, V_vec, W_vec, u_off, v_off, w_off)
    
    # Convert dicts to matrices
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(20)]
    W = [[0] * rank for _ in range(20)]
    
    for (u_idx, col), val in U_dict.items():
        U[u_idx][col] = val
    for (v_idx, col), val in V_dict.items():
        V[v_idx][col] = val
    for (w_idx, col), val in W_dict.items():
        W[w_idx][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
