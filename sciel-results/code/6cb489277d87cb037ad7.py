import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 22
    # 7 (Strassen for j=0,1; k=0,1) + 7 (Strassen for j=0,1; k=2,3) + 8 (Naive for j=2; k=0..3)
    rank = 22
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for 2x2x2
    # M1: (a+d)(e+h) -> u=(1,0,0,1), v=(1,1,0,0), w=(1,0,0,0) ? 
    # Let's define the vectors for Strassen carefully.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1: (a+d)(e+h)
    # M2: (c+d)e
    # M3: a(f-h)
    # M4: d(g-e)
    # M5: (a+b)h
    # M6: (c-a)(e+f)
    # M7: (b-d)(g+h)
    
    # We need to map these to U, V, W.
    # U[i, j, r] corresponds to A[i, j]
    # V[j, k, r] corresponds to B[j, k]
    # W[k, i, r] corresponds to C[k, i] ?? No, W[k*n+i, r] is C[i, k] in standard math?
    # Task says: W[k*n+i, r].
    # Standard matrix mult: C[i, k] = sum_j A[i, j] B[j, k].
    # Tensor T[i, j, k] = 1.
    # Decomposition: sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1.
    # So W[k, i, r] is the coefficient for C[i, k].
    
    # Let's define the 7 terms for a generic 2x2x2 block.
    # Local indices: i_loc in {0,1}, j_loc in {0,1}, k_loc in {0,1}
    
    strassen_terms = [
        # (U_vec, V_vec, W_vec)
        # M1: (a+d)(e+h)
        # U: [1, 0, 1, 0] (a, b, c, d) -> a+d is [1,0,1,0]? No.
        # a is U[0,0], b is U[0,1], c is U[1,0], d is U[1,1].
        # a+d -> U[0,0]=1, U[1,1]=1.
        # e+h -> V[0,0]=1, V[1,1]=1.
        # C00 += 1, C11 += 1.
        # W[0,0]=1, W[1,1]=1.
        ([1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        
        # M2: (c+d)e
        # c+d -> U[1,0]=1, U[1,1]=1.
        # e -> V[0,0]=1.
        # C10 += 1.
        # W[0,1]=1. (C[1,0])
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]), # M2
        
        # M3: a(f-h)
        # a -> U[0,0]=1.
        # f-h -> V[0,1]=1, V[1,1]=-1.
        # C01 += 1.
        # W[1,0]=1. (C[0,1])
        ([1, 0, 0, 0], [0, 1, -1, 0], [0, 1, 0, 0]), # M3
        
        # M4: d(g-e)
        # d -> U[1,1]=1.
        # g-e -> V[1,0]=1, V[0,0]=-1.
        # C10 += 1.
        # W[0,1]=1. (C[1,0])
        ([0, 0, 0, 1], [-1, 0, 1, 0], [0, 1, 0, 0]), # M4
        
        # M5: (a+b)h
        # a+b -> U[0,0]=1, U[0,1]=1.
        # h -> V[1,1]=1.
        # C00 -= 1, C01 += 1.
        # W[0,0]=-1, W[1,0]=1.
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        
        # M6: (c-a)(e+f)
        # c-a -> U[1,0]=1, U[0,0]=-1.
        # e+f -> V[0,0]=1, V[0,1]=1.
        # C11 += 1.
        # W[1,1]=1.
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        
        # M7: (b-d)(g+h)
        # b-d -> U[0,1]=1, U[1,1]=-1.
        # g+h -> V[1,0]=1, V[1,1]=1.
        # C00 += 1, C11 += 1.
        # W[0,0]=1, W[1,1]=1.
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]), # M7
    ]
    
    def add_strassen_block(start_col, j_offset, k_offset):
        for t_idx, (u_vec, v_vec, w_vec) in enumerate(strassen_terms):
            col = start_col + t_idx
            # Fill U
            for i_loc in range(2):
                for j_loc in range(2):
                    idx_u = i_loc * 2 + j_loc
                    # Global j = j_loc + j_offset
                    # Global i = i_loc
                    # U row index: i * m + j
                    row_u = i_loc * m + (j_loc + j_offset)
                    U[row_u, col] = u_vec[idx_u]
            
            # Fill V
            for j_loc in range(2):
                for k_loc in range(2):
                    idx_v = j_loc * 2 + k_loc
                    # Global j = j_loc + j_offset
                    # Global k = k_loc + k_offset
                    # V row index: j * p + k
                    row_v = (j_loc + j_offset) * p + (k_loc + k_offset)
                    V[row_v, col] = v_vec[idx_v]
            
            # Fill W
            for k_loc in range(2):
                for i_loc in range(2):
                    idx_w = k_loc * 2 + i_loc
                    # Global k = k_loc + k_offset
                    # Global i = i_loc
                    # W row index: k * n + i
                    row_w = (k_loc + k_offset) * n + i_loc
                    W[row_w, col] = w_vec[idx_w]
    
    # Block 1: j in {0, 1}, k in {0, 1}
    add_strassen_block(0, 0, 0)
    
    # Block 2: j in {0, 1}, k in {2, 3}
    add_strassen_block(7, 0, 2)
    
    # Block 3: j = 2, k in {0, 1, 2, 3}
    # Naive: 2 * 4 = 8 terms
    col = 14
    for i in range(n): # 0, 1
        for k in range(p): # 0, 1, 2, 3
            # j is fixed at 2
            row_u = i * m + 2
            row_v = 2 * p + k
            row_w = k * n + i
            
            U[row_u, col] = 1
            V[row_v, col] = 1
            W[row_w, col] = 1
            col += 1
            
    # Verification
    # Check shapes
    if U.shape != (6, 22): raise RuntimeError
    if V.shape != (12, 22): raise RuntimeError
    if W.shape != (8, 22): raise RuntimeError
    
    # Check half-integer
    for mat in [U, V, W]:
        if not np.all(np.isclose(2 * mat, np.round(2 * mat), atol=1e-9)):
            raise ValueError("Non-half-integer")
            
    # Check decomposition
    # T[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # Should be 1 for all i, j, k
    
    # Efficient check using einsum
    # U is (6, 22), V is (12, 22), W is (8, 22)
    # Reshape U to (2, 3, 22)
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    # Sum over r
    # Result[i, j, k] = sum_r U_3d[i, j, r] V_3d[j, k, r] W_3d[k, i, r]
    # Using einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        # Debug: print mismatch
        # print(T)
        raise ValueError("Decomposition failed")
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
