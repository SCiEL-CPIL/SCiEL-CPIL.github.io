import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n, m, p = 4, 5, 5
    
    # --- Helper: Verify decomposition ---
    def verify(U, V, W, n, m, p):
        r = U.shape[1]
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.allclose(T, 1.0, atol=1e-9)

    # --- Helper: Schoolbook decomposition ---
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # --- Strassen 4x4x4 Decomposition ---
    def get_4x4x4_strassen():
        # Returns U, V, W for 4x4x4 with rank 49
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # Strassen 2x2x2 terms
        # u, v, w are vectors of length 4
        # Indices 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
        s2x2 = [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]), # M1
            ([0,0,1,1], [1,0,0,0], [0,0,1,0]), # M2
            ([1,0,0,0], [0,1,0,-1], [0,1,0,-1]), # M3
            ([0,0,0,1], [0,-1,1,0], [1,0,1,0]), # M4
            ([1,1,0,0], [0,0,0,1], [-1,1,0,0]), # M5
            ([-1,0,1,0], [1,1,0,0], [0,0,0,1]), # M6
            ([0,1,0,-1], [0,0,1,1], [1,0,0,1])  # M7
        ]
        
        # Block indices for 4x4
        # A:
        # 11: 0,1,4,5
        # 12: 2,3,6,7
        # 21: 8,9,12,13
        # 22: 10,11,14,15
        # B: same
        # C: same
        
        # We define the linear combinations for X_k and Y_k
        # M1: X = A11 + A22, Y = B11 + B22
        # M2: X = A21 + A22, Y = B11
        # M3: X = A11, Y = B12 - B22
        # M4: X = A22, Y = B21 - B11
        # M5: X = A11 + A12, Y = B22
        # M6: X = A21 - A11, Y = B11 + B12
        # M7: X = A12 - A22, Y = B21 + B22
        
        # And the contributions to C
        # C11 += M1 + M4 - M5 + M7
        # C12 += M3 + M5
        # C21 += M2 + M4
        # C22 += M1 - M2 + M3 + M6
        
        # We will iterate over the 7 M's and the 7 S's
        # For each pair, we add 1 term to the global U, V, W
        
        # Helper to map local 2x2 index to global 4x4 index
        def local_to_global(block, local_idx):
            # block: 0=11, 1=12, 2=21, 3=22
            # local_idx: 0=00, 1=01, 2=10, 3=11
            # Global row: 2*block_row + local_row
            # Global col: 2*block_col + local_col
            # A/B: row is i, col is j
            # C: row is k, col is i (transposed in W)
            
            block_row = block // 2
            block_col = block % 2
            
            local_row = local_idx // 2
            local_col = local_idx % 2
            
            global_row = 2 * block_row + local_row
            global_col = 2 * block_col + local_col
            
            return global_row, global_col

        # Define the linear combinations
        # Each M_k is defined by a set of (block, coeff) for X and Y
        # And a set of (block, coeff) for the contribution to C blocks
        
        # M1: X: 11:+1, 22:+1. Y: 11:+1, 22:+1. C: 11:+1, 22:+1
        # M2: X: 21:+1, 22:+1. Y: 11:+1. C: 21:+1
        # M3: X: 11:+1. Y: 12:+1, 22:-1. C: 12:+1
        # M4: X: 22:+1. Y: 21:+1, 11:-1. C: 11:+1, 21:+1
        # M5: X: 11:+1, 12:+1. Y: 22:+1. C: 11:-1, 12:+1
        # M6: X: 21:+1, 11:-1. Y: 11:+1, 12:+1. C: 22:+1
        # M7: X: 12:+1, 22:-1. Y: 21:+1, 22:+1. C: 11:+1
        
        m_defs = [
            # X: [(block, coeff)], Y: [(block, coeff)], C: [(block, coeff)]
            ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]), # M1
            ([(2,1), (3,1)], [(0,1)], [(2,1)]), # M2
            ([(0,1)], [(1,1), (3,-1)], [(1,1)]), # M3
            ([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]), # M4
            ([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]), # M5
            ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]), # M6
            ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)]) # M7
        ]
        
        col = 0
        for m_idx, (x_def, y_def, c_def) in enumerate(m_defs):
            for s_idx, (u_s, v_s, w_s) in enumerate(s2x2):
                # u_s, v_s, w_s are length 4
                # We need to expand them into global indices
                # This is tricky because u_s is a linear combination of local elements
                # which are linear combinations of global elements
                
                # Let's build the global vectors
                U_global = np.zeros(16)
                V_global = np.zeros(16)
                W_global = np.zeros(16)
                
                # Expand X
                for block, coeff in x_def:
                    for local_idx in range(4):
                        if u_s[local_idx] != 0:
                            gr, gc = local_to_global(block, local_idx)
                            U_global[gr * 4 + gc] += coeff * u_s[local_idx]
                
                # Expand Y
                for block, coeff in y_def:
                    for local_idx in range(4):
                        if v_s[local_idx] != 0:
                            gr, gc = local_to_global(block, local_idx)
                            V_global[gr * 4 + gc] += coeff * v_s[local_idx]
                
                # Expand C
                for block, coeff in c_def:
                    for local_idx in range(4):
                        if w_s[local_idx] != 0:
                            # C block indices:
                            # C is 4x4.
                            # W[k*n + i, c].
                            # In our 4x4x4 context, n=4, p=4.
                            # W index is k*4 + i.
                            # local_idx in C block corresponds to (k_local, i_local)?
                            # The tensor is T[i,j,k].
                            # W[k*4+i].
                            # So we need k and i.
                            # local_idx in C block:
                            # C block is 2x2.
                            # C11: k=0,1; i=0,1.
                            # C12: k=2,3; i=0,1.
                            # C21: k=0,1; i=2,3.
                            # C22: k=2,3; i=2,3.
                            # local_idx: 0=00, 1=01, 2=10, 3=11.
                            # This local_idx maps to (k_local, i_local)?
                            # In Strassen, C11 = ...
                            # The local indices for C are (row, col) of the 2x2 block.
                            # Row is k? Col is i?
                            # Yes, C is indexed by (k, i) in W.
                            # So local_row = k_local, local_col = i_local.
                            
                            k_local, i_local = local_idx // 2, local_idx % 2
                            
                            # Global k, i
                            block_row = block // 2 # k block
                            block_col = block % 2 # i block
                            
                            k_global = 2 * block_row + k_local
                            i_global = 2 * block_col + i_local
                            
                            W_global[k_global * 4 + i_global] += coeff * w_s[local_idx]
                
                U[:, col] = U_global
                V[:, col] = V_global
                W[:, col] = W_global
                col += 1
                
        return U, V, W

    # --- Construct 4x5x5 ---
    U4, V4, W4 = get_4x4x4_strassen()
    
    # Total rank so far: 49
    # Add terms for a c, A4 b, a d
    
    # a c: 16 terms
    # A4 b: 16 terms
    # a d: 4 terms
    # Total: 49 + 36 = 85
    
    U = np.zeros((20, 85))
    V = np.zeros((25, 85))
    W = np.zeros((20, 85))
    
    # Copy 4x4x4 terms
    # U4 is 16x49. U is 20x85.
    # The 4x4 part uses A[0:16], B[0:16], C[0:16] (indices 0..15)
    # A indices: i*5+j. For 4x4, i=0..3, j=0..3.
    # A[0:16] in 4x4 corresponds to A[0:16] in 4x5?
    # A[0:4, 0:4] -> indices 0,1,2,3, 5,6,7,8, 10,11,12,13, 15,16,17,18.
    # This is NOT contiguous in 4x5.
    # 4x4 A indices:
    # i=0: 0,1,2,3
    # i=1: 5,6,7,8
    # i=2: 10,11,12,13
    # i=3: 15,16,17,18
    # 4x5 A indices:
    # i=0: 0,1,2,3,4
    # i=1: 5,6,7,8,9
    # i=2: 10,11,12,13,14
    # i=3: 15,16,17,18,19
    
    # So we need to map 4x4 indices to 4x5 indices.
    # A_4x4[i, j] -> A_4x5[i, j] (j<4)
    # B_4x4[j, k] -> B_4x5[j, k] (j<4, k<4)
    # C_4x4[k, i] -> C_4x5[k, i] (k<4, i<4)
    
    # U4[i*4+j, c] -> U[i*5+j, c]
    # V4[j*4+k, c] -> V[j*5+k, c]
    # W4[k*4+i, c] -> W[k*4+i, c] (W index is k*n+i, n=4 in 4x4, n=4 in 4x5? No, n=4 in 4x5 too. p=5. W size is p*n = 20. Index k*4+i. Same.)
    
    for c in range(49):
        for i in range(4):
            for j in range(4):
                U[i*5+j, c] = U4[i*4+j, c]
        for j in range(4):
            for k in range(4):
                V[j*5+k, c] = V4[j*4+k, c]
        for k in range(4):
            for i in range(4):
                W[k*4+i, c] = W4[k*4+i, c]
                
    # Add a c terms (16)
    # C[i, k] += a_i c_k
    # U[i*5+4, c] = 1
    # V[4*5+k, c] = 1
    # W[k*4+i, c] = 1
    col = 49
    for i in range(4):
        for k in range(4):
            U[i*5+4, col] = 1
            V[4*5+k, col] = 1
            W[k*4+i, col] = 1
            col += 1
            
    # Add A4 b terms (16)
    # C[i, 4] += sum_j A[i, j] b_j
    # U[i*5+j, c] = 1
    # V[j*5+4, c] = 1
    # W[4*4+i, c] = 1
    for i in range(4):
        for j in range(4):
            U[i*5+j, col] = 1
            V[j*5+4, col] = 1
            W[4*4+i, col] = 1
            col += 1
            
    # Add a d terms (4)
    # C[i, 4] += a_i d
    # U[i*5+4, c] = 1
    # V[4*5+4, c] = 1
    # W[4*4+i, c] = 1
    for i in range(4):
        U[i*5+4, col] = 1
        V[4*5+4, col] = 1
        W[4*4+i, col] = 1
        col += 1
        
    # Verify
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = schoolbook(n, m, p)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
