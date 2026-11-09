import numpy as np
import time

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds margin
    
    n, m, p = DIMENSIONS
    
    # If dimensions are not 4,4,4, fallback to schoolbook
    if n != 4 or m != 4 or p != 4:
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Strassen Decomposition for 4x4
    # Recursive helper
    def get_strassen_matrices(size):
        if size == 1:
            return np.array([[1]]), np.array([[1]]), np.array([[1]])
        
        half = size // 2
        U_sub, V_sub, W_sub = get_strassen_matrices(half)
        rank_sub = U_sub.shape[1]
        rank_new = 7 * rank_sub
        
        U_new = np.zeros((size * size, rank_new), dtype=int)
        V_new = np.zeros((size * size, rank_new), dtype=int)
        W_new = np.zeros((size * size, rank_new), dtype=int)
        
        # Define the 7 Strassen steps
        # Each step k has:
        # A_coeffs: list of (row_start, col_start, sign) for A quadrants
        # B_coeffs: list of (row_start, col_start, sign) for B quadrants
        # C_coeffs: list of (row_start, col_start, sign) for C quadrants (how Z_k adds to C)
        
        # Quadrants:
        # 0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right
        # Indices in flattened matrix of size 'size':
        # TL: 0..half-1, 0..half-1
        # TR: 0..half-1, half..size-1
        # BL: half..size-1, 0..half-1
        # BR: half..size-1, half..size-1
        
        # We need to build L_A, L_B, L_C for each step.
        # L_A is (half*half) x (size*size)
        
        steps = [
            # k=1: (A11+A22)(B11+B22) -> C11+1, C12+0, C21+0, C22+1? 
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            # 
            # M1 contributes to C11 (+1) and C22 (+1)
            {
                'A': [(0, 1), (3, 1)], # A11 + A22
                'B': [(0, 1), (3, 1)], # B11 + B22
                'C': [(0, 1), (3, 1)]  # C11 + C22
            },
            # M2: (A21+A22)B11 -> C21+1, C22-1
            {
                'A': [(2, 1), (3, 1)],
                'B': [(0, 1)],
                'C': [(2, 1), (3, -1)]
            },
            # M3: A11(B12-B22) -> C12+1, C22+1
            {
                'A': [(0, 1)],
                'B': [(1, 1), (3, -1)],
                'C': [(1, 1), (3, 1)]
            },
            # M4: A22(B21-B11) -> C11+1, C21+1
            {
                'A': [(3, 1)],
                'B': [(2, 1), (0, -1)],
                'C': [(0, 1), (2, 1)]
            },
            # M5: (A11+A12)B22 -> C11-1, C12+1
            {
                'A': [(0, 1), (1, 1)],
                'B': [(3, 1)],
                'C': [(0, -1), (1, 1)]
            },
            # M6: (A21-A11)(B11+B12) -> C22+1
            {
                'A': [(2, 1), (0, -1)],
                'B': [(0, 1), (1, 1)],
                'C': [(3, 1)]
            },
            # M7: (A12-A22)(B21+B22) -> C11+1
            {
                'A': [(1, 1), (3, -1)],
                'B': [(2, 1), (3, 1)],
                'C': [(0, 1)]
            }
        ]
        
        for k, step in enumerate(steps):
            # Build L_A: (half*half) x (size*size)
            L_A = np.zeros((half * half, size * size), dtype=int)
            for q_idx, sign in step['A']:
                # q_idx 0: TL, 1: TR, 2: BL, 3: BR
                r_start = (q_idx // 2) * half
                c_start = (q_idx % 2) * half
                # Extract submatrix indices
                for r in range(half):
                    for c in range(half):
                        global_r = r_start + r
                        global_c = c_start + c
                        global_idx = global_r * size + global_c
                        local_idx = r * half + c
                        L_A[local_idx, global_idx] = sign
            
            # Build L_B
            L_B = np.zeros((half * half, size * size), dtype=int)
            for q_idx, sign in step['B']:
                r_start = (q_idx // 2) * half
                c_start = (q_idx % 2) * half
                for r in range(half):
                    for c in range(half):
                        global_r = r_start + r
                        global_c = c_start + c
                        global_idx = global_r * size + global_c
                        local_idx = r * half + c
                        L_B[local_idx, global_idx] = sign
            
            # Build L_C
            L_C = np.zeros((size * size, half * half), dtype=int)
            for q_idx, sign in step['C']:
                r_start = (q_idx // 2) * half
                c_start = (q_idx % 2) * half
                for r in range(half):
                    for c in range(half):
                        global_r = r_start + r
                        global_c = c_start + c
                        global_idx = global_r * size + global_c
                        local_idx = r * half + c
                        L_C[global_idx, local_idx] = sign
            
            # Compute blocks
            offset = k * rank_sub
            U_new[:, offset:offset+rank_sub] = L_A @ U_sub
            V_new[:, offset:offset+rank_sub] = L_B @ V_sub
            W_new[:, offset:offset+rank_sub] = L_C @ W_sub
            
        return U_new, V_new, W_new

    U, V, W = get_strassen_matrices(4)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
