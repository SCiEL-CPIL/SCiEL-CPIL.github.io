import time
import numpy as np

DIMENSIONS = [2, 4, 4]
_START_TIME = time.time()
_TIME_LIMIT = 145  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # Rank = 28 (Strassen-based block decomposition)
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Output C:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Local indices:
    # A: 00:0, 01:1, 10:2, 11:3
    # B: 00:0, 01:1, 10:2, 11:3
    # C: 00:0, 01:1, 10:2, 11:3
    
    strassen_steps = [
        # M1
        {
            'u': {0: 1, 3: 1},
            'v': {0: 1, 3: 1},
            'w': {0: 1, 3: 1}
        },
        # M2
        {
            'u': {2: 1, 3: 1},
            'v': {0: 1},
            'w': {2: 1, 3: -1}
        },
        # M3
        {
            'u': {0: 1},
            'v': {1: 1, 3: -1},
            'w': {1: 1, 3: 1}
        },
        # M4
        {
            'u': {3: 1},
            'v': {2: 1, 0: -1},
            'w': {0: 1, 2: 1}
        },
        # M5
        {
            'u': {0: 1, 1: 1},
            'v': {3: 1},
            'w': {0: -1, 1: 1}
        },
        # M6
        {
            'u': {2: 1, 0: -1},
            'v': {0: 1, 1: 1},
            'w': {3: 1}
        },
        # M7
        {
            'u': {1: 1, 3: -1},
            'v': {2: 1, 3: 1},
            'w': {0: 1}
        }
    ]
    
    # Helper to add a block
    def add_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
        for s in range(7):
            step = strassen_steps[s]
            col = col_offset + s
            
            # Update U
            for local_idx, coeff in step['u'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_i = l_row
                global_j = a_col_start + l_col
                global_idx = global_i * m + global_j
                U[global_idx][col] += coeff
                
            # Update V
            for local_idx, coeff in step['v'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_i = b_row_start + l_row
                global_j = b_col_start + l_col
                global_idx = global_i * p + global_j
                V[global_idx][col] += coeff
                
            # Update W
            for local_idx, coeff in step['w'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                # W index convention: k*n + i
                # k is col of C, i is row of C
                global_k = c_col_start + l_col
                global_i = l_row
                global_idx = global_k * n + global_i
                W[global_idx][col] += coeff

    # Block 1: A_L (cols 0,1), B_TL (rows 0,1 cols 0,1) -> C_L (cols 0,1)
    add_block(a_col_start=0, b_row_start=0, b_col_start=0, c_col_start=0, col_offset=0)
    
    # Block 2: A_L (cols 0,1), B_TR (rows 0,1 cols 2,3) -> C_R (cols 2,3)
    add_block(a_col_start=0, b_row_start=0, b_col_start=2, c_col_start=2, col_offset=7)
    
    # Block 3: A_R (cols 2,3), B_BL (rows 2,3 cols 0,1) -> C_L (cols 0,1)
    add_block(a_col_start=2, b_row_start=2, b_col_start=0, c_col_start=0, col_offset=14)
    
    # Block 4: A_R (cols 2,3), B_BR (rows 2,3 cols 2,3) -> C_R (cols 2,3)
    add_block(a_col_start=2, b_row_start=2, b_col_start=2, c_col_start=2, col_offset=21)
    
    # Verify correctness with numpy
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: sum_r U[i,r]*V[j,r] should give the (i,j) entry of the multiplication tensor
        # For matrix mult: T[i,j,k] = 1 if i is row of A, j is col of B, k is (row,col) of C
        # Actually: C[i,k] = sum_j A[i,j]*B[j,k]
        # Tensor: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta (correct mult)
        
        # Verify with random test
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_direct = A_test @ B_test
        
        # Compute via tensor decomposition
        C_tensor = np.zeros((n, p))
        for r in range(rank):
            A_u = U_np[:, r].reshape(n, m)
            B_v = V_np[:, r].reshape(m, p)
            C_w = W_np[:, r].reshape(p, n).T  # W[k*n+i] -> reshape to (n,p): C_w[i,k]
            C_tensor += np.sum(A_u * B_v, axis=1)[:, None] * C_w  # This isn't right
            
        # Better: C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Let's just verify the tensor directly
        # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
        # C[i, k] = sum_j A[i,j] B[j,k]
        # Decomposition: C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k]) * W[k*n+i, r]
        # Wait no: C[i,k] = sum_r U_vec_r . A_vec * V_vec_r . B_vec * W[k*n+i, r]
        # where U_vec_r is length n*m, V_vec_r is length m*p
        
        # Actually the standard tensor decomposition:
        # C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
        # No wait. The tensor T for matrix multiplication has T[i,j,k] = delta (it's 1 for all valid i,j,k)
        # The decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # where U has shape (n, m, rank), V has shape (m, p, rank), W has shape (p, n, rank)
        
        # In our flat indexing:
        # U_flat[i*m+j, r] = U[i,j,r]
        # V_flat[j*p+k, r] = V[j,k,r]  
        # W_flat[k*n+i, r] = W[k,i,r]
        
        # Check: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should equal 1
        # (since the multiplication tensor is all ones in the valid range)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err > 1e-9:
            # Fallback to trivial rank-32
            rank32 = n * m * p
            U32 = [[0] * rank32 for _ in range(n * m)]
            V32 = [[0] * rank32 for _ in range(m * p)]
            W32 = [[0] * rank32 for _ in range(p * n)]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = i * m * p + j * p + k
                        U32[i*m+j][r] = 1
                        V32[j*p+k][r] = 1
                        W32[k*n+i][r] = 1
            return {"U": U32, "V": V32, "W": W32}
    except Exception:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
