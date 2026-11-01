import time
import numpy as np

# Dimensions: A is 2x2, B is 2x16, C is 2x16
# Tensor M_{2,2,16}
# Decomposed into 8 blocks of M_{2,2,2}
# Rank of M_{2,2,2} is 7 (Strassen)
# Total Rank = 8 * 7 = 56

def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    n, m, p = 2, 2, 16
    blocks = p // 2
    total_rank = blocks * 7

    # Initialize matrices
    U = np.zeros((n * m, total_rank), dtype=np.int8)
    V = np.zeros((m * p, total_rank), dtype=np.int8)
    W = np.zeros((p * n, total_rank), dtype=np.int8)

    # Strassen coefficients for 2x2 matrix multiplication
    # Indices:
    # U rows: 0:A00, 1:A01, 2:A10, 3:A11
    # V cols: 0:B00, 1:B01, 2:B10, 3:B11 (local to block)
    # W rows: 0:C00, 1:C01, 2:C10, 3:C11 (local to block)
    
    # s: 0..6
    # U_s: dict {row: val}
    # V_s: dict {col: val}
    # W_s: dict {row: val}
    
    strassen = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}), # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}), # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}), # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}), # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}), # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})  # M7
    ]

    for q in range(blocks):
        # Global indices for this block
        # B indices: row 0: 2q, 2q+1; row 1: p+2q, p+2q+1
        b_idx_00 = 0 * p + 2 * q
        b_idx_01 = 0 * p + 2 * q + 1
        b_idx_10 = 1 * p + 2 * q
        b_idx_11 = 1 * p + 2 * q + 1
        b_map = {0: b_idx_00, 1: b_idx_01, 2: b_idx_10, 3: b_idx_11}

        # C indices: W[k*n + i]
        # k is column (2q, 2q+1), i is row (0, 1)
        c_idx_00 = (2 * q) * n + 0
        c_idx_01 = (2 * q + 1) * n + 0
        c_idx_10 = (2 * q) * n + 1
        c_idx_11 = (2 * q + 1) * n + 1
        c_map = {0: c_idx_00, 1: c_idx_01, 2: c_idx_10, 3: c_idx_11}

        for s in range(7):
            r = q * 7 + s
            u_dict, v_dict, w_dict = strassen[s]

            # Fill U
            for u_row, val in u_dict.items():
                U[u_row, r] = val

            # Fill V
            for v_col, val in v_dict.items():
                V[b_map[v_col], r] = val

            # Fill W
            for w_row, val in w_dict.items():
                W[c_map[w_row], r] = val

    # Quick verification
    if time.time() < deadline - 5:
        np.random.seed(123)
        A = np.random.randint(-3, 3, size=(n, m))
        B = np.random.randint(-3, 3, size=(m, p))
        C = A @ B
        
        Af = A.flatten()
        Bf = B.flatten()
        
        # Check if decomposition holds
        # C_decomp[i, k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
        # Vectorized:
        # U_prod = U.T @ Af  (size total_rank)
        # V_prod = V.T @ Bf  (size total_rank)
        # C_flat = W @ (U_prod * V_prod) (size p*n)
        
        u_prod = U.T @ Af
        v_prod = V.T @ Bf
        c_flat = W @ (u_prod * v_prod)
        
        # C is 2x16. W index is k*n+i.
        # C.T.flatten() is [C[0,0], C[1,0], C[0,1], C[1,1], ...]
        c_expected = C.T.flatten()
        
        if not np.array_equal(c_flat, c_expected):
            # Fallback: Schoolbook
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=np.int8)
            V = np.zeros((m * p, rank), dtype=np.int8)
            W = np.zeros((p * n, rank), dtype=np.int8)
            
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
