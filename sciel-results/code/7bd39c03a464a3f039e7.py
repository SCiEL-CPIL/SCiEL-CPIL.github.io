import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen terms for 2x2 * 2x2
    # Each term is (u_vec, v_vec, w_vec) where vectors are length 4 representing flattened 2x2 matrices
    # Indexing: [00, 01, 10, 11] -> [0, 1, 2, 3]
    
    S = [
        # M1 = (P00+P11)*(Q00+Q11)
        (np.array([1, 0, 0, 1], dtype=float), np.array([1, 0, 0, 1], dtype=float), np.array([1, 0, 0, 1], dtype=float)),
        # M2 = (P10+P11)*Q00
        (np.array([0, 0, 1, 1], dtype=float), np.array([1, 0, 0, 0], dtype=float), np.array([0, 0, 1, -1], dtype=float)),
        # M3 = P00*(Q01-Q11)
        (np.array([1, 0, 0, 0], dtype=float), np.array([0, 1, 0, -1], dtype=float), np.array([0, 1, 0, 1], dtype=float)),
        # M4 = P11*(Q10-Q00)
        (np.array([0, 0, 0, 1], dtype=float), np.array([0, 0, 1, -1], dtype=float), np.array([1, 0, 1, 0], dtype=float)),
        # M5 = (P00+P01)*Q11
        (np.array([1, 1, 0, 0], dtype=float), np.array([0, 0, 0, 1], dtype=float), np.array([-1, 1, 0, 0], dtype=float)),
        # M6 = (P10-P00)*(Q00+Q01)
        (np.array([-1, 0, 1, 0], dtype=float), np.array([1, 1, 0, 0], dtype=float), np.array([0, 0, 0, 1], dtype=float)),
        # M7 = (P01-P11)*(Q10+Q11)
        (np.array([0, 1, 0, -1], dtype=float), np.array([0, 0, 1, 1], dtype=float), np.array([1, 0, 0, 0], dtype=float)),
    ]
    
    col_idx = 0
    for source in range(2):  # 0: A_left * B_top, 1: A_right * B_bot
        for c in range(3):   # 0, 1, 2: column blocks of B
            for u_vec, v_vec, w_vec in S:
                # Map P (2x2) to U (2x4 slice)
                # source=0: cols 0-1, source=1: cols 2-3
                # u_vec indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                u_rows = np.array([0, 1, 4, 5]) + 2 * source
                U[u_rows, col_idx] = u_vec
                
                # Map Q (2x2) to V (4x6 slice)
                # source=0: rows 0-1, source=1: rows 2-3
                # c: col block 0,1,2 -> cols 2c, 2c+1
                # v_vec indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                # Row 0 of Q -> Row source*2 of B -> V index (source*2)*6 + 2c + 0
                # Row 1 of Q -> Row source*2+1 of B -> V index (source*2+1)*6 + 2c + 0
                v_row_offsets = np.array([0, 0, 1, 1])
                v_col_offsets = np.array([0, 1, 0, 1])
                v_rows = (np.array([source*2, source*2, source*2+1, source*2+1])) * 6 + (2*c) + v_col_offsets
                V[v_rows, col_idx] = v_vec
                
                # Map R (2x2) to W (6x2 slice)
                # R is the result block for C columns 2c, 2c+1
                # W index convention: W[k*n + i] where k is C col, i is C row
                # C row 0 -> i=0, C row 1 -> i=1
                # C col 2c -> k=2c, C col 2c+1 -> k=2c+1
                # w_vec indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> (row, col) of R
                # R[0,0] -> C[0, 2c] -> W[2c*2 + 0]
                # R[0,1] -> C[0, 2c+1] -> W[(2c+1)*2 + 0]
                # R[1,0] -> C[1, 2c] -> W[2c*2 + 1]
                # R[1,1] -> C[1, 2c+1] -> W[(2c+1)*2 + 1]
                
                w_rows = np.array([
                    (2*c) * 2 + 0,       # R[0,0]
                    (2*c + 1) * 2 + 0,   # R[0,1]
                    (2*c) * 2 + 1,       # R[1,0]
                    (2*c + 1) * 2 + 1    # R[1,1]
                ])
                W[w_rows, col_idx] = w_vec
                
                col_idx += 1
                
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break
    
    # Truncate if we hit deadline early (should not happen for rank 42)
    if col_idx < rank:
        U = U[:, :col_idx]
        V = V[:, :col_idx]
        W = W[:, :col_idx]
        rank = col_idx

    # Verification (lightweight)
    # Check that the tensor reconstruction matches the identity tensor for multiplication
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # Target T[i,j,k] = 1 if i%m == j//p and i//m == k%n
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute tensor
    T_comp = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr, optimize=True)
    
    # Target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i_A in range(n):
        for j in range(m):
            for k_B in range(p):
                i = i_A * m + j
                j_idx = j * p + k_B
                k = k_B * n + i_A
                T_target[i, j_idx, k] = 1.0
                
    if not np.allclose(T_comp, T_target, atol=1e-9):
        # Fallback to trivial rank nm*p = 48 if verification fails (should not happen)
        rank = n * m * p
        U = np.eye(n*m)
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        idx = 0
        for i_A in range(n):
            for j in range(m):
                for k_B in range(p):
                    i = i_A * m + j
                    j_idx = j * p + k_B
                    k = k_B * n + i_A
                    U[i, idx] = 1.0
                    V[j_idx, idx] = 1.0
                    W[k, idx] = 1.0
                    idx += 1
        U = U[:, :rank]
        V = V[:, :rank]
        W = W[:, :rank]

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
