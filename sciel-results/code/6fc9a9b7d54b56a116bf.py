"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Constructs a rank-49 decomposition for 4x4 matrix multiplication
    via the tensor (Kronecker) product of the rank-7 Strassen scheme.
    Includes explicit deadline guard and fallback to schoolbook.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 factor matrices (4,7)
    # Rows of U/V correspond to linear indices i*2+j, W corresponds to k*2+i
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Verify Strassen correctness for 2x2 (fast sanity check)
    valid2 = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += U2[i*2+j, c] * V2[j*2+k, c] * W2[k*2+i, c]
                if s != 1:
                    valid2 = False
                    break
            if not valid2:
                break
        if not valid2:
            break
    
    if not valid2:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Check deadline
    if time.time() > deadline:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Construct 4x4 decomposition via Kronecker product
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Permute rows to match linear index convention:
    # U row index: i*4 + j
    # V row index: j*4 + k
    # W row index: k*4 + i
    
    # For np.kron(A, B) where A, B are 4x7:
    # Row index R = i_A * 4 + i_B
    # For U: i_A = i_out*2 + j_out, i_B = i_in*2 + j_in
    # R = (i_out*2 + j_out)*4 + (i_in*2 + j_in)
    #    = i_out*8 + j_out*4 + i_in*2 + j_in
    # Target S = i*4 + j = (i_out*2 + i_in)*4 + (j_out*2 + j_in)
    #          = i_out*8 + i_in*4 + j_out*2 + j_in
    
    # We need perm[S] = R such that U4_fixed[S, :] = U4[perm[S], :]
    perm_u = np.zeros(16, dtype=int)
    for S in range(16):
        i = S // 4
        j = S % 4
        i_out = i // 2
        i_in = i % 2
        j_out = j // 2
        j_in = j % 2
        a_idx = i_out * 2 + j_out
        b_idx = i_in * 2 + j_in
        R = a_idx * 4 + b_idx
        perm_u[S] = R
    
    # For V: row index j*4 + k
    # np.kron(V2, V2) row index R = j_A*4 + j_B
    # j_A corresponds to j_out*2 + k_out? No, V2 rows are indexed by j*2+k for 2x2
    # Wait, V2 has 4 rows indexed by j*2+k where j,k in {0,1}
    # So for V4 = kron(V2, V2), row R = j_A*4 + j_B
    # j_A = j_out*2 + k_out (the "outer" 2x2 index)
    # j_B = j_in*2 + k_in (the "inner" 2x2 index)
    # R = (j_out*2 + k_out)*4 + (j_in*2 + k_in)
    #   = j_out*8 + k_out*4 + j_in*2 + k_in
    # Target S = j*4 + k = (j_out*2 + j_in)*4 + (k_out*2 + k_in)
    #          = j_out*8 + j_in*4 + k_out*2 + k_in
    # So perm_v[S] = R
    
    perm_v = np.zeros(16, dtype=int)
    for S in range(16):
        j = S // 4
        k = S % 4
        j_out = j // 2
        j_in = j % 2
        k_out = k // 2
        k_in = k % 2
        a_idx = j_out * 2 + k_out
        b_idx = j_in * 2 + k_in
        R = a_idx * 4 + b_idx
        perm_v[S] = R
    
    # For W: row index k*4 + i
    # W2 rows indexed by k*2 + i for 2x2
    # W4 = kron(W2, W2), row R = k_A*4 + k_B
    # k_A = k_out*2 + i_out
    # k_B = k_in*2 + i_in
    # R = (k_out*2 + i_out)*4 + (k_in*2 + i_in)
    #   = k_out*8 + i_out*4 + k_in*2 + i_in
    # Target S = k*4 + i = (k_out*2 + k_in)*4 + (i_out*2 + i_in)
    #          = k_out*8 + k_in*4 + i_out*2 + i_in
    # So perm_w[S] = R
    
    perm_w = np.zeros(16, dtype=int)
    for S in range(16):
        k = S // 4
        i = S % 4
        k_out = k // 2
        k_in = k % 2
        i_out = i // 2
        i_in = i % 2
        a_idx = k_out * 2 + i_out
        b_idx = k_in * 2 + i_in
        R = a_idx * 4 + b_idx
        perm_w[S] = R
    
    U4_fixed = U4[perm_u, :]
    V4_fixed = V4[perm_v, :]
    W4_fixed = W4[perm_w, :]
    
    # Check deadline
    if time.time() > deadline:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Verify 4x4 correctness
    valid4 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                u_row = U4_fixed[i*4+j, :]
                v_row = V4_fixed[j*4+k, :]
                w_row = W4_fixed[k*4+i, :]
                s = np.dot(u_row, v_row * w_row)
                if s != 1:
                    valid4 = False
                    break
            if not valid4:
                break
        if not valid4:
            break
    
    if not valid4:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Check deadline one more time
    if time.time() > deadline:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {
        "U": U4_fixed.tolist(),
        "V": V4_fixed.tolist(),
        "W": W4_fixed.tolist()
    }
# EVOLVE-BLOCK-END
