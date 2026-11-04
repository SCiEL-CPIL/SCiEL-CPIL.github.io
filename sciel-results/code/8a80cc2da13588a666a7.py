import time
import sys
import numpy as np

# EVOLVE-BLOCK-START

def _strassen_2x2_2x2_UVW():
    """
    Return Strassen's 7-term decomposition for 2x2 x 2x2 -> 2x2 matrix multiplication.
    U: shape (4, 7) for A[0,0], A[0,1], A[1,0], A[1,1]
    V: shape (4, 7) for B[0,0], B[0,1], B[1,0], B[1,1]
    W: shape (4, 7) for C[0,0], C[0,1], C[1,0], C[1,1]
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],  # a00
        [0, 0, 0, 1, 0, 0, -1],  # a01
        [0, 1, 0, 0, 0, 1, 0],   # a10
        [1, 0, 0, 1, 0, 0, 1],   # a11
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],  # b00
        [0, 0, 1, 0, 0, 1, 0],   # b01
        [1, 0, -1, 0, 1, 0, 1],  # b10
        [1, 0, 0, 0, 1, 0, 1],   # b11
    ]
    W = [
        [1, 0, 0, -1, 0, 1, 0],  # C00 = M1 - M4 + M6
        [0, 0, 1, 0, 1, 0, 0],   # C01 = M3 + M5
        [0, 1, 0, 1, 0, 0, 0],   # C10 = M2 + M4
        [1, 0, -1, 0, 0, 0, 1],  # C11 = M1 - M3 + M7
    ]
    return np.array(U), np.array(V), np.array(W)

def _block_4x4x7_UVW():
    """
    Build rank-97 decomposition for 4x4x7.
    Part 1: 4x4x4 using recursive Strassen (rank 49).
    Part 2: 4x4x3 using trivial (rank 48).
    """
    n, m, p = 4, 4, 7
    total_rank = 97
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    # --- Part 1: 4x4x4 (Rank 49) ---
    U222, V222, W222 = _strassen_2x2_2x2_UVW()
    
    # Macro Strassen coefficients
    # u_s: coeffs for A blocks (A00, A01, A10, A11)
    # v_s: coeffs for B blocks (B00, B01, B10, B11)
    # w_s: coeffs for C blocks (C00, C01, C10, C11)
    macro_params = [
        # s=0: M1 = (A00+A11)(B00+B11) -> C00+1, C11+1
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        # s=1: M2 = (A10+A11)B00 -> C10+1
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, 0]},
        # s=2: M3 = A00(B01-B11) -> C01+1, C11-1
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, -1]},
        # s=3: M4 = A11(B10-B00) -> C00-1, C10+1
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [-1, 0, 1, 0]},
        # s=4: M5 = (A00+A01)B11 -> C01+1
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [0, 1, 0, 0]},
        # s=5: M6 = (A10-A00)(B00+B01) -> C00+1
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [1, 0, 0, 0]},
        # s=6: M7 = (A01-A11)(B10+B11) -> C11+1
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [0, 0, 0, 1]},
    ]
    
    # Block coordinates (row, col) for 2x2 blocks in 4x4 matrix
    block_coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for s_idx, params in enumerate(macro_params):
        for t_idx in range(7):
            r = s_idx * 7 + t_idx
            
            # Fill U
            for a_blk in range(4):
                a_coeff = params['u'][a_blk]
                if a_coeff == 0: continue
                br, bc = block_coords[a_blk]
                for lr in range(2):
                    for lc in range(2):
                        global_i = br * 2 + lr
                        global_j = bc * 2 + lc
                        u_row = global_i * 4 + global_j
                        # U222 index for local (lr, lc)
                        local_idx = lc * 2 + lr
                        U[u_row, r] += a_coeff * U222[local_idx, t_idx]
                        
            # Fill V
            for b_blk in range(4):
                b_coeff = params['v'][b_blk]
                if b_coeff == 0: continue
                br, bc = block_coords[b_blk]
                for lr in range(2):
                    for lc in range(2):
                        global_j = br * 2 + lr # B row
                        global_k = bc * 2 + lc # B col
                        v_row = global_j * 7 + global_k # m=4, p=7. V row index j*p+k
                        local_idx = lc * 2 + lr
                        V[v_row, r] += b_coeff * V222[local_idx, t_idx]
                        
            # Fill W
            for c_blk in range(4):
                c_coeff = params['w'][c_blk]
                if c_coeff == 0: continue
                br, bc = block_coords[c_blk]
                for lr in range(2):
                    for lc in range(2):
                        global_i = br * 2 + lr # C row
                        global_k = bc * 2 + lc # C col
                        w_row = global_k * 4 + global_i # p=7, n=4. W row index k*n+i
                        local_idx = lc * 2 + lr
                        W[w_row, r] += c_coeff * W222[local_idx, t_idx]
                        
    # --- Part 2: 4x4x3 (Rank 48) ---
    # Trivial decomposition
    # A: 4x4, B: 4x3, C: 4x3
    # B cols 4-6, C cols 4-6
    offset = 49
    for i in range(4):
        for j in range(4):
            for k in range(3):
                r = offset + (i * 4 + j) * 3 + k
                u_row = i * 4 + j
                v_row = j * 7 + (4 + k)
                w_row = (4 + k) * 4 + i
                U[u_row, r] = 1.0
                V[v_row, r] = 1.0
                W[w_row, r] = 1.0
                
    return U.tolist(), V.tolist(), W.tolist()

def _trivial_decomposition(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][t] = 1
                V[v_idx][t] = 1
                W[w_idx][t] = 1
                t += 1
    return U, V, W

def _verify_decomposition(n, m, p, U, V, W):
    try:
        np.random.seed(12345)
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for trial in range(2):
            A = np.random.randint(-5, 6, (n, m), dtype=float)
            B = np.random.randint(-5, 6, (m, p), dtype=float)
            C_expected = A @ B
            
            C_computed = np.zeros((n, p), dtype=float)
            for i in range(n):
                for k in range(p):
                    w_row = W_arr[k * n + i, :]
                    s = 0.0
                    for j in range(m):
                        u_row = U_arr[i * m + j, :]
                        v_row = V_arr[j * p + k, :]
                        s += np.dot(u_row * v_row, w_row)
                    C_computed[i, k] = s
            
            if not np.allclose(C_computed, C_expected, atol=1e-8):
                return False
        return True
    except Exception:
        return False

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 4, 4, 7
    
    try:
        if time.time() < deadline - 30:
            U, V, W = _block_4x4x7_UVW()
            if time.time() < deadline - 20:
                if _verify_decomposition(n, m, p, U, V, W):
                    return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    U, V, W = _trivial_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
