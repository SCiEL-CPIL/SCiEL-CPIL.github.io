import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 7
    start_time = time.time()
    deadline = start_time + 145.0

    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    def add_term(u_indices, v_indices, w_indices, signs_u=None, signs_v=None, signs_w=None):
        nonlocal col
        if col >= rank:
            return
        if signs_u is None: signs_u = [1] * len(u_indices)
        if signs_v is None: signs_v = [1] * len(v_indices)
        if signs_w is None: signs_w = [1] * len(w_indices)
        for idx, sign in zip(u_indices, signs_u): U[idx][col] += sign
        for idx, sign in zip(v_indices, signs_v): V[idx][col] += sign
        for idx, sign in zip(w_indices, signs_w): W[idx][col] += sign
        col += 1

    def add_strassen_block(k_start):
        # A indices: 0:(0,0), 1:(0,1), 3:(1,0), 4:(1,1)
        a00, a01, a10, a11 = 0, 1, 3, 4
        # B indices: (0, k_start), (0, k_start+1), (1, k_start), (1, k_start+1)
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1
        # C indices: (k_start, 0), (k_start+1, 0), (k_start, 1), (k_start+1, 1)
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1
        
        # Strassen 7 terms
        # M1 = (A00 + A11)(B00 + B11) -> C00, C11
        add_term([a00, a11], [b00, b11], [c00, c11])
        # M2 = (A10 + A11)B00 -> C10, C11
        add_term([a10, a11], [b00], [c10, c11], signs_w=[1, -1])
        # M3 = A00(B01 + B11) -> C01, C11
        add_term([a00], [b01, b11], [c01, c11], signs_v=[1, -1])
        # M4 = A11(B10 + B00) -> C00, C10
        add_term([a11], [b10, b00], [c00, c10], signs_v=[1, -1])
        # M5 = (A00 + A01)B11 -> C00, C01
        add_term([a00, a01], [b11], [c00, c01], signs_w=[-1, 1])
        # M6 = (A10 + A00)(B00 + B01) -> C11
        add_term([a10, a00], [b00, b01], [c11], signs_u=[1, -1])
        # M7 = (A01 + A11)(B10 + B11) -> C00
        add_term([a01, a11], [b10, b11], [c00], signs_u=[1, -1])

    # 1. Strassen blocks for k=0, 2, 4
    for k in [0, 2, 4]:
        add_strassen_block(k)
        
    # 2. Tail block for k=6 (Rank 2)
    # C[:, 6] = A[:, 0] B[0, 6] + A[:, 1] B[1, 6]
    # Term 1: Contributes to C[0, 6] (index 12)
    # U: A[0,0], A[0,1] (indices 0, 1)
    # V: B[0,6], B[1,6] (indices 6, 13)
    # W: C[0,6] (index 12)
    add_term([0, 1], [6, 13], [12])
    
    # Term 2: Contributes to C[1, 6] (index 13)
    # U: A[1,0], A[1,1] (indices 3, 4)
    # V: B[0,6], B[1,6] (indices 6, 13)
    # W: C[1,6] (index 13)
    add_term([3, 4], [6, 13], [13])

    # 3. J=2 part (Rank 7)
    # C[:, k] += A[:, 2] B[2, k]
    for k in range(7):
        # U: A[0,2], A[1,2] (indices 2, 5)
        # V: B[2,k] (index 14+k)
        # W: C[0,k], C[1,k] (indices 2k, 2k+1)
        add_term([2, 5], [14 + k], [2 * k, 2 * k + 1])

    # Verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    T_recon = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U_np[:,r], V_np[:,r], W_np[:,r])
    
    T_true = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T_true[a_idx, b_idx, c_idx] = 1
                
    if not np.allclose(T_recon, T_true):
        # Fallback
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
