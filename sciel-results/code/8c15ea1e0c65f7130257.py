"""Tensor rank minimization for 2x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 110.0

    # Strategy: 
    # Partition J (size 5) into J1={0,1}, J2={2,3}, J3={4}
    # Partition K (size 6) into K1={0,1}, K2={2,3}, K3={4,5}
    # 
    # Sub-problems:
    # 6 blocks of 2x2x2 (J1xK1, J1xK2, J1xK3, J2xK1, J2xK2, J2xK3)
    # Each 2x2x2 uses Strassen rank 7. Total 6*7 = 42.
    # 3 blocks of 2x1x2 (J3xK1, J3xK2, J3xK3)
    # Each 2x1x2 uses standard rank 4. Total 3*4 = 12.
    # Total Rank = 54.

    rank = 54
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            return
        for u_idx, coeff in u_coeffs:
            U[u_idx, col] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx, col] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx, col] += coeff
        col += 1

    # Strassen block for (i in {0,1}, j in {ja,jb}, k in {ka,kb})
    # u indices: i*m + j
    # v indices: j*p + k
    # w indices: k*n + i
    def strassen_block(ja, jb, ka, kb):
        # u
        u0a = 0 * m + ja
        u0b = 0 * m + jb
        u1a = 1 * m + ja
        u1b = 1 * m + jb
        
        # v
        v_a_ka = ja * p + ka
        v_a_kb = ja * p + kb
        v_b_ka = jb * p + ka
        v_b_kb = jb * p + kb
        
        # w
        w_ka_0 = ka * n + 0
        w_ka_1 = ka * n + 1
        w_kb_0 = kb * n + 0
        w_kb_1 = kb * n + 1

        # M1: (A00+A11)(B00+B11) -> C00, C11
        # A00=u0a, A11=u1b, B00=v_a_ka, B11=v_b_kb
        # C00=w_ka_0, C11=w_kb_1
        add_term([(u0a, 1), (u1b, 1)], [(v_a_ka, 1), (v_b_kb, 1)], [(w_ka_0, 1), (w_kb_1, 1)])
        
        # M2: (A10+A11)(B00) -> C10, C11
        # A10=u1a, A11=u1b, B00=v_a_ka
        # C10=w_ka_1, C11=w_kb_1
        add_term([(u1a, 1), (u1b, 1)], [(v_a_ka, 1)], [(w_ka_1, 1), (w_kb_1, -1)])
        
        # M3: (A00)(B01-B11) -> C01, C11
        # A00=u0a, B01=v_a_kb, B11=v_b_kb
        # C01=w_kb_0, C11=w_kb_1
        add_term([(u0a, 1)], [(v_a_kb, 1), (v_b_kb, -1)], [(w_kb_0, 1), (w_kb_1, 1)])
        
        # M4: (A11)(B10-B00) -> C00, C10
        # A11=u1b, B10=v_b_ka, B00=v_a_ka
        # C00=w_ka_0, C10=w_ka_1
        add_term([(u1b, 1)], [(v_b_ka, 1), (v_a_ka, -1)], [(w_ka_0, 1), (w_ka_1, 1)])
        
        # M5: (A00+A01)(B11) -> C00, C01
        # A00=u0a, A01=u0b, B11=v_b_kb
        # C00=w_ka_0, C01=w_kb_0
        add_term([(u0a, 1), (u0b, 1)], [(v_b_kb, 1)], [(w_ka_0, -1), (w_kb_0, 1)])
        
        # M6: (A10-A00)(B00+B01) -> C11
        # A10=u1a, A00=u0a, B00=v_a_ka, B01=v_a_kb
        # C11=w_kb_1
        add_term([(u1a, 1), (u0a, -1)], [(v_a_ka, 1), (v_a_kb, 1)], [(w_kb_1, 1)])
        
        # M7: (A01-A11)(B10+B11) -> C00
        # A01=u0b, A11=u1b, B10=v_b_ka, B11=v_b_kb
        # C00=w_ka_0
        add_term([(u0b, 1), (u1b, -1)], [(v_b_ka, 1), (v_b_kb, 1)], [(w_ka_0, 1)])

    # 6 Strassen blocks for the 2x4x6 part
    # J1={0,1}, J2={2,3}
    # K1={0,1}, K2={2,3}, K3={4,5}
    
    strassen_block(0, 1, 0, 1) # J1, K1
    strassen_block(0, 1, 2, 3) # J1, K2
    strassen_block(0, 1, 4, 5) # J1, K3
    strassen_block(2, 3, 0, 1) # J2, K1
    strassen_block(2, 3, 2, 3) # J2, K2
    strassen_block(2, 3, 4, 5) # J2, K3

    # 3 Standard blocks for the 2x1x6 part (J3={4})
    # Each 2x1x2 block has rank 4.
    # For j=4, k in {ka, kb}, i in {0,1}
    # We need to cover T[i, 4, k] = 1.
    # Standard multiplication: T[i, 4, k] = A[i, 4] * B[4, k].
    # This is a single rank-1 term for each (i, k) pair?
    # No, for a 2x1x2 block (i in {0,1}, j={4}, k in {ka,kb}), the tensor is:
    # T[0,4,ka]=1, T[0,4,kb]=1, T[1,4,ka]=1, T[1,4,kb]=1.
    # This can be decomposed into 4 rank-1 terms:
    # 1. U[0,4]=1, V[4,ka]=1, W[ka,0]=1
    # 2. U[0,4]=1, V[4,kb]=1, W[kb,0]=1
    # 3. U[1,4]=1, V[4,ka]=1, W[ka,1]=1
    # 4. U[1,4]=1, V[4,kb]=1, W[kb,1]=1
    
    # Block J3 x K1: j=4, k in {0,1}
    for k in [0, 1]:
        for i in [0, 1]:
            u_idx = i * m + 4
            v_idx = 4 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1

    # Block J3 x K2: j=4, k in {2,3}
    for k in [2, 3]:
        for i in [0, 1]:
            u_idx = i * m + 4
            v_idx = 4 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1

    # Block J3 x K3: j=4, k in {4,5}
    for k in [4, 5]:
        for i in [0, 1]:
            u_idx = i * m + 4
            v_idx = 4 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1

    # Check if we filled exactly 54 columns
    if col != 54:
        # If something went wrong, fallback to trivial
        pass

    U_used = U[:, :col]
    V_used = V[:, :col]
    W_used = W[:, :col]

    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_used[i * m + j, :]
                v_row = V_used[j * p + k, :]
                w_row = W_used[k * n + i, :]
                result = np.sum(u_row * v_row * w_row)
                if abs(result - 1) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to trivial rank 60
        U_fallback = np.zeros((n * m, 60), dtype=float)
        V_fallback = np.zeros((m * p, 60), dtype=float)
        W_fallback = np.zeros((p * n, 60), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fallback[i * m + j, c] = 1
                    V_fallback[j * p + k, c] = 1
                    W_fallback[k * n + i, c] = 1
                    c += 1
        return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}

    return {"U": U_used.tolist(), "V": V_used.tolist(), "W": W_used.tolist()}
# EVOLVE-BLOCK-END
