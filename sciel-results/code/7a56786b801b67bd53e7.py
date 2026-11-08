import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    # Valid decomposition rank for 2x3x4 is 22 (7+4 + 7+4)
    # Minimal is 15, but 22 is safe and valid.
    rank = 22
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val

    # Helper to get index
    def u_idx(i, k): return i * m + k
    def v_idx(k, j): return k * p + j
    def w_idx(j, i): return j * n + i

    # --- Part 1: A[:, :2] * B[:2, :2] -> C[:, :2] (Strassen, Rank 7) ---
    # A: a00=u_idx(0,0), a01=u_idx(0,1), a10=u_idx(1,0), a11=u_idx(1,1)
    # B: b00=v_idx(0,0), b01=v_idx(0,1), b10=v_idx(1,0), b11=v_idx(1,1)
    # C: c00=w_idx(0,0), c01=w_idx(1,0), c10=w_idx(0,1), c11=w_idx(1,1)
    
    # M1: (a00+a11)(b00+b11) -> c00+c11
    add_term([(u_idx(0,0), 1), (u_idx(1,1), 1)], [(v_idx(0,0), 1), (v_idx(1,1), 1)], [(w_idx(0,0), 1), (w_idx(1,1), 1)], 0)
    # M2: (a10+a11)b00 -> c10-c11
    add_term([(u_idx(1,0), 1), (u_idx(1,1), 1)], [(v_idx(0,0), 1)], [(w_idx(0,1), 1), (w_idx(1,1), -1)], 1)
    # M3: a00(b01-b11) -> c01+c11
    add_term([(u_idx(0,0), 1)], [(v_idx(0,1), 1), (v_idx(1,1), -1)], [(w_idx(1,0), 1), (w_idx(1,1), 1)], 2)
    # M4: a11(b10-b00) -> c00+c10
    add_term([(u_idx(1,1), 1)], [(v_idx(1,0), 1), (v_idx(0,0), -1)], [(w_idx(0,0), 1), (w_idx(0,1), 1)], 3)
    # M5: (a00+a01)b11 -> -c00+c01
    add_term([(u_idx(0,0), 1), (u_idx(0,1), 1)], [(v_idx(1,1), 1)], [(w_idx(0,0), -1), (w_idx(1,0), 1)], 4)
    # M6: (a10-a00)(b00+b01) -> c11
    add_term([(u_idx(1,0), 1), (u_idx(0,0), -1)], [(v_idx(0,0), 1), (v_idx(0,1), 1)], [(w_idx(1,1), 1)], 5)
    # M7: (a01-a11)(b10+b11) -> c00
    add_term([(u_idx(0,1), 1), (u_idx(1,1), -1)], [(v_idx(1,0), 1), (v_idx(1,1), 1)], [(w_idx(0,0), 1)], 6)

    # --- Part 2: A[:, :2] * B[:2, 2:4] -> C[:, 2:4] (Strassen, Rank 7) ---
    # B cols 2,3: b02=v_idx(0,2), b03=v_idx(0,3), b12=v_idx(1,2), b13=v_idx(1,3)
    # C cols 2,3: c02=w_idx(2,0), c03=w_idx(3,0), c12=w_idx(2,1), c13=w_idx(3,1)
    
    # M1': (a00+a11)(b02+b13) -> c02+c13
    add_term([(u_idx(0,0), 1), (u_idx(1,1), 1)], [(v_idx(0,2), 1), (v_idx(1,3), 1)], [(w_idx(2,0), 1), (w_idx(3,1), 1)], 7)
    # M2': (a10+a11)b02 -> c12-c13
    add_term([(u_idx(1,0), 1), (u_idx(1,1), 1)], [(v_idx(0,2), 1)], [(w_idx(2,1), 1), (w_idx(3,1), -1)], 8)
    # M3': a00(b03-b13) -> c03+c13
    add_term([(u_idx(0,0), 1)], [(v_idx(0,3), 1), (v_idx(1,3), -1)], [(w_idx(3,0), 1), (w_idx(3,1), 1)], 9)
    # M4': a11(b12-b02) -> c02+c12
    add_term([(u_idx(1,1), 1)], [(v_idx(1,2), 1), (v_idx(0,2), -1)], [(w_idx(2,0), 1), (w_idx(2,1), 1)], 10)
    # M5': (a00+a01)b13 -> -c02+c03
    add_term([(u_idx(0,0), 1), (u_idx(0,1), 1)], [(v_idx(1,3), 1)], [(w_idx(2,0), -1), (w_idx(3,0), 1)], 11)
    # M6': (a10-a00)(b02+b03) -> c13
    add_term([(u_idx(1,0), 1), (u_idx(0,0), -1)], [(v_idx(0,2), 1), (v_idx(0,3), 1)], [(w_idx(3,1), 1)], 12)
    # M7': (a01-a11)(b12+b13) -> c02
    add_term([(u_idx(0,1), 1), (u_idx(1,1), -1)], [(v_idx(1,2), 1), (v_idx(1,3), 1)], [(w_idx(2,0), 1)], 13)

    # --- Part 3: A[:, 2:3] * B[2:3, :2] -> C[:, :2] (Rank 4) ---
    # A: a02=u_idx(0,2), a12=u_idx(1,2)
    # B: b20=v_idx(2,0), b21=v_idx(2,1)
    # C: c00, c01, c10, c11
    
    # T14: a02 * b20 -> c00
    add_term([(u_idx(0,2), 1)], [(v_idx(2,0), 1)], [(w_idx(0,0), 1)], 14)
    # T15: a02 * b21 -> c01
    add_term([(u_idx(0,2), 1)], [(v_idx(2,1), 1)], [(w_idx(1,0), 1)], 15)
    # T16: a12 * b20 -> c10
    add_term([(u_idx(1,2), 1)], [(v_idx(2,0), 1)], [(w_idx(0,1), 1)], 16)
    # T17: a12 * b21 -> c11
    add_term([(u_idx(1,2), 1)], [(v_idx(2,1), 1)], [(w_idx(1,1), 1)], 17)

    # --- Part 4: A[:, 2:3] * B[2:3, 2:4] -> C[:, 2:4] (Rank 4) ---
    # B: b22=v_idx(2,2), b23=v_idx(2,3)
    # C: c02, c03, c12, c13
    
    # T18: a02 * b22 -> c02
    add_term([(u_idx(0,2), 1)], [(v_idx(2,2), 1)], [(w_idx(2,0), 1)], 18)
    # T19: a02 * b23 -> c03
    add_term([(u_idx(0,2), 1)], [(v_idx(2,3), 1)], [(w_idx(3,0), 1)], 19)
    # T20: a12 * b22 -> c12
    add_term([(u_idx(1,2), 1)], [(v_idx(2,2), 1)], [(w_idx(2,1), 1)], 20)
    # T21: a12 * b23 -> c13
    add_term([(u_idx(1,2), 1)], [(v_idx(2,3), 1)], [(w_idx(3,1), 1)], 21)

    # Verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    np.random.seed(42)
    A_test = np.random.randint(-5, 5, size=(n, m)).astype(float)
    B_test = np.random.randint(-5, 5, size=(m, p)).astype(float)
    C_expected = A_test @ B_test
    
    C_reconstructed = np.zeros((n, p), dtype=float)
    for i in range(n):
        for j in range(p):
            val = 0.0
            for r in range(rank):
                u_prod = sum(U_np[i * m + k, r] * A_test[i, k] for k in range(m))
                v_prod = sum(V_np[k * p + j, r] * B_test[k, j] for k in range(m))
                val += W_np[j * n + i, r] * u_prod * v_prod
            C_reconstructed[i, j] = val
            
    if not np.allclose(C_reconstructed, C_expected, atol=1e-10):
        raise ValueError("Decomposition verification failed")

    return {"U": U, "V": V, "W": W}
