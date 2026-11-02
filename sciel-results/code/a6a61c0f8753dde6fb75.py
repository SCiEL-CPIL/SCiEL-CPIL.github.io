import time

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 120.0

def _check_deadline():
    return (time.time() - _START_TIME) > (_DEADLINE - 5.0)

def construct():
    """
    Construct a rank-56 decomposition for 2x4 by 4x8 matrix multiplication.
    Uses 8 independent 2x2 Strassen decompositions.
    """
    n, m, p = DIMENSIONS
    
    # Fallback: Rank 64 schoolbook
    if _check_deadline():
        rank_fb = n * m * p
        U_fb = [[0.0] * rank_fb for _ in range(n * m)]
        V_fb = [[0.0] * rank_fb for _ in range(m * p)]
        W_fb = [[0.0] * rank_fb for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U_fb[i * m + j][r] = 1.0
                    V_fb[j * p + k][r] = 1.0
                    W_fb[k * n + i][r] = 1.0
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    rank = 56
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # 8 block products:
    # d in 0..3 (column block of B and C)
    # h in 0..1 (row block of B, corresponding to column half of A)
    # h=0: A[:, 0:2] * B[0:2, 2d:2d+2]
    # h=1: A[:, 2:4] * B[2:4, 2d:2d+2]
    
    for d in range(4):
        for h in range(2):
            base = (d * 2 + h) * 7
            
            # Define index mappings
            if h == 0:
                def a_idx(r, c): return r * m + c
                def b_idx(r, c): return r * p + (2 * d + c)
            else:
                def a_idx(r, c): return r * m + (2 + c)
                def b_idx(r, c): return (2 + r) * p + (2 * d + c)
            
            def c_idx(r, c): return (2 * d + c) * n + r

            # Strassen products
            # M1 = (A00 + A11) * (B00 + B11)
            col = base + 0
            U[a_idx(0, 0)][col] += 1.0
            U[a_idx(1, 1)][col] += 1.0
            V[b_idx(0, 0)][col] += 1.0
            V[b_idx(1, 1)][col] += 1.0
            W[c_idx(0, 0)][col] += 1.0
            W[c_idx(1, 1)][col] += 1.0

            # M2 = (A10 + A11) * B00
            col = base + 1
            U[a_idx(1, 0)][col] += 1.0
            U[a_idx(1, 1)][col] += 1.0
            V[b_idx(0, 0)][col] += 1.0
            W[c_idx(1, 0)][col] += 1.0
            W[c_idx(1, 1)][col] -= 1.0

            # M3 = A00 * (B01 - B11)
            col = base + 2
            U[a_idx(0, 0)][col] += 1.0
            V[b_idx(0, 1)][col] += 1.0
            V[b_idx(1, 1)][col] -= 1.0
            W[c_idx(0, 1)][col] += 1.0
            W[c_idx(1, 1)][col] += 1.0

            # M4 = A11 * (B10 - B00)
            col = base + 3
            U[a_idx(1, 1)][col] += 1.0
            V[b_idx(1, 0)][col] += 1.0
            V[b_idx(0, 0)][col] -= 1.0
            W[c_idx(0, 0)][col] += 1.0
            W[c_idx(1, 0)][col] += 1.0

            # M5 = (A00 + A01) * B11
            col = base + 4
            U[a_idx(0, 0)][col] += 1.0
            U[a_idx(0, 1)][col] += 1.0
            V[b_idx(1, 1)][col] += 1.0
            W[c_idx(0, 0)][col] -= 1.0
            W[c_idx(0, 1)][col] += 1.0

            # M6 = (A10 - A00) * (B00 + B01)
            col = base + 5
            U[a_idx(1, 0)][col] += 1.0
            U[a_idx(0, 0)][col] -= 1.0
            V[b_idx(0, 0)][col] += 1.0
            V[b_idx(0, 1)][col] += 1.0
            W[c_idx(1, 1)][col] += 1.0

            # M7 = (A01 - A11) * (B10 + B11)
            col = base + 6
            U[a_idx(0, 1)][col] += 1.0
            U[a_idx(1, 1)][col] -= 1.0
            V[b_idx(1, 0)][col] += 1.0
            V[b_idx(1, 1)][col] += 1.0
            W[c_idx(0, 0)][col] += 1.0

    return {"U": U, "V": V, "W": W}
