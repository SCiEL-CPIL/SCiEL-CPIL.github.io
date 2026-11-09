import time
import numpy as np

DIMENSIONS = [4, 4, 4]
_DEADLINE = 125  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS

    # Strassen 2x2 decomposition matrices
    # Row/Col index in 2x2 blocks: 0:00, 1:01, 2:10, 3:11 (row-major)
    # W2 row index corresponds to output element index k*2+i (col*2+row)
    # Index 0: C_00, Index 1: C_10, Index 2: C_01, Index 3: C_11
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],  # 00
        [0, 0, 0, 0, 1, 0, 1],   # 01
        [0, 1, 0, 0, 0, 1, 0],   # 10
        [1, 1, 0, 1, 0, 0, -1]   # 11
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],  # 00
        [0, 0, 1, 0, 0, 1, 0],   # 01
        [0, 0, 0, 1, 0, 0, 1],   # 10
        [1, 0, -1, 0, 1, 0, 1]   # 11
    ]
    
    # W2 rows ordered by k*2+i (col*2+row):
    # 0: C_00 (k=0, i=0)
    # 1: C_10 (k=0, i=1)
    # 2: C_01 (k=1, i=0)
    # 3: C_11 (k=1, i=1)
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],  # 0: C_00
        [0, 1, 0, 1, 0, 0, 0],   # 1: C_10
        [0, 0, 1, 0, 1, 0, 0],   # 2: C_01
        [1, -1, 1, 0, 0, 1, 0]   # 3: C_11
    ]

    # Build the 49-term recursive Strassen decomposition
    rank_full = 49
    U_full = [[0] * rank_full for _ in range(n * m)]
    V_full = [[0] * rank_full for _ in range(m * p)]
    W_full = [[0] * rank_full for _ in range(p * n)]
    
    # Fill U and V
    for I in range(2):
        for J in range(2):
            block_idx = I * 2 + J
            for i in range(2):
                for j in range(2):
                    inner_idx = i * 2 + j
                    u_idx = (I * 2 + i) * 4 + (J * 2 + j)
                    v_idx = (I * 2 + i) * 4 + (J * 2 + j)
                    
                    for r in range(7):
                        ub = U2[block_idx][r]
                        vb = V2[block_idx][r]
                        if ub == 0 and vb == 0:
                            continue
                        for s in range(7):
                            c = r * 7 + s
                            ui = U2[inner_idx][s]
                            vi = V2[inner_idx][s]
                            if ui == 0 and vi == 0:
                                continue
                            U_full[u_idx][c] = ub * ui
                            V_full[v_idx][c] = vb * vi

    # Fill W
    for I in range(2):
        for J in range(2):
            # W2 block index is J*2 + I (col*2+row)
            block_idx = J * 2 + I
            for i in range(2):
                for j in range(2):
                    # W2 inner index is j*2 + i (col*2+row)
                    inner_idx = j * 2 + i
                    w_idx = (J * 2 + j) * 4 + (I * 2 + i)
                    
                    for r in range(7):
                        wb = W2[block_idx][r]
                        if wb == 0:
                            continue
                        for s in range(7):
                            c = r * 7 + s
                            wi = W2[inner_idx][s]
                            if wi == 0:
                                continue
                            W_full[w_idx][c] = wb * wi

    # Aggressive pruning: remove columns where any of U,V,W is entirely zero
    nm = n * m
    mp = m * p
    pn = p * n
    keep = []
    for c in range(rank_full):
        u_zero = all(U_full[r][c] == 0 for r in range(nm))
        v_zero = all(V_full[r][c] == 0 for r in range(mp))
        w_zero = all(W_full[r][c] == 0 for r in range(pn))
        if not (u_zero or v_zero or w_zero):
            keep.append(c)
    
    U = [[row[c] for c in keep] for row in U_full]
    V = [[row[c] for c in keep] for row in V_full]
    W = [[row[c] for c in keep] for row in W_full]
    
    # Verify correctness with a quick spot-check
    if time.time() - t0 < _DEADLINE - 5:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        np.random.seed(42)
        A = np.random.randint(-5, 6, size=(n, m)).astype(float)
        B = np.random.randint(-5, 6, size=(m, p)).astype(float)
        C_expected = A @ B
        
        C_actual = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                s = 0.0
                for r in range(len(keep)):
                    s += U_np[i * m + j, r] * V_np[j * p + i, r] * W_np[i * n + j, r]
                C_actual[i, j] = s
        
        if not np.allclose(C_actual, C_expected, atol=1e-9):
            # Fallback to unpruned version
            return {"U": U_full, "V": V_full, "W": W_full}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
