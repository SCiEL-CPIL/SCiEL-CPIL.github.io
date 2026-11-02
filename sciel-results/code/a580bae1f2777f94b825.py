import numpy as np
import time

def construct():
    n, m, p = 4, 4, 4
    start = time.time()
    DEADLINE = 145.0

    def elapsed():
        return time.time() - start

    # 2x2 Strassen bilinear forms: (A_mask, B_mask, C_mask)
    S = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]

    def build_recursive_strassen():
        rank = 49
        U = np.zeros((16, rank), dtype=int)
        V = np.zeros((16, rank), dtype=int)
        W = np.zeros((16, rank), dtype=int)
        for t in range(7):
            bA, bB, bC = S[t]
            for s in range(7):
                sA, sB, sC = S[s]
                r = t * 7 + s
                for (bi, bj), ca in bA.items():
                    for (sr, sc), cs in sA.items():
                        U[(2*bi + sr)*4 + (2*bj + sc), r] += ca * cs
                for (bi, bj), cb in bB.items():
                    for (sr, sc), cs in sB.items():
                        V[(2*bi + sr)*4 + (2*bj + sc), r] += cb * cs
                for (ci, cj), cc in bC.items():
                    for (sr, sc), cs in sC.items():
                        W[(2*cj + sc)*4 + (2*ci + sr), r] += cc * cs
        return U, V, W, rank

    def build_naive():
        rank = 64
        U = np.zeros((16, rank), dtype=int)
        V = np.zeros((16, rank), dtype=int)
        W = np.zeros((16, rank), dtype=int)
        r = 0
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    U[i*4 + k, r] = 1
                    V[k*4 + j, r] = 1
                    W[j*4 + i, r] = 1
                    r += 1
        return U, V, W, rank

    def verify(U, V, W, rank, ntests=120, seed=987654321):
        rng = np.random.default_rng(seed)
        for _ in range(ntests):
            A = rng.integers(-5, 6, size=(4,4))
            B = rng.integers(-5, 6, size=(4,4))
            C_true = A @ B
            C_rec = np.zeros((4,4))
            for i in range(4):
                for j in range(4):
                    acc = 0
                    for r in range(rank):
                        ua = 0
                        for k in range(4):
                            ua += A[i, k] * U[i*4 + k, r]
                        vb = 0
                        for k in range(4):
                            vb += B[k, j] * V[k*4 + j, r]
                        acc += W[j*4 + i, r] * ua * vb
                    C_rec[i, j] = acc
            if not np.array_equal(C_true, C_rec):
                return False
        return True

    # Primary: rank-49 recursive Strassen (best known upper bound for M(4))
    U, V, W, rank = build_recursive_strassen()
    if verify(U, V, W, rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: naive rank-64 (always valid)
    U, V, W, rank = build_naive()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
