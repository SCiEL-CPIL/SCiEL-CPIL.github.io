"""Matrix multiplication 2x3 by 3x9 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    num_a = n * m  # 6
    num_b = m * p  # 27
    num_c = p * n  # 18
    
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i*m+j, j*p+k, k*n+i))
    valid_set = set(triples)
    
    # Precompute index lists for efficiency
    a_to_triples = {}
    b_to_triples = {}
    c_to_triples = {}
    for idx, (a, b, c) in enumerate(triples):
        a_to_triples.setdefault(a, []).append(idx)
        b_to_triples.setdefault(b, []).append(idx)
        c_to_triples.setdefault(c, []).append(idx)
    
    def compute_error(U, V, W):
        err = 0.0
        for (a, b, c) in triples:
            val = np.dot(U[a], V[b] * W[c])
            err += (val - 1.0)**2
        for a in range(num_a):
            for b in range(num_b):
                for c in range(num_c):
                    if (a, b, c) not in valid_set:
                        val = np.dot(U[a], V[b] * W[c])
                        err += val**2
        return err
    
    def try_rank(R, n_restarts=8, max_iters=150):
        best_err = float('inf')
        best = None
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            U = np.random.randn(num_a, R) * 0.5
            V = np.random.randn(num_b, R) * 0.5
            W = np.random.randn(num_c, R) * 0.5
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                for a in range(num_a):
                    idxs = a_to_triples[a]
                    A = np.array([V[triples[i][1]] * W[triples[i][2]] for i in idxs])
                    bvec = np.ones(len(idxs))
                    U[a] = np.linalg.lstsq(A, bvec, rcond=None)[0]
                for b in range(num_b):
                    idxs = b_to_triples[b]
                    A = np.array([U[triples[i][0]] * W[triples[i][2]] for i in idxs])
                    bvec = np.ones(len(idxs))
                    V[b] = np.linalg.lstsq(A, bvec, rcond=None)[0]
                for c in range(num_c):
                    idxs = c_to_triples[c]
                    A = np.array([U[triples[i][0]] * V[triples[i][1]] for i in idxs])
                    bvec = np.ones(len(idxs))
                    W[c] = np.linalg.lstsq(A, bvec, rcond=None)[0]
                err = compute_error(U, V, W)
                if err < best_err:
                    best_err = err
                    best = (U.copy(), V.copy(), W.copy())
                if err < 1e-18:
                    break
        if best and best_err < 1e-12:
            return best
        return None
    
    for R in [18, 27, 36, 45, 54]:
        if time.time() > deadline - 5:
            break
        res = try_rank(R)
        if res:
            U, V, W = res
            if compute_error(U, V, W) < 1e-12:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    rank = n*m*p
    U = [[0]*rank for _ in range(num_a)]
    V = [[0]*rank for _ in range(num_b)]
    W = [[0]*rank for _ in range(num_c)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
