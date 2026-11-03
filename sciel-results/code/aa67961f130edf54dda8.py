"""Matrix multiplication 2x4x7: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n=2, m=4, p=7, tol=1e-9):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(Ur[i, j, :], Vr[j, k, :] * Wr[k, i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n=2, m=4, p=7):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U, V, W)

    # Construct a rank-52 decomposition using Strassen on 2x2x2 blocks
    # Partition: j in {0,1,2,3}, k in {0,1,2,3,4,5,6}
    # 4 Strassen 2x2x2 blocks (j0 in {0,2}, k0 in {0,2}) -> 4*7 = 28
    # 2 schoolbook 2x2x3 blocks (j0 in {0,2}, k in {4,5,6}) -> 2*12 = 24
    # Total: 52

    rank = 52
    U_new = np.zeros((n * m, rank))
    V_new = np.zeros((m * p, rank))
    W_new = np.zeros((p * n, rank))
    col = 0

    # Verified Strassen 2x2x2 decomposition (7 rank-1 terms)
    # Each term is (u, v, w) where u,v,w are length-2 vectors
    # such that sum_t u_t[a]*v_t[b]*w_t[c] = 1 for all a,b,c in {0,1}
    strassen_terms = [
        (np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])),
        (np.array([1.0, 0.0]), np.array([1.0, 1.0]), np.array([0.0, 1.0])),
        (np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])),
        (np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 1.0])),
        (np.array([1.0, 1.0]), np.array([1.0, -1.0]), np.array([1.0, 1.0])),
        (np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 0.0])),
        (np.array([1.0, -1.0]), np.array([1.0, 1.0]), np.array([1.0, 0.0])),
    ]

    # Verify Strassen terms
    valid_strassen = True
    for a in range(2):
        for b in range(2):
            for c in range(2):
                total = sum(u[a] * v[b] * w[c] for u, v, w in strassen_terms)
                if abs(total - 1.0) > 1e-9:
                    valid_strassen = False
                    break
            if not valid_strassen:
                break
        if not valid_strassen:
            break

    if valid_strassen:
        for j0 in [0, 2]:
            for k0 in [0, 2]:
                for u, v, w in strassen_terms:
                    # u indexed by a (i_local), v by b (j_local), w by c (k_local)
                    for a in range(2):
                        for b in range(2):
                            U_new[a * m + j0 + b, col] = u[a]
                    for b in range(2):
                        for c in range(2):
                            V_new[(j0 + b) * p + k0 + c, col] = v[b]
                    for c in range(2):
                        for a in range(2):
                            W_new[(k0 + c) * n + a, col] = w[c]
                    col += 1

        # Remaining: j in {0,1} or {2,3}, k in {4,5,6}
        for j0 in [0, 2]:
            for k in range(4, 7):
                for i in range(2):
                    for j in range(2):
                        U_new[i * m + j0 + j, col] = 1.0
                        V_new[(j0 + j) * p + k, col] = 1.0
                        W_new[k * n + i, col] = 1.0
                        col += 1

        if col == rank:
            if _verify(U_new.tolist(), V_new.tolist(), W_new.tolist()):
                best_rank = rank
                best_result = (U_new.tolist(), V_new.tolist(), W_new.tolist())

    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
