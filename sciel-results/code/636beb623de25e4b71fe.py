import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    U_t = U.reshape(n, m, r)
    V_t = V.reshape(m, p, r)
    W_t = W.reshape(p, n, r)
    prod = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t, optimize=True)
    return np.allclose(prod, np.ones((n, m, p)), atol=tol)


def _block_strassen(n, m, p):
    """
    Build a decomposition using 2x2 Strassen blocks where possible.
    Split j into pairs, k into pairs; scalar leftover handled by schoolbook.
    Returns (U, V, W) as numpy arrays of shape (n*m, r), (m*p, r), (p*n, r).
    """
    # Determine pairing
    j_pairs = []
    j = 0
    while j + 1 < m:
        j_pairs.append((j, j + 1))
        j += 2
    j_leftover = [j] if j < m else []

    k_pairs = []
    k = 0
    while k + 1 < p:
        k_pairs.append((k, k + 1))
        k += 2
    k_leftover = [k] if k < p else []

    rank = 0
    # 7 per (j_pair, k_pair), 2n per (j_leftover, k), 2n per (j, k_leftover)
    rank += 7 * len(j_pairs) * len(k_pairs)
    rank += 2 * n * len(j_leftover) * p
    rank += 2 * n * m * len(k_leftover)

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    # Strassen 7-term block for 2x2 A-block (rows i=0..n-1, cols j_a,j_b)
    # times 2x2 B-block (rows j_a,j_b, cols k_a,k_b).
    # We apply the 2x2x2 Strassen pattern in the (j,k) plane, replicated over i.
    for (j_a, j_b) in j_pairs:
        for (k_a, k_b) in k_pairs:
            # M1 = (A[i,j_a]+A[i,j_b])*(B[j_a,k_a]+B[j_b,k_b])
            for i in range(n):
                U[i * m + j_a, col] += 1
                U[i * m + j_b, col] += 1
                V[j_a * p + k_a, col] += 1
                V[j_b * p + k_b, col] += 1
                W[k_a * n + i, col] += 1
                W[k_b * n + i, col] += 1
            col += 1

            # M2 = (A[i,j_a]+A[i,j_b])*B[j_a,k_a]
            for i in range(n):
                U[i * m + j_a, col] += 1
                U[i * m + j_b, col] += 1
                V[j_a * p + k_a, col] += 1
                W[k_a * n + i, col] += 1
                W[k_b * n + i, col] -= 1
            col += 1

            # M3 = A[i,j_a]*(B[j_a,k_b]-B[j_b,k_b])
            for i in range(n):
                U[i * m + j_a, col] += 1
                V[j_a * p + k_b, col] += 1
                V[j_b * p + k_b, col] -= 1
                W[k_b * n + i, col] += 1
            col += 1

            # M4 = A[i,j_b]*(B[j_b,k_a]-B[j_a,k_a])
            for i in range(n):
                U[i * m + j_b, col] += 1
                V[j_b * p + k_a, col] += 1
                V[j_a * p + k_a, col] -= 1
                W[k_a * n + i, col] += 1
            col += 1

            # M5 = (A[i,j_a]+A[i,j_b])*B[j_b,k_b]
            for i in range(n):
                U[i * m + j_a, col] += 1
                U[i * m + j_b, col] += 1
                V[j_b * p + k_b, col] += 1
                W[k_a * n + i, col] -= 1
                W[k_b * n + i, col] += 1
            col += 1

            # M6 = (A[i,j_a]-A[i,j_a])... use (A[i,j_b]-A[i,j_a])*(B[j_a,k_a]+B[j_a,k_b])
            for i in range(n):
                U[i * m + j_b, col] += 1
                U[i * m + j_a, col] -= 1
                V[j_a * p + k_a, col] += 1
                V[j_a * p + k_b, col] += 1
                W[k_b * n + i, col] += 1
            col += 1

            # M7 = (A[i,j_a]-A[i,j_b])*(B[j_b,k_a]+B[j_b,k_b])
            for i in range(n):
                U[i * m + j_a, col] += 1
                U[i * m + j_b, col] -= 1
                V[j_b * p + k_a, col] += 1
                V[j_b * p + k_b, col] += 1
                W[k_a * n + i, col] += 1
            col += 1

    # Leftover j (scalar column) times all k
    for j in j_leftover:
        for k in range(p):
            for i in range(n):
                U[i * m + j, col] += 1
                V[j * p + k, col] += 1
                W[k * n + i, col] += 1
                col += 1

    # All j times leftover k (scalar column of B)
    for k in k_leftover:
        for j in range(m):
            for i in range(n):
                U[i * m + j, col] += 1
                V[j * p + k, col] += 1
                W[k * n + i, col] += 1
                col += 1

    return U, V, W


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _snap_half(x):
    return np.round(x * 2.0) / 2.0


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # 1) Structural block-Strassen candidate (deterministic, fast).
    U_b, V_b, W_b = _block_strassen(n, m, p)
    best = None
    if _verify(U_b, V_b, W_b, n, m, p):
        best = (U_b, V_b, W_b)

    # 2) Schoolbook fallback (always valid).
    U_s, V_s, W_s = _schoolbook(n, m, p)
    if best is None or U_s.shape[1] < best[0].shape[1]:
        best = (U_s, V_s, W_s)

    # 3) Try to beat the structural rank via a small number of randomized
    #    tensor-decomposition solves, snapping to half-integers and verifying.
    target_rank = best[0].shape[1] - 1
    min_rank = max(1, target_rank - 6)
    T = np.ones((n, m, p), dtype=float)

    for r in range(target_rank, min_rank - 1, -1):
        if time.time() > deadline:
            break
        for restart in range(6):
            if time.time() > deadline:
                break
            np.random.seed(12345 + r * 100 + restart)
            U0 = np.random.randn(n * m, r) * 0.5
            V0 = np.random.randn(m * p, r) * 0.5
            W0 = np.random.randn(p * n, r) * 0.5

            def objective(params, n=n, m=m, p=p, r=r, T=T):
                Uf, Vf, Wf = params[:n*m*r], params[n*m*r:n*m*r+m*p*r], params[n*m*r+m*p*r:]
                Ut = Uf.reshape(n, m, r)
                Vt = Vf.reshape(m, p, r)
                Wt = Wf.reshape(p, n, r)
                prod = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt, optimize=True)
                return float(np.sum((prod - T) ** 2))

            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            try:
                from scipy.optimize import minimize
                res = minimize(objective, x0, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 40})
                if res.fun < 1e-7:
                    Usol = res.x[:n*m*r].reshape(n, m, r)
                    Vsol = res.x[n*m*r:n*m*r+m*p*r].reshape(m, p, r)
                    Wsol = res.x[n*m*r+m*p*r:].reshape(p, n, r)
                    Uq, Vq, Wq = _snap_half(Usol), _snap_half(Vsol), _snap_half(Wsol)
                    if _verify(Uq, Vq, Wq, n, m, p):
                        best = (Uq, Vq, Wq)
                        break
            except Exception:
                pass

    Uo, Vo, Wo = best
    return {
        "U": Uo.reshape(n * m, Uo.shape[1]).tolist(),
        "V": Vo.reshape(m * p, Vo.shape[1]).tolist(),
        "W": Wo.reshape(p * n, Wo.shape[1]).tolist(),
    }
# EVOLVE-BLOCK-END
