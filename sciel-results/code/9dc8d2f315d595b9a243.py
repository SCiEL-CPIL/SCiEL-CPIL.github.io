"""Search for a low-rank exact tensor decomposition of 3x3 * 3x8 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]


def _make_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


def _to_list(M):
    return [[float(M[i, j]) for j in range(M.shape[1])] for i in range(M.shape[0])]


def _verify(U, V, W, T, tol=1e-7):
    # Reconstruct tensor: T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
    n, m, p = T.shape[0], T.shape[1], T.shape[2]
    # Use einsum: sum over c of U[i,c]*V[j,c]*W[k,c]
    recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    return float(np.max(np.abs(recon - T)))


def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # safety margin under 120s
    n, m, p = DIMENSIONS
    T = _make_tensor(n, m, p)

    best_rank = n * m * p
    best = None
    U0, V0, W0 = _schoolbook(n, m, p)
    best = (U0, V0, W0)

    rng = np.random.default_rng(12345)

    def time_left():
        return deadline - time.time()

    # Try ranks below schoolbook, descending, with a few restarts each.
    # 3x3x8: schoolbook = 72. Try 50..40 range realistically.
    for rank in range(55, 39, -1):
        if time_left() < 25.0:
            break
        # Number of restarts scales with remaining time
        max_restarts = 6
        for restart in range(max_restarts):
            if time_left() < 20.0:
                break
            # Random init
            U = rng.standard_normal((n * m, rank))
            V = rng.standard_normal((m * p, rank))
            W = rng.standard_normal((p * n, rank))
            # Normalize columns to help conditioning
            for c in range(rank):
                nrm = np.linalg.norm(U[:, c]) + np.linalg.norm(V[:, c]) + np.linalg.norm(W[:, c]) + 1e-12
                U[:, c] /= nrm
                V[:, c] /= nrm
                W[:, c] /= nrm

            # Alternating least squares (ALS) for rank-r decomposition of T
            # T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
            # Fix V,W -> solve for U: U[i,c] = sum_{j,k} T[i,j,k] V[j,c] W[k,c] / sum_{j,k} V[j,c]^2 W[k,c]^2
            # etc.
            prev_err = np.inf
            for it in range(300):
                if time_left() < 18.0:
                    break
                # Solve for U
                # U[i,c] = sum_{j,k} T[i,j,k] V[j,c] W[k,c] / denom[c]
                # denom[c] = sum_{j,k} V[j,c]^2 W[k,c]^2 = (sum_j V[j,c]^2)*(sum_k W[k,c]^2)
                v2 = np.sum(V * V, axis=0)  # shape (rank,)
                w2 = np.sum(W * W, axis=0)  # shape (rank,)
                denom = v2 * w2 + 1e-12
                # numerator[i,c] = sum_{j,k} T[i,j,k] V[j,c] W[k,c]
                # = sum_j V[j,c] (sum_k T[i,j,k] W[k,c])
                # Let A[i,j,c] = sum_k T[i,j,k] W[k,c]
                A = np.einsum('ijk,kc->ijc', T, W)  # (n*m, m*p, rank)
                numU = np.einsum('jc,ijc->ic', V, A)  # (n*m, rank)
                U = numU / denom[None, :]

                # Solve for V
                u2 = np.sum(U * U, axis=0)
                denom = u2 * w2 + 1e-12
                B = np.einsum('ijk,ic->jk c'.replace(' c',''), T, U)  # (m*p, p*n, rank)
                B = np.einsum('ijk,ic->jkc', T, U)
                numV = np.einsum('kc,jkc->jc', W, B)
                V = numV / denom[None, :]

                # Solve for W
                u2 = np.sum(U * U, axis=0)
                v2 = np.sum(V * V, axis=0)
                denom = u2 * v2 + 1e-12
                C = np.einsum('ijk,jc->ikc', T, V)  # (n*m, p*n, rank)
                numW = np.einsum('ic,ikc->kc', U, C)
                W = numW / denom[None, :]

                err = _verify(U, V, W, T)
                if err < 1e-7:
                    # Found a valid decomposition at this rank
                    if rank < best_rank:
                        best_rank = rank
                        best = (U.copy(), V.copy(), W.copy())
                        break
                if abs(err - prev_err) < 1e-10:
                    # Stagnant; try a small perturbation
                    U += 1e-3 * rng.standard_normal(U.shape)
                    V += 1e-3 * rng.standard_normal(V.shape)
                    W += 1e-3 * rng.standard_normal(W.shape)
                prev_err = err
            if best_rank == rank:
                # Found at this rank; try to go lower
                continue

    # Final: ensure we return something valid
    if best is not None and best_rank < n * m * p:
        U, V, W = best
    else:
        U, V, W = U0, V0, W0
        best_rank = n * m * p

    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
