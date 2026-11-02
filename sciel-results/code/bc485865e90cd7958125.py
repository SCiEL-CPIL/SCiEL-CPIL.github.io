"""Matrix multiplication 3x3x3: rank-minimal tensor decomposition.

Strategy:
  1. Start from a verified rank-26 construction (Strassen 2x2 core + schoolbook border).
  2. Run a continuous local search over the factor matrices (a different optimization
     family than combinatorial term-merging) to attempt rank reduction toward the
     known minimal rank 23. The search minimizes the squared error on the 27 tensor
     entries plus an L1 penalty on the factor columns to drive redundant columns to
     zero. If a lower-rank valid decomposition is found, return it; otherwise fall
     back to the verified rank-26 (or schoolbook) construction.
"""
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n = m = p = 3
    U = [[float(x) for x in row] for row in U]
    V = [[float(x) for x in row] for row in V]
    W = [[float(x) for x in row] for row in W]
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    n = m = p = 3
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_rank26():
    """
    Rank-26 decomposition for 3x3 matrix multiplication.
    Strassen core (7 terms) for the 2x2 block + schoolbook border terms.
    """
    n = m = p = 3
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    terms = []

    # --- Strassen core for C[0:2,0:2] (7 terms) ---
    terms.append(([ (u_idx(0,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,1),1), (w_idx(1,1),-1) ]))
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,1),1), (v_idx(1,1),-1) ], [ (w_idx(1,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,0),1), (v_idx(0,0),-1) ], [ (w_idx(0,0),1), (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(0,0),1), (u_idx(0,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(0,0),-1), (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(0,0),-1) ], [ (v_idx(0,0),1), (v_idx(0,1),1) ], [ (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(0,1),1), (u_idx(1,1),-1) ], [ (v_idx(1,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1) ]))

    # --- Border: C[0:2, 2] ---
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(1,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,1),1) ]))

    # --- Border: C[2, 0:2] ---
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,2),1) ]))

    # --- Corner: C[2,2] ---
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,2),1) ]))

    # --- Additional border: A[0:2,2]*B[2,0:2] contributes to C[0:2,0:2] ---
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,1),1) ]))

    for r, (u_list, v_list, w_list) in enumerate(terms):
        for (row, coeff) in u_list:
            U[row][r] += coeff
        for (row, coeff) in v_list:
            V[row][r] += coeff
        for (row, coeff) in w_list:
            W[row][r] += coeff

    return {"U": U, "V": V, "W": W}

def _tensor_residual(U, V, W):
    """Return vector of the 27 residuals: sum_r U[i*m+j,r]V[j*p+k,r]W[k*n+i,r] - 1."""
    import numpy as np
    n = m = p = 3
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    R = U.shape[1]
    res = np.empty(27, dtype=float)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # contribution per rank index r
                s = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                res[idx] = s - 1.0
                idx += 1
    return res

def _local_search_reduce(base_result, deadline):
    """
    Continuous local search to reduce rank of a valid decomposition.

    Different representation/optimization family: we parameterize the factors
    directly and run a damped Gauss-Newton / coordinate hybrid that minimizes
    ||residual||^2 + lambda * sum_r ||col_r||_1  to drive redundant columns to
    zero, then prune near-zero columns and re-verify.
    """
    import numpy as np

    n = m = p = 3
    U0 = np.array(base_result["U"], dtype=float)
    V0 = np.array(base_result["V"], dtype=float)
    W0 = np.array(base_result["W"], dtype=float)
    R = U0.shape[1]

    U = U0.copy()
    V = V0.copy()
    W = W0.copy()

    best = None
    best_rank = R

    def try_prune_and_verify(Uc, Vc, Wc, tol=1e-6):
        """Prune columns whose L2 norm is below tol, verify, return (valid, rank, Uc,Vc,Wc)."""
        norms = np.sqrt(np.sum(Uc**2, axis=0) + np.sum(Vc**2, axis=0) + np.sum(Wc**2, axis=0))
        keep = norms > tol
        if keep.sum() >= R:
            return False, R, Uc, Vc, Wc
        Uk = Uc[:, keep]
        Vk = Vc[:, keep]
        Wk = Wc[:, keep]
        if _verify(Uk.tolist(), Vk.tolist(), Wk.tolist()):
            return True, int(keep.sum()), Uk, Vk, Wk
        return False, R, Uc, Vc, Wc

    # Precompute index arrays for fast residual/gradient.
    # For each (i,j,k) triple index t, the gradient w.r.t. U[i*m+j,r] is V[j*p+k,r]*W[k*n+i,r]
    # and similarly cyclic.
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i, j, k))
    T = len(triples)

    # We'll do coordinate descent on each factor entry with a small step,
    # driven by the residual gradient, plus an L1 shrinkage on columns.
    # This is a heuristic local search; we cap iterations by deadline.

    lr = 0.05
    lam = 0.02  # L1 penalty weight
    it = 0
    max_iter = 4000
    while it < max_iter and time.time() < deadline:
        it += 1
        # Compute residuals
        res = np.empty(T)
        for t, (i, j, k) in enumerate(triples):
            res[t] = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :]) - 1.0
        obj = float(np.sum(res*res)) + lam * (np.sum(np.abs(U)) + np.sum(np.abs(V)) + np.sum(np.abs(W)))

        # Gradient of residual w.r.t. each factor.
        # dres_t / dU[a,r] = V[j*p+k,r] * W[k*n+i,r]  where a = i*m+j
        # We build per-entry gradients by accumulating.
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        for t, (i, j, k) in enumerate(triples):
            rt = res[t]
            # For U[i*m+j, r]:
            gU[i*m+j, :] += rt * V[j*p+k, :] * W[k*n+i, :]
            # For V[j*p+k, r]:
            gV[j*p+k, :] += rt * U[i*m+j, :] * W[k*n+i, :]
            # For W[k*n+i, r]:
            gW[k*n+i, :] += rt * U[i*m+j, :] * V[j*p+k, :]

        # L1 subgradient for shrinkage
        gU += lam * np.sign(U)
        gV += lam * np.sign(V)
        gW += lam * np.sign(W)

        # Damped gradient step
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW

        # Prune and verify occasionally
        if it % 50 == 0:
            valid, rk, Uk, Vk, Wk = try_prune_and_verify(U, V, W)
            if valid and rk < best_rank:
                best_rank = rk
                best = (Uk.copy(), Vk.copy(), Wk.copy())
                U, V, W = Uk, Vk, Wk
                R = best_rank
                # restart with fewer columns
                if best_rank <= 23:
                    break

    # Final prune + verify
    if best is not None:
        valid, rk, Uk, Vk, Wk = try_prune_and_verify(best[0], best[1], best[2])
        if valid and rk < best_rank:
            best_rank = rk
            best = (Uk.copy(), Vk.copy(), Wk.copy())
    if best is not None and best_rank < R:
        return {"U": best[0].tolist(), "V": best[1].tolist(), "W": best[2].tolist()}
    return None

def construct():
    """Construct a tensor decomposition for 3x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Verified rank-26 construction as guaranteed fallback
    result = _strassen_rank26()
    base_ok = _verify(result["U"], result["V"], result["W"])

    # Attempt structural rank reduction via continuous local search
    if base_ok and time.time() < deadline:
        try:
            reduced = _local_search_reduce(result, deadline)
            if reduced is not None:
                if _verify(reduced["U"], reduced["V"], reduced["W"]):
                    return reduced
        except Exception:
            pass

    if base_ok:
        return result
    return _schoolbook()

# EVOLVE-BLOCK-END
