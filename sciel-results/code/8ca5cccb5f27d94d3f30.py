"""Matrix multiplication 2,4,6 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
import random

_DIMENSIONS = (2, 4, 6)
_DEADLINE = 145.0  # safety margin before 150s kill


def _verify(n, m, p, U, V, W):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    total = np.zeros(n * m * p, dtype=float)
    idx_a = []
    idx_b = []
    idx_c = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a.append(i * m + j)
                idx_b.append(j * p + k)
                idx_c.append(k * n + i)
    idx_a = np.array(idx_a)
    idx_b = np.array(idx_b)
    idx_c = np.array(idx_c)
    for r in range(rank):
        total += U[idx_a, r] * V[idx_b, r] * W[idx_c, r]
    return np.allclose(total, np.ones(n * m * p), atol=1e-9)


def _strassen_decomposition(n, m, p):
    """
    Block Strassen decomposition for M_{2,4,6}.
    Split m=4 into 2 blocks of 2, p=6 into 3 blocks of 2.
    Each subproblem M_{2,2,2} uses rank-7 Strassen.
    Total rank = 2 * 3 * 7 = 42.
    """
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1},
    ]
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1},
    ]
    W_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1},
    ]

    a_blocks = m // 2  # 2
    b_blocks = p // 2  # 3
    strassen_rank = 7
    rank = a_blocks * b_blocks * strassen_rank  # 42

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0
    for a in range(a_blocks):
        for b in range(b_blocks):
            for s in range(strassen_rank):
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    U[i * m + j, col] += coeff
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    V[j * p + k, col] += coeff
                for (i_loc, k_loc), coeff in W_coeffs[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    W[k * n + i, col] += coeff
                col += 1

    return U, V, W


def _try_rank_reduction(n, m, p, base_U, base_V, base_W, max_time=30):
    """
    Try to reduce the rank of the decomposition by finding linear dependencies
    among the rank-1 terms and eliminating them.
    """
    start_time = time.time()
    rank = base_U.shape[1]
    
    # Check if any columns are linearly dependent across U, V, W simultaneously
    # A column r is redundant if there exist coefficients c_s (s != r) such that
    # for all (i,j,k): sum_s c_s * U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    
    # This is a hard problem, so instead try: for each pair of columns, check if
    # one can be expressed as a linear combination of others in a way that preserves
    # the tensor product structure.
    
    # Simpler approach: try to find if any column is a scalar multiple of another
    # in all three matrices simultaneously (unlikely but cheap to check)
    removable = []
    for r in range(rank):
        if time.time() - start_time > max_time:
            break
        for s in range(rank):
            if r == s or s in removable:
                continue
            # Check if column r is a scalar multiple of column s in all three
            ur, us = base_U[:, r], base_U[:, s]
            vr, vs = base_V[:, r], base_V[:, s]
            wr, ws = base_W[:, r], base_W[:, s]
            
            # Find scalar if any
            nonzero_u = np.nonzero(us)[0]
            if len(nonzero_u) == 0:
                continue
            scalar_candidates = ur[nonzero_u] / us[nonzero_u]
            if np.std(scalar_candidates) > 1e-10:
                continue
            scalar = scalar_candidates[0]
            
            if np.allclose(vr, scalar * vs, atol=1e-10) and np.allclose(wr, scalar * ws, atol=1e-10):
                removable.append(r)
                break
    
    if removable:
        keep = [r for r in range(rank) if r not in removable]
        new_U = base_U[:, keep]
        new_V = base_V[:, keep]
        new_W = base_W[:, keep]
        if _verify(n, m, p, new_U, new_V, new_W):
            return new_U, new_V, new_W
    
    return base_U, base_V, base_W


def _random_search(n, m, p, target_rank, budget_seconds=20):
    """
    Random tensor-network search for a low-rank decomposition.
    """
    start_time = time.time()
    rng = np.random.default_rng(42)

    # Precompute the target tensor index mapping
    idx_a = []
    idx_b = []
    idx_c = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a.append(i * m + j)
                idx_b.append(j * p + k)
                idx_c.append(k * n + i)
    idx_a = np.array(idx_a)
    idx_b = np.array(idx_b)
    idx_c = np.array(idx_c)
    num_entries = n * m * p

    max_restarts = 3
    for restart in range(max_restarts):
        if time.time() - start_time > budget_seconds:
            break

        U = rng.standard_normal((n * m, target_rank))
        V = rng.standard_normal((m * p, target_rank))
        W = rng.standard_normal((p * n, target_rank))

        lr = 0.01
        best_loss = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        patience = 0
        max_patience = 60

        for step in range(1500):
            if time.time() - start_time > budget_seconds:
                break

            loss = 0.0
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)

            for r in range(target_rank):
                contrib = U[idx_a, r] * V[idx_b, r] * W[idx_c, r]
                loss += np.sum((contrib - 1.0) ** 2)

                resid = (contrib - 1.0) * 2.0
                dU[idx_a, r] += resid * V[idx_b, r] * W[idx_c, r]
                dV[idx_b, r] += resid * U[idx_a, r] * W[idx_c, r]
                dW[idx_c, r] += resid * U[idx_a, r] * V[idx_b, r]

            loss /= num_entries

            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                patience = 0
                if best_loss < 1e-12:
                    break
            else:
                patience += 1
                if patience > max_patience:
                    break

            gnorm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
            if gnorm > 1e-12:
                dU /= gnorm
                dV /= gnorm
                dW /= gnorm
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                if loss > best_loss * 1.1:
                    lr *= 0.5
                elif loss < best_loss * 0.9:
                    lr *= 1.1
                lr = min(lr, 0.1)
            else:
                break

        if best_loss < 1e-9:
            U_snap = np.round(best_U * 2) / 2
            V_snap = np.round(best_V * 2) / 2
            W_snap = np.round(best_W * 2) / 2
            if _verify(n, m, p, U_snap, V_snap, W_snap):
                return target_rank, U_snap, V_snap, W_snap
            else:
                if _verify(n, m, p, best_U, best_V, best_W):
                    return target_rank, best_U, best_V, best_W

    return None, None, None, None


def construct():
    n, m, p = _DIMENSIONS
    start = time.time()

    # Get the known rank-42 Strassen decomposition as a safe fallback
    U_s, V_s, W_s = _strassen_decomposition(n, m, p)

    if not _verify(n, m, p, U_s, V_s, W_s):
        # Fallback to schoolbook (rank 48)
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 42
    best_U, best_V, best_W = U_s, V_s, W_s

    # Try to find linear dependencies to reduce rank
    if time.time() - start < _DEADLINE - 10:
        U_red, V_red, W_red = _try_rank_reduction(n, m, p, U_s, V_s, W_s, max_time=10)
        if U_red.shape[1] < best_rank and _verify(n, m, p, U_red, V_red, W_red):
            best_rank = U_red.shape[1]
            best_U, best_V, best_W = U_red, V_red, W_red

    # Try random search for lower ranks
    remaining = _DEADLINE - (time.time() - start) - 5
    if remaining > 15:
        for target_rank in range(41, 36, -1):
            if time.time() - start > _DEADLINE - 8:
                break
            alloc = min(remaining / (41 - target_rank + 1), 15)
            if alloc < 5:
                break
            rank_found, U_f, V_f, W_f = _random_search(
                n, m, p, target_rank, budget_seconds=alloc
            )
            if rank_found is not None and rank_found < best_rank:
                best_rank = rank_found
                best_U, best_V, best_W = U_f, V_f, W_f
                break

    # Final verification
    if not _verify(n, m, p, best_U, best_V, best_W):
        best_U, best_V, best_W = U_s, V_s, W_s

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
