"""Matrix multiplication 2,4,6 - rank-minimal exact tensor decomposition search."""
import time
import numpy as np
import random

_DIMENSIONS = (2, 4, 6)
_DEADLINE = 135.0  # safety margin before 150s kill


def _verify(n, m, p, U, V, W):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
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


def _random_search(n, m, p, start_rank, end_rank, budget_seconds):
    """
    Random tensor-network search for a low-rank decomposition.
    Tries random Gaussian initializations and does gradient descent
    on the squared residual of the tensor entries.
    """
    best_rank = None
    best_result = None
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

    for rank in range(start_rank, max(end_rank, start_rank - 1), -1):
        if time.time() - start_time > budget_seconds:
            break
        # If we've already found a better rank, stop
        if best_rank is not None and rank >= best_rank:
            break

        max_restarts = 4
        for restart in range(max_restarts):
            if time.time() - start_time > budget_seconds:
                break

            U = rng.standard_normal((n * m, rank))
            V = rng.standard_normal((m * p, rank))
            W = rng.standard_normal((p * n, rank))

            # Simple gradient descent with adaptive step
            lr = 0.01
            best_loss = np.inf
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            patience = 0
            max_patience = 80

            for step in range(2000):
                if time.time() - start_time > budget_seconds:
                    break

                # Compute loss
                loss = 0.0
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)

                for r in range(rank):
                    contrib = U[idx_a, r] * V[idx_b, r] * W[idx_c, r]
                    loss += np.sum((contrib - 1.0) ** 2)

                    # Gradients
                    # d/dU[a,r] = 2*(contrib-1)*V[b,r]*W[c,r]
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

                # Normalize gradients
                gnorm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                if gnorm > 1e-12:
                    dU /= gnorm
                    dV /= gnorm
                    dW /= gnorm
                    # Adaptive step size
                    U -= lr * dU
                    V -= lr * dV
                    W -= lr * dW
                    # Reduce learning rate if loss increasing
                    if loss > best_loss * 1.1:
                        lr *= 0.5
                    elif loss < best_loss * 0.9:
                        lr *= 1.1
                    lr = min(lr, 0.1)
                else:
                    break

            # Check if we found a valid decomposition
            if best_loss < 1e-9:
                # Snap to nearest half-integers
                U_snap = np.round(best_U * 2) / 2
                V_snap = np.round(best_V * 2) / 2
                W_snap = np.round(best_W * 2) / 2
                if _verify(n, m, p, U_snap, V_snap, W_snap):
                    return rank, U_snap, V_snap, W_snap
                else:
                    # Try original (non-snapped)
                    if _verify(n, m, p, best_U, best_V, best_W):
                        return rank, best_U, best_V, best_W

    return None, None, None, None


def construct():
    n, m, p = _DIMENSIONS
    start = time.time()

    # First, get the known rank-42 Strassen decomposition as a safe fallback
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
        return {"U": U, "V": V, "W": W}

    # Try random search for a lower rank
    best_rank = 42
    best_U, best_V, best_W = U_s, V_s, W_s

    # Time budget for search
    search_budget = min(_DEADLINE - (time.time() - start) - 5, 120)

    if search_budget > 10:
        # Try ranks from 38 down to 34
        for target_rank in range(38, 33, -1):
            if time.time() - start > _DEADLINE - 5:
                break
            remaining = _DEADLINE - (time.time() - start) - 3
            if remaining < 5:
                break

            rank_found, U_f, V_f, W_f = _random_search(
                n, m, p, target_rank, target_rank, min(remaining, 25)
            )
            if rank_found is not None and rank_found < best_rank:
                best_rank = rank_found
                best_U, best_V, best_W = U_f, V_f, W_f
                break  # Found a better rank, stop searching lower

    # Final verification
    if not _verify(n, m, p, best_U, best_V, best_W):
        best_U, best_V, best_W = U_s, V_s, W_s

    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]

    return {"U": U_list, "V": V_list, "W": W_list}
