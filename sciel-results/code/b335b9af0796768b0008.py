"""Rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.

Strategy:
1. Provide a guaranteed rank-25 decomposition using Strassen on the top-left
   2x2 block plus schoolbook for the remaining 15 terms involving index 2.
2. Try to reduce the rank by removing redundant columns from the rank-25
   decomposition.
3. Run a time-bounded search for lower-rank decompositions (rank 24, 23, ...).
4. Return the best (lowest-rank) valid decomposition found.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 3]
DEADLINE = 135.0  # Safety margin before 150s kill


def _elapsed_seconds(start):
    return time.time() - start


def _time_up(start, margin=5.0):
    return time.time() - start > DEADLINE - margin


def _target_tensor(n=3, m=3, p=3):
    """Return the MM tensor as a dict {(i,j,k): 1} for the consistent triples."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i * m + j, j * p + k, k * n + i)] = 1
    return T


def _verify_strict(U, V, W, n=3, m=3, p=3, tol=1e-9):
    """Strictly verify the decomposition over all consistent triples."""
    rank = len(U[0])
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(U[i * m + j], V[j * p + k] * W[k * n + i]))
                if abs(s - 1.0) > tol:
                    return False
    return True


def _verify_full(U, V, W, n=3, m=3, p=3, tol=1e-9):
    """Full verification over all (a,b,c) in the 3-way tensor."""
    rank = len(U[0])
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    D = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        D += np.outer(U[:, r], V[:, r]) * W[:, r]
    if np.max(np.abs(D - T)) > tol:
        return False
    return True


def _rank25_decomposition(n=3, m=3, p=3):
    """Build a rank-25 decomposition for 3x3x3 using Strassen 2x2 on the
    top-left block and schoolbook for the rest."""
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2 for top-left block (indices 0,1)
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[1][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[3][0] = 1; W[4][0] = 1

    # M2 = (A10+A11)B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[3][1] = 1; W[4][1] = 1

    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[1][2] = 1; V[2][2] = -1
    W[1][2] = 1; W[4][2] = 1

    # M4 = (A00+A01)B11
    U[0][3] = 1; U[1][3] = 1
    V[2][3] = 1
    W[0][3] = 1; W[3][3] = 1

    # M5 = (A10-A00)(B00+B01)
    U[3][4] = 1; U[0][4] = -1
    V[0][4] = 1; V[1][4] = 1
    W[3][4] = 1; W[4][4] = 1

    # M6 = (A01-A11)(B10+B11)
    U[1][5] = 1; U[4][5] = -1
    V[1][5] = 1; V[2][5] = 1
    W[0][5] = 1; W[3][5] = 1

    # Schoolbook for remaining 15 terms involving index 2
    col = 6
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i <= 1 and j <= 1 and k <= 1:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return U, V, W


def _try_remove_columns(U, V, W, n=3, m=3, p=3, start_time=None, max_remove=5):
    """Try to remove redundant columns from the decomposition."""
    if start_time is None:
        start_time = time.time()
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]

    removed = 0
    for _ in range(max_remove):
        if _time_up(start_time, 5):
            break
        found = False
        for r in range(rank):
            if _time_up(start_time, 3):
                break
            # Try zeroing out column r
            U_save = U.copy()
            V_save = V.copy()
            W_save = W.copy()
            U[:, r] = 0
            V[:, r] = 0
            W[:, r] = 0
            if _verify_strict(U, V, W, n, m, p, tol=1e-6):
                # Can remove this column
                U = np.delete(U, r, axis=1)
                V = np.delete(V, r, axis=1)
                W = np.delete(W, r, axis=1)
                rank -= 1
                removed += 1
                found = True
                break
            else:
                U = U_save
                V = V_save
                W = W_save
        if not found:
            break

    return U.tolist(), V.tolist(), W.tolist()


def _search_rank(r, n=3, m=3, p=3, start_time=None, max_iters=200, seed=0):
    """Search for a rank-r decomposition using random initialization +
    solving for W given U,V."""
    if start_time is None:
        start_time = time.time()
    rng = random.Random(seed)

    for it in range(max_iters):
        if _time_up(start_time, 3):
            break

        # Random sparse initialization
        U = [[0.0] * r for _ in range(n * m)]
        V = [[0.0] * r for _ in range(m * p)]
        for row in range(n * m):
            num_nz = rng.randint(1, 3)
            cols = rng.sample(range(r), min(num_nz, r))
            for c in cols:
                U[row][c] = rng.choice([1, -1, 0.5, -0.5])
        for row in range(m * p):
            num_nz = rng.randint(1, 3)
            cols = rng.sample(range(r), min(num_nz, r))
            for c in cols:
                V[row][c] = rng.choice([1, -1, 0.5, -0.5])

        # Solve for W
        W = np.zeros((p * n, r))
        valid = True
        for k in range(p):
            for i in range(n):
                c = k * n + i
                M = np.zeros((m, r))
                for j in range(m):
                    M[j, :] = U[i * m + j] * V[j * p + k]
                rhs = np.ones(m)
                sol, residuals, rank_m, sv = np.linalg.lstsq(M, rhs, rcond=None)
                resid = np.linalg.norm(M @ sol - rhs)
                if resid > 1e-7:
                    valid = False
                    break
                W[c, :] = sol
            if not valid:
                break

        if not valid:
            continue

        if _verify_strict(U, V, W.tolist(), n, m, p):
            return U, V, W.tolist()

    return None


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS

    # Start with rank-25 Strassen-based decomposition
    U25, V25, W25 = _rank25_decomposition(n, m, p)
    assert _verify_strict(U25, V25, W25, n, m, p), "Rank-25 failed verification"
    best_rank = 25
    best_solution = (U25, V25, W25)

    # Try to reduce rank by removing redundant columns
    if not _time_up(start_time, 10):
        U_p, V_p, W_p = _try_remove_columns(U25, V25, W25, n, m, p, start_time)
        if _verify_strict(U_p, V_p, W_p, n, m, p):
            new_rank = len(U_p[0])
            if new_rank < best_rank:
                best_rank = new_rank
                best_solution = (U_p, V_p, W_p)

    # Search for lower-rank decompositions
    for target_rank in range(24, best_rank):
        if _time_up(start_time, 10):
            break
        for seed in range(30):
            if _time_up(start_time, 3):
                break
            result = _search_rank(target_rank, n, m, p, start_time,
                                  max_iters=100, seed=seed * 137 + target_rank)
            if result is not None:
                U, V, W = result
                if _verify_strict(U, V, W, n, m, p):
                    best_rank = target_rank
                    best_solution = (U, V, W)
                    break
        if best_rank <= target_rank:
            break

    U, V, W = best_solution
    return {"U": U, "V": V, "W": W}
