"""Search for a rank-minimal exact tensor decomposition of 2x3 x 3x9 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]
DEADLINE = 135.0  # safety margin before 150s kill

def _target_tensor(n, m, p):
    T = np.zeros((n, m, p), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T

def _verify(U, V, W, n, m, p, tol=0):
    """Check that sum_c U[:,c] outer V[:,c] outer W[:,c] equals T[i,j,k]=1."""
    rank = len(U[0])
    T = np.zeros((n, m, p), dtype=object)
    for c in range(rank):
        uc = np.array(U[:, c], dtype=float)
        vc = np.array(V[:, c], dtype=float)
        wc = np.array(W[:, c], dtype=float)
        for i in range(n):
            ui = uc[i * m]
            if ui == 0:
                continue
            for j in range(m):
                uij = uc[i * m + j]
                if uij == 0:
                    continue
                for k in range(p):
                    vjk = vc[j * p + k]
                    if vjk == 0:
                        continue
                    T[i, j, k] = T[i, j, k] + uij * vjk * wc[k * n + i]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i, j, k] - 1) > tol:
                    return False
    return True

def _to_lists(mat):
    return [[float(x) for x in row] for row in mat]

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    rank0 = n * m * p
    # Schoolbook baseline
    U = [[0.0] * rank0 for _ in range(n * m)]
    V = [[0.0] * rank0 for _ in range(m * p)]
    W = [[0.0] * rank0 for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    best = {"U": U, "V": V, "W": W}
    best_rank = rank0

    rng = random.Random(12345)

    def try_candidate(Uc, Vc, Wc, rank):
        nonlocal best, best_rank
        if rank >= best_rank:
            return
        if _verify(Uc, Vc, Wc, n, m, p):
            best = {"U": _to_lists(Uc), "V": _to_lists(Vc), "W": _to_lists(Wc)}
            best_rank = rank
            return True
        return False

    # Strategy: random search over half-integer vectors.
    # Each term is (u, v, w) with entries in {0, 0.5, 1}.
    # We try random sets of terms and check if they sum to T.
    # To make it efficient, we use a greedy/repair approach.

    # Try to find a decomposition with rank < best_rank by random restarts.
    max_rank_to_try = best_rank - 1
    attempts = 0
    while time.time() - start < DEADLINE:
        attempts += 1
        # Pick a target rank to try
        target_rank = rng.randint(max(1, max_rank_to_try - 20), max_rank_to_try)
        # Generate random half-integer vectors
        Uc = np.zeros((n * m, target_rank))
        Vc = np.zeros((m * p, target_rank))
        Wc = np.zeros((p * n, target_rank))
        for c in range(target_rank):
            for idx in range(n * m):
                Uc[idx, c] = rng.choice([0, 0.5, 1])
            for idx in range(m * p):
                Vc[idx, c] = rng.choice([0, 0.5, 1])
            for idx in range(p * n):
                Wc[idx, c] = rng.choice([0, 0.5, 1])
        try_candidate(Uc, Vc, Wc, target_rank)

        # Also try a structured approach: start from schoolbook and randomly zero out
        # some entries, then check.
        if attempts % 3 == 0:
            Uc = np.array(U, dtype=float)
            Vc = np.array(V, dtype=float)
            Wc = np.array(W, dtype=float)
            # Randomly set some entries to 0 or 0.5
            for c in range(rank0):
                if rng.random() < 0.3:
                    Uc[rng.randint(0, n * m - 1), c] = rng.choice([0, 0.5])
                if rng.random() < 0.3:
                    Vc[rng.randint(0, m * p - 1), c] = rng.choice([0, 0.5])
                if rng.random() < 0.3:
                    Wc[rng.randint(0, p * n - 1), c] = rng.choice([0, 0.5])
            # Try to drop some columns
            if rng.random() < 0.5:
                keep = sorted(rng.sample(range(rank0), rng.randint(max(1, rank0 - 5), rank0)))
                Uc = Uc[:, keep]
                Vc = Vc[:, keep]
                Wc = Wc[:, keep]
            try_candidate(Uc, Vc, Wc, Uc.shape[1])

    return best
