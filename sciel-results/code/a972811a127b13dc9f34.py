"""Matrix multiplication 3x3 x 3x4 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 36

    # Precompute the target tensor entries: T[alpha, beta, gamma]
    # alpha = i*m + j (row of A * col of A)
    # beta  = j*p + k (row of B * col of B)
    # gamma = k*n + i (row of W, official trace convention)
    valid = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                valid.append((alpha, beta, gamma))
    valid_set = set(valid)

    # All (alpha, beta, gamma) triples
    all_triples = []
    for a in range(n * m):
        for b in range(m * p):
            for g in range(p * n):
                all_triples.append((a, b, g))
    invalid = [t for t in all_triples if t not in valid_set]

    def verify(U, V, W):
        """Check exact decomposition with half-integer entries."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        # Snap to half-integers
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        T = np.einsum('ac,bc,dc->abd', U, V, W)
        for (a, b, g) in valid:
            if abs(T[a, b, g] - 1.0) > 1e-9:
                return False
        for (a, b, g) in invalid:
            if abs(T[a, b, g]) > 1e-9:
                return False
        return True

    def to_lists(U, V, W):
        U = np.round(np.asarray(U, dtype=float) * 2) / 2
        V = np.round(np.asarray(V, dtype=float) * 2) / 2
        W = np.round(np.asarray(W, dtype=float) * 2) / 2
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook decomposition
    def schoolbook():
        rank = rank_schoolbook
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Try to find a lower-rank decomposition via structured random search + local refinement
    # The matrix multiplication tensor for 3x3 * 3x4 has 36 nonzero entries.
    # Theoretical lower bound is around 24-27 for these dimensions.
    # We try ranks from a conservative lower bound up to schoolbook-1.

    # For each rank, we do a few random restarts with {0, 0.5, 1, 1.5, 2} entries
    # and check if the decomposition is exact after snapping.
    # This is a heuristic; if it fails, we fall back to schoolbook.

    best_result = schoolbook()

    # Try a few low-rank candidates with structured initialization
    # We use a simple approach: random sparse matrices with small integer/half-integer values
    for r in range(24, rank_schoolbook):
        if time.time() > deadline:
            break

        found = False
        # Number of restarts: keep small to stay within time budget
        max_restarts = 5
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break

            # Random initialization with small values
            U = np.random.choice([0, 0.5, 1, 1.5, 2], size=(n * m, r))
            V = np.random.choice([0, 0.5, 1, 1.5, 2], size=(m * p, r))
            W = np.random.choice([0, 0.5, 1, 1.5, 2], size=(p * n, r))

            # Quick check: compute the tensor and see if it matches
            T = np.einsum('ac,bc,dc->abd', U, V, W)
            ok = True
            for (a, b, g) in valid:
                if abs(T[a, b, g] - 1.0) > 1e-6:
                    ok = False
                    break
            if ok:
                for (a, b, g) in invalid:
                    if abs(T[a, b, g]) > 1e-6:
                        ok = False
                        break
            if ok:
                # Snap to half-integers and verify
                if verify(U, V, W):
                    best_result = to_lists(U, V, W)
                    found = True
                    break

        if found:
            break

    return best_result
# EVOLVE-BLOCK-END
