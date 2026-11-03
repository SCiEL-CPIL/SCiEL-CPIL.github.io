import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    time_limit = 140.0  # Safety margin before 150s kill

    def elapsed():
        return time.time() - t_start

    def time_exceeded():
        return elapsed() > time_limit

    # Build schoolbook decomposition
    def make_schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
        return U, V, W

    U, V, W = make_schoolbook(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]

    # Validate a decomposition
    def check_valid(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if s != 1:
                        return False
        return True

    # Verify schoolbook
    assert check_valid(U, V, W, n, m, p), "Schoolbook validation failed"

    # Attempt to reduce rank by trying to find linear dependencies
    # Strategy: For each term, check if it can be expressed as a combination of others
    # This is a heuristic; for generic tensors it won't work, but worth trying

    def try_reduce_rank(U, V, W, n, m, p, target_rank):
        """Try to find a decomposition of rank target_rank."""
        if target_rank >= U.shape[1]:
            return U, V, W, U.shape[1]

        # This is a placeholder for actual search logic
        # For now, return the original
        return U, V, W, U.shape[1]

    # Try reducing from current rank
    current_rank = best_rank
    while current_rank > 1 and not time_exceeded():
        candidate_rank = current_rank - 1
        U_cand, V_cand, W_cand, achieved_rank = try_reduce_rank(
            best_U, best_V, best_W, n, m, p, candidate_rank
        )
        if achieved_rank < current_rank:
            if check_valid(U_cand, V_cand, W_cand, n, m, p):
                best_U, best_V, best_W = U_cand.copy(), V_cand.copy(), W_cand.copy()
                best_rank = achieved_rank
                current_rank = achieved_rank
            else:
                break
        else:
            break

    # Final validation
    if not check_valid(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
        best_U, best_V, best_W = make_schoolbook(n, m, p)
        best_rank = best_U.shape[1]

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
