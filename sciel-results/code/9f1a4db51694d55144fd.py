"""Matrix multiplication 3x5x7: search for low-rank exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=3, m=5, p=7, tol=1e-6):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c] for all i,j,k."""
    r = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c]
    # W[k*n+i, c]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = U[i*m+j] * V[j*p+k]
                lhs = np.sum(lhs)
                rhs = np.sum(W[k*n+i])
                err = abs(lhs - rhs)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _verify_tensor(U, V, W, n=3, m=5, p=7, tol=1e-6):
    """Verify tensor identity: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
       should equal 1 for all (i,j,k)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(np.dot(U[i*m+j], V[j*p+k]), W[k*n+i])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _schoolbook(n=3, m=5, p=7):
    rank = n * m * p
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1.0
                V[j*p+k][c] = 1.0
                W[k*n+i][c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}

def _random_half_integer():
    """Random value from {-1, -0.5, 0, 0.5, 1}."""
    return random.choice([-1.0, -0.5, 0.0, 0.5, 1.0])

def _random_candidate(rank, n=3, m=5, p=7, density=0.5):
    """Generate a random sparse candidate with half-integer entries."""
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    for row in range(n*m):
        for col in range(rank):
            if random.random() < density:
                U[row][col] = _random_half_integer()
    for row in range(m*p):
        for col in range(rank):
            if random.random() < density:
                V[row][col] = _random_half_integer()
    for row in range(p*n):
        for col in range(rank):
            if random.random() < density:
                W[row][col] = _random_half_integer()
    return {"U": U, "V": V, "W": W}

def _local_search(start_rank, n=3, m=5, p=7, max_iter=200, deadline=None):
    """Try to find a valid decomposition at or below start_rank via random restarts."""
    best = None
    best_rank = start_rank
    for rank in range(start_rank, max(10, start_rank - 20), -1):
        if deadline and time.time() > deadline - 5:
            break
        for attempt in range(max_iter):
            if deadline and time.time() > deadline - 5:
                break
            cand = _random_candidate(rank, n, m, p, density=0.3)
            if _verify_tensor(cand["U"], cand["V"], cand["W"], n, m, p):
                best = cand
                best_rank = rank
                return best, best_rank
    return best, best_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill

    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 105

    # Try to find lower rank via random search
    # Known bounds: for 3x5x7, schoolbook is 105.
    # Try ranks from 104 downward.
    for target_rank in range(best_rank - 1, max(20, best_rank - 40), -1):
        if time.time() > deadline - 10:
            break
        found, found_rank = _local_search(target_rank, n, m, p,
                                           max_iter=50, deadline=deadline)
        if found is not None:
            best = found
            best_rank = found_rank
            break

    return best
# EVOLVE-BLOCK-END
