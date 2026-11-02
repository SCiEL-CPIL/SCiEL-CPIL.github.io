"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 4, 7]
_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != Fraction(1):
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using floating point as a quick filter."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct_schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
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

def _random_decomposition(n, m, p, rank, rng):
    """Generate a random decomposition with entries from {-1, 0, 1} and check if it works."""
    U = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
    V = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
    W = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
    return U, V, W

def _try_structured_search(n, m, p, max_rank_to_try, min_rank, rng):
    """Try to find a decomposition with rank between min_rank and max_rank_to_try."""
    for rank in range(min_rank, max_rank_to_try + 1):
        if time.time() - _START_TIME > _TIME_LIMIT:
            return None
        # Try several random initializations
        for attempt in range(200):
            if time.time() - _START_TIME > _TIME_LIMIT:
                return None
            U, V, W = _random_decomposition(n, m, p, rank, rng)
            # Quick float check
            if _verify_fast(U, V, W, n, m, p):
                # Exact check
                if _verify_exact(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 56
    
    # Start with schoolbook as guaranteed fallback
    best = _construct_schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    min_possible = max(n * m, m * p, p * n)
    
    # Try ranks from min_possible up to schoolbook_rank - 1
    rng = random.Random(42)
    
    # First try a few ranks near the lower bound with more attempts
    for rank in range(min_possible, min(schoolbook_rank, min_possible + 15)):
        if time.time() - _START_TIME > _TIME_LIMIT:
            break
        result = _try_structured_search(n, m, p, rank, rank, rng)
        if result is not None:
            best = result
            best_rank = rank
            break  # Found the lowest rank, stop
    
    # If not found near lower bound, try a wider range with fewer attempts
    if best_rank == schoolbook_rank and time.time() - _START_TIME < _TIME_LIMIT:
        for rank in range(min_possible + 15, schoolbook_rank):
            if time.time() - _START_TIME > _TIME_LIMIT:
                break
            # Fewer attempts for higher ranks
            U = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
            if _verify_fast(U, V, W, n, m, p):
                if _verify_exact(U, V, W, n, m, p):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    break
    
    return best

# EVOLVE-BLOCK-END
