"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i*m + j][c]) * Fraction(V[j*p + k][c]) * Fraction(W[k*n + i][c])
                if total != 1:
                    return False
    return True

def _try_random_decomposition(n, m, p, rank, num_trials=200, timeout=5):
    """Try to find a rank-rank decomposition using random search."""
    start_time = time.time()
    
    for trial in range(num_trials):
        if time.time() - start_time > timeout:
            break
        
        # Random initialization with small integers
        U = [[random.randint(-3, 3) for _ in range(rank)] for _ in range(n*m)]
        V = [[random.randint(-3, 3) for _ in range(rank)] for _ in range(m*p)]
        W = [[random.randint(-3, 3) for _ in range(rank)] for _ in range(p*n)]
        
        if _verify_decomposition(U, V, W, n, m, p, rank):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(4, 30, 30) = 30
    lower_bound = max(n*m, m*p, p*n)
    
    random.seed(42)
    
    # Try ranks from a reasonable range
    # For (2,2,15), let's try from 40 up to 59
    for rank in range(max(lower_bound, 40), rank_schoolbook):
        if time.time() > deadline:
            break
        
        result = _try_random_decomposition(n, m, p, rank, num_trials=300, timeout=4)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
            break
    
    return best_result
# EVOLVE-BLOCK-END

