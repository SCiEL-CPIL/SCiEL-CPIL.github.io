"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _try_random_rank(n, m, p, target_rank, num_trials=100, seed=42):
    """Try to find a decomposition of the given rank using random search."""
    random.seed(seed)
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    # Candidate values: small integers and half-integers
    candidates = [-1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    for trial in range(num_trials):
        U = [[random.choice(candidates) for _ in range(target_rank)] for _ in range(U_size)]
        V = [[random.choice(candidates) for _ in range(target_rank)] for _ in range(V_size)]
        W = [[random.choice(candidates) for _ in range(target_rank)] for _ in range(W_size)]
        
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # First, try to find a lower-rank decomposition
    schoolbook_rank = n * m * p  # 45
    
    # Try ranks from just below schoolbook down to a reasonable minimum
    for target_rank in range(schoolbook_rank - 1, max(10, schoolbook_rank // 2), -1):
        if time.time() > deadline:
            break
        result = _try_random_rank(n, m, p, target_rank, num_trials=50, seed=target_rank)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
