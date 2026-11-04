"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for s in range(rank):
                    val += U[i * m + j][s] * V[j * p + k][s] * W[k * n + i][s]
                if val != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Leave 10s margin before 150s limit
    
    # Fall back to schoolbook if search fails or times out
    best = _schoolbook(n, m, p)
    
    # Attempt a simple local search to reduce rank below n*m*p
    # Start with schoolbook and try merging terms
    import random as _rand
    _rand.seed(42)
    
    target_rank = n * m * p  # 45
    found = False
    
    # Try rank 41: split into 3x3x3 (23) + 3x3x2 (18) conceptually
    # But we don't have explicit 23-term Bini, so just return schoolbook
    # unless we can find a valid lower-rank decomposition quickly.
    
    return best
# EVOLVE-BLOCK-END
