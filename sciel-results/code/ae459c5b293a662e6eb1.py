"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # and check it equals 1 if i,j,k are all valid (which they are), else 0.
    # Actually for matrix mult: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So tensor T[i,j,k] should be 1 for all valid i,j,k, and the decomposition
    # should give: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(rank, n, m, p, deadline, max_trials=1000):
    """Try to find a rank-rank decomposition via random search with local refinement."""
    start_time = time.time()
    for trial in range(max_trials):
        if time.time() > deadline:
            return None
        # Initialize with small random values
        U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n*m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m*p)]
        W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p*n)]
        
        # Local search: try flipping individual entries
        improved = True
        while improved:
            if time.time() > deadline:
                return None
            improved = False
            # Pick a random tensor entry (i,j,k) and a random factor component
            for _ in range(10):
                if time.time() > deadline:
                    return None
                i = random.randint(0, n-1)
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                c = random.randint(0, rank-1)
                
                # Try flipping U[i*m+j][c]
                old_u = U[i*m+j][c]
                U[i*m+j][c] = -old_u
                if _verify(U, V, W, n, m, p):
                    return U, V, W
                U[i*m+j][c] = old_u
                
                # Try setting U[i*m+j][c] to 0
                U[i*m+j][c] = 0
                if _verify(U, V, W, n, m, p):
                    return U, V, W
                U[i*m+j][c] = old_u
                
                # Try flipping V[j*p+k][c]
                old_v = V[j*p+k][c]
                V[j*p+k][c] = -old_v
                if _verify(U, V, W, n, m, p):
                    return U, V, W
                V[j*p+k][c] = old_v
                
                # Try flipping W[k*n+i][c]
                old_w = W[k*n+i][c]
                W[k*n+i][c] = -old_w
                if _verify(U, V, W, n, m, p):
                    return U, V, W
                W[k*n+i][c] = old_w
    
    return None

def _structured_decomposition(n, m, p):
    """Try structured decompositions based on known algorithms."""
    # For small matrices, try various structured approaches
    
    # Approach 1: Block decomposition
    # If we can factor the dimensions, we might get better rank
    
    # Approach 2: Use the fact that rank >= max(n*m, m*p, n*p)
    # Lower bound for 3x4x8: max(12, 32, 24) = 32
    
    # Try ranks from lower bound upward
    lb = max(n*m, m*p, n*p)
    for rank in range(lb, n*m*p + 1):
        # For now, just return schoolbook if we don't find better
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    # First, get the schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_solution = (U_fb, V_fb, W_fb)
    
    # Try to find better solutions with lower rank
    lower_bound = max(n*m, m*p, n*p)
    
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        result = _try_rank(rank, n, m, p, deadline, max_trials=50)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_solution = (U, V, W)
            # Found a solution, try to go even lower
            continue
    
    return {"U": best_solution[0], "V": best_solution[1], "W": best_solution[2]}
# EVOLVE-BLOCK-END
