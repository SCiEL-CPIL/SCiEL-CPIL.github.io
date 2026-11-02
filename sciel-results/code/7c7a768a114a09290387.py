"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # For each output element (i, k), check:
    # sum_c U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] 
    # Actually, the tensor is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And T[i,j,k] should be 1 if the (i,k) entry of (A*B) comes from A[i,j]*B[j,k]
    # Wait, let me re-read the convention.
    # W[k*n+i, r] - this is the output factor.
    # The tensor for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
    # T[i, j, k] = 1 for all valid i,j,k (since C[i,k] = sum_j A[i,j] * B[j,k])
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that doesn't look right dimensionally. Let me think again.
    
    # Standard tensor decomposition for matmul:
    # T_{i,j,k} = sum_c U_{(i,j),c} * V_{(j,k),c} * W_{(i,k),c}
    # where T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # But the task says W has shape (p*n, rank) with convention W[k*n+i, r]
    # So W[k*n+i, c] corresponds to output position (i, k)
    
    # Let me verify: for each (i, j, k), we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if s != 1:
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

def _random_decomp_attempt(n, m, p, rank, seed=None):
    """Try to find a rank-`rank` decomposition using random initialization and iterative refinement."""
    if seed is not None:
        random.seed(seed)
    
    # Initialize with random small integers
    U = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(n * m)]
    V = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(m * p)]
    W = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}

def _try_rank(n, m, p, rank, max_iters=5000, deadline=None):
    """Try to find a valid decomposition of given rank."""
    if deadline is None:
        deadline = time.time() + 10
    
    for attempt in range(max_iters):
        if time.time() > deadline:
            return None
        
        # Random initialization
        U = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(p * n)]
        
        # Check if this random assignment works
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        
        # Try targeted fixes: for each (i,j,k) where the sum is wrong, adjust
        # This is a heuristic approach
        
    return None

def _greedy_decomposition(n, m, p, max_rank, deadline):
    """Try to build a decomposition greedily, one rank at a time."""
    # Start with schoolbook and try to merge terms
    # For now, try various ranks from low to high
    for rank in range(1, max_rank + 1):
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, rank, max_iters=1000, deadline=deadline)
        if result is not None:
            return result
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 30
    
    # Get schoolbook as fallback
    best = _schoolbook(n, m, p)
    
    # Try to find lower rank decompositions
    # The theoretical lower bound for 2x3 * 3x5 is at least max(n*m, m*p, n*p) = max(6, 15, 10) = 15
    # But actual minimum rank is likely higher.
    
    deadline = time.time() + 120  # Leave safety margin
    
    # Try ranks from 15 up to schoolbook_rank - 1
    for rank in range(15, schoolbook_rank):
        if time.time() > deadline - 5:  # Leave 5s margin
            break
        result = _try_rank(n, m, p, rank, max_iters=2000, deadline=deadline - 5)
        if result is not None:
            best = result
            break  # Found a valid one, return it
    
    return best
# EVOLVE-BLOCK-END
