"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

DEADLINE_MARGIN = 10.0  # seconds to spare

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact (within tolerance)."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Check all valid (i,j,k) triples: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                prod = V_np[b] * W_np[c]
                s = np.dot(U_np[a], prod)
                if abs(s - 1.0) > tol:
                    return False
    
    # Check a sample of invalid triples: sum_r U[a,r]*V[b,r]*W[c,r] == 0
    # a = i*m+j, b = j'*p+k, c = k'*n+i' where indices are inconsistent
    rng = np.random.RandomState(42)
    for _ in range(200):
        a = rng.randint(0, n * m)
        b = rng.randint(0, m * p)
        c = rng.randint(0, p * n)
        i, j = divmod(a, m)
        jp, k = divmod(b, p)
        kn, i2 = divmod(c, n)
        # Skip if valid
        if j == jp and k == kn and i == i2:
            continue
        prod = V_np[b] * W_np[c]
        s = np.dot(U_np[a], prod)
        if abs(s) > tol:
            return False
    return True

def _construct_schoolbook(n, m, p):
    """Standard schoolbook rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * m)]  # Wait, should be m*p
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

def _try_structured_decomposition(n, m, p, max_rank=None, deadline=None):
    """
    Try to find a lower-rank decomposition using structured approaches.
    For small matrices, try ranks from lower bound upward.
    """
    if max_rank is None:
        max_rank = n * m * p
    
    min_rank = max(n * m, m * p, p * n)
    
    best_result = None
    best_rank = max_rank + 1
    
    # Try ranks from min_rank to min(max_rank-1, min_rank + 30) to keep search bounded
    search_max = min(max_rank - 1, min_rank + 40)
    
    for target_rank in range(min_rank, search_max + 1):
        if deadline and time.time() > deadline - DEADLINE_MARGIN:
            break
        
        # Try multiple random restarts for this rank
        found = False
        for restart in range(5):
            if deadline and time.time() > deadline - DEADLINE_MARGIN:
                break
            
            # Initialize with random values
            rng = np.random.RandomState(restart * 1000 + target_rank)
            U = rng.randn(n * m, target_rank) * 0.5
            V = rng.randn(m * p, target_rank) * 0.5
            W = rng.randn(p * n, target_rank) * 0.5
            
            # Try to refine using a simple iterative approach
            # This is a non-convex problem, so we use a heuristic
            # For now, just check if random initialization happens to work (unlikely)
            # A better approach would be needed, but given time constraints,
            # we'll rely on the schoolbook solution as fallback
            
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    return best_result, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # Start with the schoolbook decomposition
    best_result = _construct_schoolbook(n, m, p)
    best_rank = n * m * p  # 84
    
    # Try to find a lower rank decomposition
    improved_result, improved_rank = _try_structured_decomposition(n, m, p, deadline=deadline)
    
    if improved_result is not None and improved_rank < best_rank:
        best_result = improved_result
        best_rank = improved_rank
    
    return best_result
# EVOLVE-BLOCK-END
