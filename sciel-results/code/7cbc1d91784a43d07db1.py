"""Matrix multiplication 4x4x8 - find rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    def make_schoolbook():
        """Schoolbook decomposition: rank = n*m*p = 128."""
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

    best_result = make_schoolbook()
    best_rank = n * m * p

    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    import random
    rng = random.Random(12345)

    # Strategy: try to find a lower-rank decomposition
    # Use structured random search with small integer entries
    # The tensor is M_{i,j,k} = 1 for all (i,j,k)
    
    # Known: for small matrix multiplication, schoolbook is often optimal
    # but let's try ranks from 127 down to see if we can find anything
    
    # Use a more structured approach: try decompositions where each
    # rank-1 term has a specific pattern
    
    # Try ranks from best_rank-1 down to a lower bound
    # Lower bound: at least max(n*m, m*p, p*n) = max(16,32,32) = 32
    # But actual lower bounds are much higher
    
    # Let's try a few ranks near 128 with randomized structured search
    for target_rank in range(best_rank - 1, max(100, best_rank - 30), -1):
        if time.time() > deadline - 15:
            break
        
        # Try a few random structured decompositions
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            # Generate sparse random matrices with entries in {-1, 0, 1}
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            for c in range(target_rank):
                # Each column of U: set 1-4 random entries to +/-1
                num_u = rng.randint(1, 4)
                rows_u = rng.sample(range(n * m), num_u)
                for row in rows_u:
                    U[row, c] = rng.choice([-1, 1])
                
                num_v = rng.randint(1, 4)
                rows_v = rng.sample(range(m * p), num_v)
                for row in rows_v:
                    V[row, c] = rng.choice([-1, 1])
                
                num_w = rng.randint(1, 4)
                rows_w = rng.sample(range(p * n), num_w)
                for row in rows_w:
                    W[row, c] = rng.choice([-1, 1])
            
            if verify(U, V, W):
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = target_rank
                break  # Found a lower rank, try even lower
    
    return best_result
