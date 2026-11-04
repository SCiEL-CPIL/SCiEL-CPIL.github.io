"""Combined program for matrix multiplication 2x3x7 rank minimization."""
import numpy as np
import random
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    deadline = time.time() + 140  # safety margin before 150s kill

    # Schoolbook as baseline
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    def verify_fast(U, V, W, rank):
        """Vectorized verification using numpy."""
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        U3 = U_np.reshape(n, m, rank)
        V3 = V_np.reshape(m, p, rank)
        W3 = W_np.reshape(p, n, rank)
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # We need T[i,j,k] == 1 for all i,j,k
        # And we also need to ensure that the decomposition is exact.
        # The tensor equation is sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all valid i,j,k.
        # Are there invalid i,j,k?
        # In M_{n,m,p}, all i in 1..n, j in 1..m, k in 1..p are valid.
        # So we just need sum == 1.
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.all(T == 1)

    def try_local_search(start_rank, max_steps):
        """
        Start from a random valid decomposition of start_rank (or schoolbook)
        and try to reduce rank by merging columns or setting entries to 0.
        This is complex. Instead, we try to find a solution of rank R < start_rank
        by random restarts but with a bias towards the schoolbook structure.
        """
        pass

    def try_random_rank(target_rank, max_attempts, value_range):
        """Try to find a decomposition of the given rank with random integer initialization."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
            
            # Initialize with small integers
            U = [[random.randint(value_range[0], value_range[1]) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.randint(value_range[0], value_range[1]) for _ in range(target_rank)] for _ in range(mp)]
            W = [[random.randint(value_range[0], value_range[1]) for _ in range(target_rank)] for _ in range(pn)]
            
            if verify_fast(U, V, W, target_rank):
                return {"U": U, "V": V, "W": W}
        return None

    # Try to find lower rank decompositions
    # Lower bound: max(nm, mp, pn) = 21
    # Known rank for 2x2x2 is 7.
    # For 2x3x7, we expect rank to be significantly lower than 42.
    # Let's try ranks from 30 down to 21? No, we want minimal, so 21 up.
    # But 21 is very hard.
    # Let's try a range of ranks that are likely to have solutions.
    
    # Strategy: Try ranks 30, 31, ..., 41.
    # If we find one, we can try to go lower.
    # But to be safe, we iterate from low to high.
    
    # To improve chances, we use a larger value range for integers, e.g., -2 to 2.
    # And we try more attempts for lower ranks.
    
    for target_rank in range(21, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Increase attempts for lower ranks as they are harder
        max_attempts = 1000 if target_rank < 35 else 200
        
        result = try_random_rank(target_rank, max_attempts, (-2, 2))
        if result:
            best = result
            best_rank = target_rank
            break

    return best
# EVOLVE-BLOCK-END
