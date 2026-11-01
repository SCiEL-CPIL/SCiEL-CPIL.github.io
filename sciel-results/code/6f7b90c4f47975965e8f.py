import time
import numpy as np
import random

DIMENSIONS = [2, 5, 6]
DEADLINE = 115.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # 1. Schoolbook Decomposition (Rank = n*m*p)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    # Strategy: Try to find a lower rank via a structured search.
    # We will try ranks from 59 down to 55.
    # We use a randomized search with a "greedy" initialization.
    
    def check_solution(U, V, W, rank):
        """
        Checks if U, V, W (integer/half-integer arrays) satisfy the identity exactly.
        Identity: sum_r U[i*m+j1, r] V[j2*p+k, r] W[k*n+i, r] = delta(j1, j2)
        """
        # Convert to float for exact check? No, use integer if possible.
        # The problem allows half-integers. We can multiply by 2 to check integers.
        # But the input might be floats. Let's check with tolerance 1e-9.
        for i in range(n):
            for k in range(p):
                U_slice = U[i*m:(i+1)*m, :] # (m, rank)
                V_slice = np.empty((m, rank))
                for j in range(m):
                    V_slice[j, :] = V[j*p + k, :]
                w_val = W[k*n + i, :] # (rank,)
                
                prod = U_slice * w_val[None, :]
                mat = prod @ V_slice.T
                
                if not np.allclose(mat, np.eye(m), atol=1e-9):
                    return False
        return True

    # Search for rank < 60
    # We try a few random restarts for rank 59, 58, etc.
    # To make it more likely, we initialize with the schoolbook solution 
    # and try to "compress" it by finding linear dependencies or 
    # replacing pairs of terms with a single term.
    
    # Since schoolbook terms are independent, we must find a new basis.
    # We will use a random search over integer entries in {-2, -1, 0, 1, 2}.
    
    max_search_time = 110.0
    # Try a few ranks
    for target_rank in range(rank_schoolbook - 1, max(rank_schoolbook - 10, 29), -1):
        if time.time() - start_time > max_search_time:
            break
        
        # Limit restarts per rank to ensure we finish
        restarts = 200
        for _ in range(restarts):
            if time.time() - start_time > max_search_time:
                break
            
            # Random initialization with small integers
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Fill with sparse random integers
            # Density 0.2
            for r in range(target_rank):
                # U
                if random.random() < 0.3:
                    idx = random.randint(0, n*m - 1)
                    U[idx, r] = random.choice([-2, -1, 1, 2])
                # V
                if random.random() < 0.3:
                    idx = random.randint(0, m*p - 1)
                    V[idx, r] = random.choice([-2, -1, 1, 2])
                # W
                if random.random() < 0.3:
                    idx = random.randint(0, p*n - 1)
                    W[idx, r] = random.choice([-2, -1, 1, 2])
            
            if check_solution(U, V, W, target_rank):
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                # If we find a valid solution, we can break early if we are confident
                # or continue to see if we can go lower.
                # Given the difficulty, we break to save time.
                break

    # Return the best found
    # Ensure we return lists of numbers
    U_out = best_U
    V_out = best_V
    W_out = best_W
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
