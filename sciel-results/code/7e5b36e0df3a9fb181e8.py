import time
import sys
import itertools
from fractions import Fraction

DIMENSIONS = [3, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    global START_TIME, TIME_LIMIT
    
    # Trivial rank decomposition (baseline)
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_trivial
    best_result = {"U": U_trivial, "V": V_trivial, "W": W_trivial}
    
    # Search for lower rank decompositions
    # Strategy: try ranks from low to high, use integer coefficients in {-1, 0, 1}
    # For small matrices, we can enumerate systematically
    
    # The tensor T[i,j,k] = 1 if (i,j,k) is a valid triple
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all valid (i,j,k)
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # For rank r, we have r*3*n*m... variables. Let's be smart.
    # Try rank 12, 13, ..., up to some limit
    
    # Key insight: For 3x4x4, known lower bounds suggest rank might be around 12-15
    # Let's try to find exact decompositions with small integer coefficients
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    val = 0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if val != 1:
                        return False
        return True
    
    def try_rank(r, n, m, p, max_iterations=1000000):
        """Try to find a rank-r decomposition with coefficients in {-1, 0, 1}."""
        # This is a constraint satisfaction problem
        # For small r, we can try random sampling
        
        import random
        random.seed(42)
        
        num_u = n * m
        num_v = m * p
        num_w = p * n
        
        for _ in range(max_iterations):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                return None
            
            # Generate random U, V, W with entries in {-1, 0, 1}
            U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(num_u)]
            V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(num_v)]
            W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(num_w)]
            
            if verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try structured approaches first
    # Approach 1: Try ranks from 12 upward
    for r in range(12, min(rank_trivial, 20)):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        result = try_rank(r, n, m, p, max_iterations=50000)
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    # If no improvement found, return the trivial decomposition
    return best_result
# EVOLVE-BLOCK-END
