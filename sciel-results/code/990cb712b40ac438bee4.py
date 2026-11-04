"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]
import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    best_rank = n * m * p  # 36
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize schoolbook
    U = [[0] * best_rank for _ in range(n * m)]
    V = [[0] * best_rank for _ in range(m * p)]
    W = [[0] * best_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    best_U, best_V, best_W = U, V, W
    
    def verify(U_mat, V_mat, W_mat, rank):
        """Verify that the decomposition is exact using exact arithmetic.
        
        The condition is: for all i in [0,n), k in [0,p), j in [0,m), j2 in [0,m):
        sum_c W[k*n+i,c] * U[i*m+j,c] * V[j2*p+k,c] = 1 if j==j2 else 0
        """
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for j2 in range(m):
                        total = Fraction(0)
                        for c in range(rank):
                            total += Fraction(W_mat[k*n+i][c]) * Fraction(U_mat[i*m+j][c]) * Fraction(V_mat[j2*p+k][c])
                        expected = 1 if j == j2 else 0
                        if total != expected:
                            return False
        return True
    
    # Try to find lower-rank decompositions using random search
    # Start from rank 35 down to some minimum
    min_rank_to_try = 20
    
    for target_rank in range(best_rank - 1, max(min_rank_to_try, 0) - 1, -1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations for this rank
        max_attempts = 50
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 30:
                break
            
            # Initialize with random small integers/half-integers
            U_try = [[random.choice([0, 0.5, 1, 1.5, 2, -0.5, -1]) for _ in range(target_rank)] for _ in range(n*m)]
            V_try = [[random.choice([0, 0.5, 1, 1.5, 2, -0.5, -1]) for _ in range(target_rank)] for _ in range(m*p)]
            W_try = [[random.choice([0, 0.5, 1, 1.5, 2, -0.5, -1]) for _ in range(target_rank)] for _ in range(p*n)]
            
            # Verify
            if verify(U_try, V_try, W_try, target_rank):
                best_rank = target_rank
                best_U = U_try
                best_V = V_try
                best_W = W_try
                found = True
                break
        
        if found:
            # Found a lower rank, continue searching for even lower
            continue
        else:
            # Didn't find for this rank, try lower rank (but this is expensive)
            # For now, break to save time
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
