"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # safety margin before 120s limit

    # Start with schoolbook
    rank_school = n * m * p  # 120
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    def verify(U, V, W):
        """Verify that U, V, W form a valid decomposition of matrix mult tensor."""
        nm = n * m
        mp = m * p
        pn = p * n
        r = len(U[0])
        # For each (i,j,k), check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Try to find a lower-rank decomposition using structured approaches
    # Strategy: For each j, the sub-tensor T_j[i,k] = 1 for all i,k
    # We can try to share rank terms across different j values
    
    # Approach: Use the fact that if we have a decomposition of the n×p identity-like tensor
    # we can reuse it for each j, but with different weightings
    
    # Let's try a more systematic search:
    # For rank r, we need to find U (nm×r), V (mp×r), W (pn×r) such that
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # One structured approach: assume U[i*m+j,c] = a[i,c]*b[j,c] (rank-1 in i,j for each c)
    # Similarly V[j*p+k,c] = b[j,c]*d[k,c] and W[k*n+i,c] = d[k,c]*a[i,c]
    # Then sum_c a[i,c]*b[j,c]*b[j,c]*d[k,c]*d[k,c]*a[i,c] = sum_c (a[i,c])^2*(b[j,c])^2*(d[k,c])^2
    # This won't generally work for all i,j,k to be 1.
    
    # Better approach: Try to find a decomposition where for each j, the contribution
    # from rank terms c is such that the sum over c gives 1 for all (i,k)
    
    # Let's try a different structured approach:
    # Suppose we use r = n*p + (m-1)*n*p = m*n*p (schoolbook) but try to reduce
    # by finding linear dependencies.
    
    # Actually, let's try a concrete search for small rank reductions:
    # We'll try to find if rank 119, 118, etc. is possible by solving a linear system
    
    random.seed(42)
    
    # Try to find a decomposition of rank r < 120
    # For each candidate rank, try to solve the system
    for trial_rank in range(rank_school - 1, max(n*m, m*p, p*n) - 1, -1):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Initialize with a perturbed schoolbook decomposition
            # Start with schoolbook and try to zero out some entries
            U = [row[:] for row in U_school]
            V = [row[:] for row in V_school]
            W = [row[:] for row in W_school]
            
            # Try to reduce rank by setting some columns to zero and adjusting others
            # This is a heuristic approach
            
            # Alternative: try a completely different initialization
            # Use random initialization and try to solve
            U_rand = [[random.choice([-1, 0, 1]) for _ in range(trial_rank)] for _ in range(n * m)]
            V_rand = [[random.choice([-1, 0, 1]) for _ in range(trial_rank)] for _ in range(m * p)]
            W_rand = [[random.choice([-1, 0, 1]) for _ in range(trial_rank)] for _ in range(p * n)]
            
            if verify(U_rand, V_rand, W_rand):
                best_U, best_V, best_W = U_rand, V_rand, W_rand
                best_rank = trial_rank
                found = True
                break
        
        if found:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
