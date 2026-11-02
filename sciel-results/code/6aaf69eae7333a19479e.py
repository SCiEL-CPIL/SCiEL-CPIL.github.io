"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction
from itertools import combinations

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_schoolbook
    
    def verify(U, V, W, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(len(U_arr[0])):
                        total += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to reduce rank by finding linear dependencies
    # Strategy: try all subsets of terms to see if they can be replaced by fewer terms
    
    # For each pair of terms, check if a "Strassen-like" combination works
    # Term (i1,j1,k1) and (i2,j2,k2) can potentially be merged if there exist
    # coefficients a, b, c such that the combined decomposition still works
    
    # More systematic: try to find rank r < 42 by solving the system
    
    # Let's try a greedy approach: start with schoolbook, try to remove one term at a time
    # and see if we can adjust the remaining terms to compensate
    
    # Alternative: use the fact that for n=2, we can try to exploit the structure
    
    # Let's try to find a decomposition with rank < 42 by solving linear systems
    # We'll try ranks from 41 down to some lower bound
    
    # For a given rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # This is a nonlinear system, but we can try to solve it iteratively
    
    # Let's try a simpler approach: look for specific patterns
    
    # Pattern 1: If two terms have the same (i,j) but different k, they might be combinable
    # Pattern 2: If two terms have the same (j,k) but different i, they might be combinable
    # Pattern 3: If two terms have the same (k,i) but different j, they might be combinable
    
    # Let's try to find rank reductions by checking these patterns
    
    # For now, let's try a randomized search for rank reductions
    
    np.random.seed(42)
    
    # Try to find a decomposition with rank 41, 40, 39, etc.
    for target_rank in range(rank_schoolbook - 1, max(rank_schoolbook - 10, 30), -1):
        if time.time() > deadline - 30:
            break
            
        # Try to find a rank-target_rank decomposition
        # Start from schoolbook and try to merge terms
        
        # Method: try all pairs of terms and see if they can be replaced by 1 term
        U_curr = [row[:] for row in U_school]
        V_curr = [row[:] for row in V_school]
        W_curr = [row[:] for row in W_school]
        
        # Create a list of active terms
        active = list(range(rank_schoolbook))
        
        # Try to merge pairs
        merged_any = False
        for idx1 in range(len(active)):
            for idx2 in range(idx1 + 1, len(active)):
                t1 = active[idx1]
                t2 = active[idx2]
                
                # Check if terms t1 and t2 can be merged
                # This requires solving for new U, V, W values
                # For simplicity, try a few specific merge patterns
                
                # Pattern A: Same i, same j, different k -> can we combine?
                # Term t1: (i1, j1, k1), Term t2: (i2, j2, k2)
                i1, j1 = divmod(t1, m * p)
                k1 = t1 % p
                # Wait, the indexing is: col corresponds to (i, j, k) where col = i*m*p + j*p + k
                # Let me recompute
                # Actually from the schoolbook: col increments for each (i,j,k) triple
                # i in 0..n-1, j in 0..m-1, k in 0..p-1
                # col = i * m * p + j * p + k
                
                i1 = t1 // (m * p)
                rem1 = t1 % (m * p)
                j1 = rem1 // p
                k1 = rem1 % p
                
                i2 = t2 // (m * p)
                rem2 = t2 % (m * p)
                j2 = rem2 // p
                k2 = rem2 % p
                
                # Try to see if these two terms can be replaced by one term
                # This is complex, so let's skip for now
                
        # If we can't easily merge, try a different approach
        
        # Let's try a completely different strategy: use known results
        # For 2x3x7, the rank might be 39 or lower based on Strassen-like methods
        
        # Let's try to construct a rank-39 decomposition manually
        # by combining some terms using Strassen's idea
        
        if target_rank == 39:
            # Try to find a rank-39 decomposition
            # Strassen combines 8 terms into 7 for 2x2x2
            # For our case, we can try to apply similar ideas
            
            # Let's try to merge terms systematically
            # Group terms by (i, j) pairs and see if we can reduce the k dimension
            
            # For fixed (i, j), we have p=7 terms: (i, j, 0), (i, j, 1), ..., (i, j, 6)
            # Can we represent these 7 terms with fewer than 7 rank-1 terms?
            # In general, no, but maybe across different (i, j) we can share structure
            
            # Let's try a different approach: use linear algebra to find the rank
            # of the tensor under the given decomposition format
            
            # For now, return schoolbook if we can't find a better one
            pass
    
    # If no improvement found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
