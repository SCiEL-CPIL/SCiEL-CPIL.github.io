"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

import time
import numpy as np

# EVOLVE-BLOCK-START
import random
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # safety margin before 150s
    
    # Schoolbook decomposition: rank = n*m*p
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try to find lower-rank decompositions
    # The tensor T[i,j,k] = delta_{i,k} (for C[i,k] = sum_j A[i,j] B[j,k])
    # We need: for all (i,j,k): sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 if i==k else 0
    
    # Strategy: exploit the structure. Since n=2, m=2, p=13,
    # the tensor has a special form. Let's try various rank reductions.
    
    # First, let's understand the tensor structure better.
    # T[i,j,k] = 1 if i==k, else 0
    # This is a "diagonal" tensor in the i-k plane.
    
    # For rank r, we need to find U (4 x r), V (26 x r), W (26 x r)
    # such that for all (i,j,k): sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] = T[i,j,k]
    
    # Let's try a structured approach:
    # Since T[i,j,k] = delta_{i,k}, we can try decompositions where
    # the W factor separates the i,k dependence.
    
    # Try rank = 2*p = 26 (use 2 ranks per k value)
    # Or try other structured ranks.
    
    # Let's use a more systematic search with random initialization
    # and gradient-free optimization (since we need exact integer/half-integer solutions)
    
    # Actually, for exact decompositions, let's try known constructions.
    # The key insight: T[i,j,k] = delta_{i,k} means the tensor is a sum of
    # "slices" T_i where T_i[j,k] = delta_{i,k}
    
    # For each i, the slice is: T_0[j,k] = 1 if k==0, T_1[j,k] = 1 if k==1
    # Each slice is rank 1 (as a 2x13 matrix), but they share the j dimension.
    
    # Let's try: rank = 2*p with U depending only on (i,j), V on (j,k), W on (k,i)
    # with a Kronecker-like structure.
    
    # Try: for each (k,i) pair where i==k, contribute rank 1 terms.
    # That gives rank = n = 2? No, that's too simple.
    
    # Let me think differently. The tensor is:
    # T = sum_i (e_i ⊗ I_m ⊗ e_i^T) where e_i are standard basis vectors
    # Actually: T[i,j,k] = delta_{i,k}
    # So T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} delta_{i,k} * e_i ⊗ e_j ⊗ e_k
    # = sum_i (e_i ⊗ (sum_j e_j) ⊗ e_i^T) ... no that's not right either.
    
    # T[i,j,k] = delta_{i,k} for all j. So for fixed (i,k), T[i,:,k] is either
    # all 1s (if i==k) or all 0s (if i!=k).
    
    # So T = sum_{i=0}^{n-1} (e_i ⊗ 1_m ⊗ e_i^T) where 1_m = (1,1,...,1)^T
    # Wait, that's not a tensor decomposition in the right form.
    
    # Let's just do a numerical search for small ranks.
    
    # Try ranks from a lower bound upward
    # Lower bound: The tensor has at least max(n,m,p) = 13 rank? No, that's not right.
    # The rank is at least the max of the slice ranks. Each (i,j) slice over k has rank 1.
    # Each (j,k) slice over i has rank 1. Each (i,k) slice over j has rank 1 (or 0).
    # So the tensor rank is at least... well, it could be as low as n*m = 4? 
    # But we need to verify.
    
    # Let's try to find a decomposition of rank r for r < 52.
    
    # Use a random search with exact arithmetic verification.
    # Initialize with random half-integer values and check if they work.
    
    # Actually, let's try a more principled approach.
    # Since T[i,j,k] = delta_{i,k}, we can write:
    # T = sum_{i=0}^{1} (e_i ⊗ J_m ⊗ e_i^T) where J_m is the all-ones row vector
    # But this isn't a valid tensor decomposition because J_m isn't a vector in the right space.
    
    # Let me just try random searches for lower ranks.
    
    def check_decomposition(U, V, W, r):
        """Check if U (4xr), V (26xr), W (26xr) give exact decomposition."""
        # For each (i,j,k), verify sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == delta_{i,k}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(r):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    expected = 1 if i == k else 0
                    if val != expected:
                        return False
        return True
    
    def try_random_rank(r, max_iter=5000):
        """Try to find a decomposition of rank r with random half-integer entries."""
        rng = random.Random(42 + r)
        
        for attempt in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Generate random half-integer matrices
            # Use small values to keep numbers manageable
            def rand_half_int():
                return rng.choice([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
            
            U = [[rand_half_int() for _ in range(r)] for _ in range(n*m)]
            V = [[rand_half_int() for _ in range(r)] for _ in range(m*p)]
            W = [[rand_half_int() for _ in range(r)] for _ in range(p*n)]
            
            if check_decomposition(U, V, W, r):
                return U, V, W
        
        return None
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Theoretical lower bound: at least n = 2 (since we need to distinguish i values)
    # But realistically, we need at least m*p/n or similar. Let's try from 4 upward.
    
    # Actually, let's be smarter. Try specific structured ranks first.
    
    # Try rank = 2*p = 26: one rank per (k, i) pair where i==k, doubled
    # Try rank = p + n = 15
    # Try rank = 2*m*p/n = 26
    
    # Let's just do a search from rank 4 to min(20, rank_schoolbook-1)
    for r in range(4, min(20, rank_schoolbook)):
        if time.time() > deadline:
            break
        
        result = try_random_rank(r, max_iter=2000)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = r
            break  # Found a valid lower-rank decomposition
    
    return best_result
# EVOLVE-BLOCK-END
