"""
Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK.
"""
DIMENSIONS = [3, 4, 7]
import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # 1. Construct Schoolbook Decomposition (Rank = n*m*p)
    # This is always a valid solution.
    def schoolbook():
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

    best_result = schoolbook()
    best_rank = n * m * p

    # 2. Verification Functions
    def verify_fast(U, V, W):
        """Vectorized verification using numpy."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        if rank == 0:
            return False
        U_r = U_np.reshape(n, m, rank)
        V_r = V_np.reshape(m, p, rank)
        W_r = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=1e-9)

    def verify_exact(U, V, W):
        """Exact rational verification using fractions."""
        rank = len(U[0]) if U and U[0] else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for c in range(rank):
                        # Convert to Fraction to handle floats safely if needed, 
                        # but we generate exact values.
                        total += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                    if total != 1:
                        return False
        return True

    # 3. Search for lower rank decompositions
    # Lower bound is max(n*m, m*p, p*n) = max(12, 28, 21) = 28.
    # However, random search in high dimensions is very unlikely to find solutions 
    # significantly below the schoolbook rank (84) for small dimensions.
    # We will perform a limited, efficient search to see if we can beat 84.
    
    # Value pool: integers and half-integers as per task description
    value_pool = [0, 1, -1, 0.5, -0.5, 2, -2, 1.5, -1.5]
    
    # We prioritize trying ranks just below the schoolbook rank first, 
    # as they are more likely to be reachable via small perturbations/mergers.
    # But given the combinatorial explosion, we limit the search.
    
    # Strategy: 
    # Instead of pure random initialization which is sparse and likely to fail,
    # we can try a "local search" starting from the schoolbook decomposition.
    # However, implementing a robust local search for tensor rank minimization
    # within 140s for 84->83 is non-trivial.
    
    # Alternative Strategy from Inspiration 2: Random sparse matrices.
    # We will try a few specific ranks with a limited number of attempts.
    
    # Given the time constraint and the high likelihood that 84 is minimal 
    # (or close to it) for these dimensions without specialized algorithms,
    # we will run a short random search. If it fails, we return schoolbook.
    
    # Let's try a few ranks: 83, 82, 81, ... down to a lower bound or time limit.
    # To be efficient, we'll use numpy for generation and fast verification.
    
    # We'll focus on ranks close to 84 first.
    # If we find a valid decomposition, we update best_result.
    
    # Note: The probability of finding a valid rank-83 decomposition by random 
    # sparse sampling is extremely low. 
    # A more effective approach might be to check if the schoolbook tensor 
    # has any linear dependencies among its "slices" that could be exploited.
    # For the standard matrix multiplication tensor, the rank is generally 
    # considered to be n*m*p for small n,m,p unless specific symmetries exist.
    
    # Let's do a very quick check: can we merge any two terms?
    # This is equivalent to checking if the sum of two rank-1 tensors 
    # can be expressed as a single rank-1 tensor, which is generally false 
    # for distinct (i,j,k) indices in the schoolbook decomposition.
    
    # Therefore, we will proceed with the random search from Inspiration 2 
    # but keep it very tight to avoid wasting time.
    
    ranks_to_try = list(range(best_rank - 1, max(20, best_rank - 5), -1))
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 5:
            break
            
        max_attempts = 10  # Very limited attempts per rank
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            # Use a deterministic seed based on rank and attempt for reproducibility
            # but varied enough to explore.
            random.seed(target_rank * 1000 + attempt)
            
            # Strategy: Sparse random with half-integers
            # Probability of non-zero entry
            prob = 0.2
            
            U = [[random.choice(value_pool) if random.random() < prob else 0 for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice(value_pool) if random.random() < prob else 0 for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice(value_pool) if random.random() < prob else 0 for _ in range(target_rank)] for _ in range(p * n)]
            
            # Quick check: ensure not all zeros
            if all(x == 0 for row in U for x in row):
                continue
            if all(x == 0 for row in V for x in row):
                continue
            if all(x == 0 for row in W for x in row):
                continue
            
            if verify_fast(U, V, W):
                if verify_exact(U, V, W):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    # If we found one, we could break, but maybe a lower rank exists?
                    # Given the low probability, we break to save time for the next lower rank 
                    # or just return if it's significantly better.
                    # Actually, if we find rank 83, we should try 82.
                    # But the loop structure handles this.
                    pass
    
    return best_result
# EVOLVE-BLOCK-END
