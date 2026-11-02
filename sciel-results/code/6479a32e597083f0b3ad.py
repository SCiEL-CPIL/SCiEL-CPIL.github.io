import numpy as np
from itertools import product, combinations

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a rank-15 decomposition using exhaustive search
    # over support patterns with integer coefficients in {-2, -1, 0, 1, 2}
    # 
    # The tensor T has entries T[i,j,k] = 1 if the indices are consistent
    # (i.e., for matrix multiplication A[i,j] * B[j,k] = C[k,i] or similar)
    # 
    # We model each rank-1 term as choosing a support for U (subset of {0..nm-1}),
    # V (subset of {0..mp-1}), W (subset of {0..pn-1}), and a coefficient.
    # 
    # For efficiency, we restrict to "simple" patterns where each support has
    # size at most 2, and we search for exactly 15 such patterns.
    
    # First, let's try a known rank-15 construction if available,
    # otherwise do a bounded search.
    
    # The 18 constraints are for (i,j,k) in {0,1} x {0,1,2} x {0,1,2}
    # where the tensor entry is 1.
    
    # We'll use a randomized + local search approach to find rank 15.
    # If we can't find it quickly, return the rank-18 fallback.
    
    import random
    import time
    
    start_time = time.time()
    time_limit = 100  # seconds, leave buffer for evaluation overhead
    
    # Fallback: rank 18
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Target tensor: T[i,j,k] = 1 for all valid (i,j,k)
    # We need to represent T as sum of rank-1 tensors.
    # A rank-1 tensor is defined by U_vec, V_vec, W_vec where
    # T[i,j,k] += U_vec[i*m+j] * V_vec[j*p+k] * W_vec[k*n+i]
    
    # Let's try to find a rank-15 solution.
    # We'll parameterize each rank-1 term by its support and coefficient.
    
    # For a rank-1 term to contribute to T[i,j,k], we need:
    # U[i*m+j] * V[j*p+k] * W[k*n+i] = 1 for all (i,j,k) in the "block" it covers.
    
    # Actually, let me think differently. The standard approach for small matrix
    # multiplication is to use known decompositions.
    
    # For 2x3 times 3x3, the known minimal rank is 15 (Landsberg's bound or similar).
    # Let me try to construct it.
    
    # One approach: use the fact that 2x3*3x3 can be decomposed by splitting
    # the 2x3 matrix into two 1x3 rows, and using rank-8 decompositions for
    # 1x3*3x3 (which has rank 8? No, 1x3*3x3 = 9 entries, rank at least 9... 
    # wait, that's not right either).
    
    # Actually, let me just do a search. We'll represent each rank-1 term as
    # a triple of vectors with entries in {-2,-1,0,1,2} (or similar).
    
    # Given the complexity, let me try a simpler approach: 
    # Use a known rank-15 pattern from the literature if I can derive it.
    
    # Alternatively, I'll implement a simple local search:
    # 1. Start with 15 random rank-1 terms
    # 2. Adjust coefficients to minimize error
    # 3. If error is 0, we're done
    
    # Let's define the tensor as a vector of 18 entries (all 1s)
    tensor = [1.0] * (n * m * p)
    
    # Each rank-1 term is defined by (u_idx, v_idx, w_idx, coeff)
    # where u_idx is a subset of U indices, etc.
    # For simplicity, let's use full vectors with entries in {-1, 0, 1}
    
    # This is getting complex. Let me just return the fallback for now
    # and note that a proper search would be needed.
    
    # Actually, let me try one more thing: use a deterministic pattern
    # that I know works for small cases.
    
    # For 2x3 * 3x3, I'll try to construct rank 15 explicitly.
    # The idea: use 15 rank-1 terms where each term covers a "block" of
    # the tensor.
    
    # Let me define the 15 terms as follows:
    # Term t has U support S_U(t), V support S_V(t), W support S_W(t), coeff c(t)
    # The contribution to T[i,j,k] is sum over t of c(t) * [i*m+j in S_U(t)] * [j*p+k in S_V(t)] * [k*n+i in S_W(t)]
    
    # This is a system of 18 equations in 15*3*? variables.
    # Let me just try a random search with integer coefficients.
    
    # For speed, I'll try a limited number of random restarts.
    
    def random_search(max_iters=5000):
        for _ in range(max_iters):
            if time.time() - start_time > time_limit:
                return None
            
            # Generate 15 random rank-1 terms
            terms = []
            for _ in range(15):
                # Random support sizes
                su = random.randint(1, 2)
                sv = random.randint(1, 2)
                sw = random.randint(1, 2)
                
                u_supp = set(random.sample(range(n*m), su))
                v_supp = set(random.sample(range(m*p), sv))
                w_supp = set(random.sample(range(p*n), sw))
                
                # Coefficient
                coeff = random.choice([-2, -1, 1, 2])
                
                terms.append((u_supp, v_supp, w_supp, coeff))
            
            # Check if this combination satisfies all 18 constraints
            # Build the resulting tensor
            result = [0.0] * (n * m * p)
            for u_supp, v_supp, w_supp, coeff in terms:
                for a in u_supp:
                    i, j = divmod(a, m)
                    for b in v_supp:
                        j2, k = divmod(b, p)
                        if j == j2:
                            for c in w_supp:
                                k2, i2 = divmod(c, n)
                                if k == k2 and i == i2:
                                    idx = i * m * p + j * p + k
                                    result[idx] += coeff
            
            # Check if result matches tensor
            if all(abs(result[i] - tensor[i]) < 1e-9 for i in range(len(tensor))):
                # Found a solution!
                U = [[0] * 15 for _ in range(n * m)]
                V = [[0] * 15 for _ in range(m * p)]
                W = [[0] * 15 for _ in range(p * n)]
                for t, (u_supp, v_supp, w_supp, coeff) in enumerate(terms):
                    for a in u_supp:
                        U[a][t] = coeff
                    for b in v_supp:
                        V[b][t] = 1
                    for c in w_supp:
                        W[c][t] = 1
                return {"U": U, "V": V, "W": W}
        
        return None
    
    result = random_search(2000)
    if result is not None:
        return result
    
    return fallback
# EVOLVE-BLOCK-END
