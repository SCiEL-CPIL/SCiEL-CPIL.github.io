import numpy as np
from itertools import product, combinations

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Fallback rank 18
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

    # Search for rank 15
    # Generate candidate boxes
    # Box is (I_set, J_set, K_set)
    # I_set is a subset of {0, 1}
    # J_set is a subset of {0, 1, 2}
    # K_set is a subset of {0, 1, 2}
    
    I_options = [frozenset([i]) for i in range(n)] + [frozenset(range(n))]
    J_options = [frozenset([j]) for j in range(m)] + [frozenset(range(m))] + [frozenset([j, k]) for j, k in combinations(range(m), 2)]
    K_options = [frozenset([k]) for k in range(p)] + [frozenset(range(p))] + [frozenset([k, l]) for k, l in combinations(range(p), 2)]
    
    # Filter to keep the list manageable
    # We'll include all 1x1x1, and some larger ones
    candidates = []
    for I in I_options:
        for J in J_options:
            for K in K_options:
                # Only include boxes that cover at least 1 point (always true)
                # To reduce search space, maybe only include boxes with size >= 2?
                # But 1x1x1 are needed for fine tuning.
                candidates.append((I, J, K))
                
    # We need to select 15 boxes from candidates.
    # This is too many. We need a smarter search.
    
    # Let's try a specific known structure or a limited random search.
    # Given the difficulty, I will try a few random subsets of 15 from a smaller pool.
    
    # Smaller pool: 
    # 1. All 1x1x1 (18 boxes)
    # 2. All 2x1x1, 1x2x1, 1x1x2 (where the 2 is the full dimension)
    # 3. All 2x3x1, 2x1x3, 1x3x3, etc.
    
    pool = []
    # 1x1x1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                pool.append((frozenset([i]), frozenset([j]), frozenset([k])))
                
    # 2x1x1 (full I)
    for j in range(m):
        for k in range(p):
            pool.append((frozenset(range(n)), frozenset([j]), frozenset([k])))
            
    # 1x2x1 (full J is not possible as J is size 3, so 2-subsets)
    for i in range(n):
        for J in combinations(range(m), 2):
            for k in range(p):
                pool.append((frozenset([i]), frozenset(J), frozenset([k])))
                
    # 1x1x2 (full K is not possible, so 2-subsets)
    for i in range(n):
        for j in range(m):
            for K in combinations(range(p), 2):
                pool.append((frozenset([i]), frozenset([j]), frozenset(K)))
                
    # 2x3x1 (full I, full J)
    for k in range(p):
        pool.append((frozenset(range(n)), frozenset(range(m)), frozenset([k])))
        
    # 2x1x3 (full I, full K)
    for j in range(m):
        pool.append((frozenset(range(n)), frozenset([j]), frozenset(range(p))))
        
    # 1x3x3 (full J, full K)
    for i in range(n):
        pool.append((frozenset([i]), frozenset(range(m)), frozenset(range(p))))
        
    # 2x3x3 (full I, full J, full K) - Rank 1, probably not useful alone but part of combo?
    pool.append((frozenset(range(n)), frozenset(range(m)), frozenset(range(p))))
    
    # Remove duplicates
    pool = list(set(pool))
    
    # Now search for 15 boxes from pool
    # We will try a few random samples
    import random
    random.seed(42)
    
    def check_solution(box_indices):
        R = len(box_indices)
        # Build the linear system A x = b
        # A has shape (n*m*p, R)
        # b is vector of 1s
        A = np.zeros((n * m * p, R))
        b = np.ones(n * m * p)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r, (I, J, K) in enumerate(box_indices):
                        if i in I and j in J and k in K:
                            A[idx, r] = 1
                    idx += 1
                    
        try:
            # Solve A x = b
            # Since R=15 and constraints=18, the system is overdetermined.
            # We want a least squares solution that is close to integers.
            x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            
            # Check if x is close to integers or half-integers
            # We can check if 2*x is close to integers
            x_half = 2 * x
            x_round = np.round(x_half)
            if np.allclose(x_half, x_round, atol=1e-9):
                # Valid solution
                # Construct U, V, W
                U = np.zeros((n * m, R))
                V = np.zeros((m * p, R))
                W = np.zeros((p * n, R))
                
                for r, (I, J, K) in enumerate(box_indices):
                    coeff = x[r]
                    for i in I:
                        for j in J:
                            U[i * m + j, r] = 1
                    for j in J:
                        for k in K:
                            V[j * p + k, r] = 1
                    for k in K:
                        for i in I:
                            # Wait, W index is k*n + i
                            W[k * n + i, r] = coeff
                            
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except:
            pass
        return None

    # Try random subsets
    for _ in range(1000):
        if len(pool) < 15:
            break
        indices = random.sample(range(len(pool)), 15)
        boxes = [pool[i] for i in indices]
        result = check_solution(boxes)
        if result:
            return result
            
    return fallback
# EVOLVE-BLOCK-END
