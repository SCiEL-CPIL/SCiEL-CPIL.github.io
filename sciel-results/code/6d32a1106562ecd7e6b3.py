import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
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
    
    start_time = time.time()
    deadline = start_time + 110 # 110s limit to be safe
    
    # Precompute valid and invalid indices
    valid_indices = []
    invalid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                valid_indices.append((alpha, beta, gamma))
                for jp in range(m):
                    if j != jp:
                        beta_inv = jp * p + k
                        invalid_indices.append((alpha, beta_inv, gamma))
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try ranks from 25 up to best_rank - 1
    for R in range(25, best_rank):
        if time.time() > deadline:
            break
            
        found = False
        attempts = 0
        max_attempts = 200 # Increase attempts significantly
        
        while not found and attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Initialize with random integers
            # Using small integers to keep numbers manageable
            U = np.random.randint(-2, 3, (n * m, R))
            V = np.random.randint(-2, 3, (m * p, R))
            W = np.random.randint(-2, 3, (p * n, R))
            
            # Efficient check using vectorized operations where possible
            # For each valid index, compute sum_r U[alpha, r] V[beta, r] W[gamma, r]
            # This is equivalent to (U[alpha] * V[beta]) @ W[gamma]
            
            valid = True
            # Check valid terms
            for alpha, beta, gamma in valid_indices:
                val = np.dot(U[alpha] * V[beta], W[gamma])
                if val != 1:
                    valid = False
                    break
            if not valid:
                continue
                
            # Check invalid terms
            for alpha, beta, gamma in invalid_indices:
                val = np.dot(U[alpha] * V[beta], W[gamma])
                if val != 0:
                    valid = False
                    break
            
            if valid:
                found = True
                best_rank = R
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break # Found a solution for this R, go to next R (lower is better)
        
        if found:
            # If we found a solution for R, we can try lower R?
            # The loop goes 25, 26... 
            # If we find 25, we break out of the R loop?
            # No, we want the MINIMAL rank.
            # So we should check 25, then 26...
            # If we find 25, that's the best possible (if lower bound is 25).
            # But lower bound is 43.
            # So we will likely not find 25.
            # We should continue to higher R?
            # No, we want to find ANY rank < 50.
            # If we find R=45, score is 50/45.
            # If we find R=44, score is 50/44.
            # So we should try to find the SMALLEST R that works.
            # My loop starts at 25. If it fails, it goes to 26.
            # If it finds 43, it updates best_rank to 43.
            # It should then stop? Or try 42?
            # The loop is `for R in range(25, best_rank)`.
            # If best_rank becomes 43, the range stops at 42.
            # So it will try 25..42.
            # This is correct.
            pass

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
