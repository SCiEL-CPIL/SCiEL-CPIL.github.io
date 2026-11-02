import time
import numpy as np
import itertools

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s limit to be safe within 150s
    
    # Schoolbook decomposition (Rank 105)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    # Search for lower rank
    # Theoretical lower bound
    lower_bound = max(n * m, m * p, p * n) # 35
    
    # We try ranks from lower_bound to best_rank - 1
    # Start from higher ranks to find a solution faster if one exists near 105
    # Or start from lower? Usually, if a solution exists, it might be sparse.
    # Let's try from 104 downwards to 35.
    # If we find one, it's likely the minimum or close to it?
    # No, we want the MINIMUM. So we should search from 35 upwards.
    # But 35 is very low. The probability is negligible.
    # Let's search from 104 downwards to find *a* better solution first,
    # then continue to lower ranks if time permits?
    # Actually, the score is 1.0 for 105. Any rank < 105 is an improvement.
    # To maximize score, we want the lowest rank.
    # But finding rank 35 is impossible.
    # Finding rank 104 is more likely than 100?
    # Not necessarily.
    # Let's try a range of ranks.
    
    # Strategy:
    # 1. Try ranks 104, 103, 102, 101, 100.
    # 2. If found, return.
    # 3. If not, try 99...
    # 4. If time runs out, return best found (105).
    
    # To be efficient, we can try a few ranks in parallel or sequence.
    # Let's iterate R from 104 down to 35.
    # For each R, do T trials.
    # T can be adaptive.
    
    for R in range(best_rank - 1, lower_bound - 1, -1):
        if time.time() > deadline - 5:
            break
            
        # Number of trials
        # Higher R (closer to 105) might be easier?
        # Let's do 20 trials for each R to save time, as success is unlikely.
        trials = 20
        
        for t in range(trials):
            if time.time() > deadline - 5:
                break
                
            # Generate random U and V with entries in {-1, 0, 1}
            U_rand = np.random.randint(-1, 2, size=(n * m, R))
            V_rand = np.random.randint(-1, 2, size=(m * p, R))
            
            # Construct linear system A w = b
            # A is (n*m*p) x (p*n*R)
            # b is (n*m*p)
            b = np.ones(n * m * p)
            A = np.zeros((n * m * p, p * n * R))
            
            # Fill A
            # Loop over equations (i, j, k)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * (m * p) + j * p + k
                        # Term: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        # A[eq_idx, (k*n+i)*R + r] = U[i*m+j, r] * V[j*p+k, r]
                        for r in range(R):
                            coeff = U_rand[i * m + j, r] * V_rand[j * p + k, r]
                            if coeff != 0:
                                unk_idx = (k * n + i) * R + r
                                A[eq_idx, unk_idx] = coeff
            
            # Solve A w = b
            try:
                w_sol, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                
                # Round to nearest half-integer
                w_rounded = np.round(w_sol * 2) / 2
                
                # Verify if A @ w_rounded == b
                # Use allclose with tight tolerance
                if np.allclose(A @ w_rounded, b, atol=1e-8):
                    # Construct the solution matrices
                    U_list = U_rand.tolist()
                    V_list = V_rand.tolist()
                    W_list = [[0] * R for _ in range(p * n)]
                    for idx in range(len(w_sol)):
                        k_val = idx // (n * R)
                        rem = idx % (n * R)
                        i_val = rem // R
                        r_val = rem % R
                        W_list[k_val * n + i_val][r_val] = w_rounded[idx]
                    
                    return {"U": U_list, "V": V_list, "W": W_list}
                        
            except Exception:
                continue
                
    return best_solution
# EVOLVE-BLOCK-END
