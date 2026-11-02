import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit

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

    # Theoretical lower bound
    lower_bound = max(n * m, m * p, p * n)  # 35

    # Helper to verify a solution
    def verify(U, V, W, R):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    # Strategy: Try to find a lower rank decomposition
    # We will search from best_rank - 1 down to lower_bound
    # For each rank R, we will try a few random initializations of U and V
    # and solve for W using least squares, then round and verify.
    
    # Optimization: Instead of building a massive A matrix for each trial,
    # we can exploit the structure.
    # The equations are: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    # This is a linear system in W given U and V.
    # The number of unknowns is p * n * R.
    # The number of equations is n * m * p.
    # For n=3, m=5, p=7, equations = 105.
    # Unknowns for R=100: 7 * 3 * 100 = 2100.
    # This is an underdetermined system. We want a solution with half-integer entries.
    
    # Let's iterate R from best_rank - 1 down to lower_bound.
    # We limit the number of trials per R to avoid timeout.
    
    R_search_start = best_rank - 1
    R_search_end = lower_bound
    
    # To be safe with time, we only search a few ranks near the top first.
    # If we don't find anything quickly, we return schoolbook.
    
    # Actually, finding a rank < 105 is extremely unlikely to succeed with random search
    # in this time frame unless there's a very specific structure.
    # However, we can try a few specific structured decompositions or just a limited random search.
    
    # Let's try a limited random search for R in [104, 100] just to be sure.
    # If it fails, return schoolbook.
    
    for R in range(R_search_start, max(R_search_end, R_search_start - 10), -1):
        if time.time() > deadline - 5:
            break
            
        # Try a few random trials for this R
        trials = 5
        for t in range(trials):
            if time.time() > deadline - 5:
                break
                
            # Random U and V with entries in {-1, 0, 1}
            # Using integers helps with the "half-integer" requirement for W potentially
            U_rand = np.random.randint(-1, 2, size=(n * m, R))
            V_rand = np.random.randint(-1, 2, size=(m * p, R))
            
            # Build linear system A @ w = b
            # w is flattened W of size p*n*R
            # b is ones of size n*m*p
            b = np.ones(n * m * p)
            # A is (n*m*p) x (p*n*R)
            # To save memory and time, we don't build A explicitly if possible, 
            # but for R~100, A is 105 x 2100, which is small.
            A = np.zeros((n * m * p, p * n * R))
            
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
            
            # Solve for w
            try:
                w_sol, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                
                # Round to nearest half-integer
                w_rounded = np.round(w_sol * 2) / 2
                
                # Verify
                # Check if A @ w_rounded == b
                # This is equivalent to checking the tensor identity
                # Let's do a direct check to be safe, or use A @ w_rounded
                if np.allclose(A @ w_rounded, b, atol=1e-8):
                    # Construct W
                    W_list = [[0] * R for _ in range(p * n)]
                    for idx in range(len(w_sol)):
                        k_val = idx // (n * R)
                        rem = idx % (n * R)
                        i_val = rem // R
                        r_val = rem % R
                        W_list[k_val * n + i_val][r_val] = w_rounded[idx]
                    
                    # Final verification
                    if verify(U_rand.tolist(), V_rand.tolist(), W_list, R):
                        return {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_list}
                        
            except Exception:
                continue
                
    return best_solution
# EVOLVE-BLOCK-END
