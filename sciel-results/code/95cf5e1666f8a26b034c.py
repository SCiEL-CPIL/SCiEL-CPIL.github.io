import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    # Target tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k) else 0
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1

    # Randomized search for lower rank
    # We look for U, V, W with entries in {-1, -0.5, 0, 0.5, 1}
    # Rank R
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school
    
    start_time = time.time()
    deadline = start_time + 100  # 100s guard
    
    # Try ranks from 40 down to 35
    for R in range(40, 34, -1):
        if time.time() > deadline:
            break
            
        # Try a few random restarts
        for restart in range(20):
            if time.time() > deadline:
                break
                
            # Initialize with random small integers/halves
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, R))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, R))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, R))
            
            # Local search: flip random entries to minimize error
            # Error = || sum_r U[:,r] V[:,r] W[:,r] - T ||^2
            # This is expensive to compute fully.
            # We can compute the residual tensor Res = T - sum_r U_r V_r W_r
            # and then update one entry at a time.
            
            # Compute initial residual
            # Res[a,b,c] = T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r]
            # We can compute this using einsum
            # sum_r U[a,r] V[b,r] W[c,r] = np.einsum('ar,br,cr->abc', U, V, W)
            
            current_U = U.copy()
            current_V = V.copy()
            current_W = W.copy()
            
            # Compute current error
            # To speed up, we can just check a few random points or use a fast approximation.
            # But for exactness, we need to check all.
            # n*m=9, m*p=15, p*n=15. Size 9*15*15 = 2025.
            # R=40. 2025 * 40 = 81000 ops. Very fast.
            
            def calc_error(U, V, W):
                # Compute S = sum_r U[:,r] V[:,r] W[:,r]
                # S[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                S = np.einsum('ar,br,cr->abc', U, V, W)
                return np.sum((S - T)**2)
            
            err = calc_error(current_U, current_V, current_W)
            
            # Simple local search: iterate over random positions in U, V, W
            # and try flipping values
            for step in range(200):
                if time.time() > deadline:
                    break
                
                # Pick a random matrix and a random entry
                mat_idx = np.random.choice(3)
                if mat_idx == 0:
                    r, c = np.random.randint(0, n*m), np.random.randint(0, R)
                    old_val = current_U[r, c]
                    # Try new values
                    for new_val in [-1, -0.5, 0, 0.5, 1]:
                        if new_val == old_val: continue
                        current_U[r, c] = new_val
                        new_err = calc_error(current_U, current_V, current_W)
                        if new_err < err:
                            err = new_err
                        else:
                            current_U[r, c] = old_val
                            
                elif mat_idx == 1:
                    r, c = np.random.randint(0, m*p), np.random.randint(0, R)
                    old_val = current_V[r, c]
                    for new_val in [-1, -0.5, 0, 0.5, 1]:
                        if new_val == old_val: continue
                        current_V[r, c] = new_val
                        new_err = calc_error(current_U, current_V, current_W)
                        if new_err < err:
                            err = new_err
                        else:
                            current_V[r, c] = old_val
                            
                else:
                    r, c = np.random.randint(0, p*n), np.random.randint(0, R)
                    old_val = current_W[r, c]
                    for new_val in [-1, -0.5, 0, 0.5, 1]:
                        if new_val == old_val: continue
                        current_W[r, c] = new_val
                        new_err = calc_error(current_U, current_V, current_W)
                        if new_err < err:
                            err = new_err
                        else:
                            current_W[r, c] = old_val
                            
            if err < 1e-6:
                # Found a solution
                # Convert to lists
                best_result = {
                    "U": current_U.tolist(),
                    "V": current_V.tolist(),
                    "W": current_W.tolist()
                }
                best_rank = R
                # We can break early if we find a low rank
                if R <= 35:
                    break
        
        if best_rank <= R:
             break # Found a solution for this R or lower

    return best_result
