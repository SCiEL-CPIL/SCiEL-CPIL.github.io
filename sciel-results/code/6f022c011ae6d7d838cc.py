import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook solution
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_schoolbook
    best_solution = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds search time
    
    # Try to find a lower rank solution
    # We search for ranks from rank_schoolbook - 1 down to 15
    for rank in range(rank_schoolbook - 1, 14, -1):
        if time.time() > deadline:
            break
            
        # Try random solutions for this rank
        # We limit the number of trials per rank to avoid getting stuck
        max_trials = 10000
        for trial in range(max_trials):
            if time.time() > deadline:
                break
                
            # Generate random U, V, W with entries in {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(n * m, rank))
            V = np.random.choice([-1, 0, 1], size=(m * p, rank))
            W = np.random.choice([-1, 0, 1], size=(p * n, rank))
            
            # Check if the decomposition is valid
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        # Compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                        # We can vectorize this
                        # u = U[i*m+j, :]
                        # v = V[j*p+k, :]
                        # w = W[k*n+i, :]
                        # s = np.dot(u * v, w)
                        u = U[i * m + j, :]
                        v = V[j * p + k, :]
                        w = W[k * n + i, :]
                        s = np.dot(u * v, w)
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                    
            if valid:
                best_rank = rank
                best_solution = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break # Found a solution for this rank, try lower rank? 
                # Actually, we want the minimal rank, so we should continue to lower ranks.
                # But if we found rank R, we can try R-1.
                # So we don't break the outer loop, we just update best and continue.
                # But the outer loop is descending, so it will try lower ranks next.
                
    return best_solution
