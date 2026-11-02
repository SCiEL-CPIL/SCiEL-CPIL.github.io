import time
import random
import numpy as np

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    # Try ranks from 8 upwards
    for rank in range(8, n*m*p + 1):
        if time.time() > deadline:
            break
            
        # Try to find a solution for this rank
        # We use a random search with local optimization
        
        # Max iterations per rank
        max_iters = 50000
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random initialization
            # Use small integers
            U = np.random.randint(-1, 2, (n*m, rank))
            V = np.random.randint(-1, 2, (m*p, rank))
            W = np.random.randint(-1, 2, (p*n, rank))
            
            # Local search
            # Calculate error
            def calc_error(U, V, W):
                # T_ijk = sum_c U_ij_c V_jk_c W_ki_c
                # We want T_ijk = 1
                # i in n, j in m, k in p
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err += (s - 1)**2
                return err
            
            e = calc_error(U, V, W)
            if e == 0:
                # Found solution
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Simple hill climbing
            # Pick random variable
            # This is slow in Python. Use numpy for vectorization?
            # The error calculation is O(nmp * rank).
            # 2*2*4 * 8 = 128 ops. Very fast.
            
            # Actually, let's just do a few restarts with better initialization
            # Or use a known structure.
            
            # Let's try a more structured random search
            # Since rank 8 is tight, random -1/1 might miss.
            # Try -1, 0, 1
            
            # Let's just loop and hope.
            # To speed up, we can check if the current state is a local minimum
            # and then perturb.
            
            # For simplicity and robustness, I'll implement a basic random restart search.
            # If it doesn't find it quickly, it will return the best found or fall back.
            
            # Actually, I should implement a proper local search.
            # But 50k iterations of Python loops might be slow.
            # 50k * 128 ops = 6.4M ops. Should be < 1s.
            
            # Let's refine the local search step
            best_e = e
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for step in range(100): # 100 steps of local search
                if time.time() > deadline:
                    break
                
                # Randomly pick a matrix and an element
                mat_idx = np.random.randint(0, 3)
                if mat_idx == 0:
                    r_idx = np.random.randint(0, n*m)
                    c_idx = np.random.randint(0, rank)
                    old_val = U[r_idx, c_idx]
                    U[r_idx, c_idx] = old_val + np.random.choice([-1, 1])
                    new_e = calc_error(U, V, W)
                    if new_e < best_e:
                        best_e = new_e
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    else:
                        U[r_idx, c_idx] = old_val # revert
                elif mat_idx == 1:
                    r_idx = np.random.randint(0, m*p)
                    c_idx = np.random.randint(0, rank)
                    old_val = V[r_idx, c_idx]
                    V[r_idx, c_idx] = old_val + np.random.choice([-1, 1])
                    new_e = calc_error(U, V, W)
                    if new_e < best_e:
                        best_e = new_e
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    else:
                        V[r_idx, c_idx] = old_val
                else:
                    r_idx = np.random.randint(0, p*n)
                    c_idx = np.random.randint(0, rank)
                    old_val = W[r_idx, c_idx]
                    W[r_idx, c_idx] = old_val + np.random.choice([-1, 1])
                    new_e = calc_error(U, V, W)
                    if new_e < best_e:
                        best_e = new_e
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    else:
                        W[r_idx, c_idx] = old_val
                
                if best_e == 0:
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
                    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
