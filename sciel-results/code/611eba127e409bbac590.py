import time
import numpy as np
import random
import math

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    
    # Target tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait, the tensor is defined by the multiplication.
    # C_ik = sum_j A_ij B_jk
    # The term A_ij B_jk appears with coefficient 1.
    # So the tensor component corresponding to indices (i, j, k) is 1.
    
    # We want U, V, W such that sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1
    # for all i, j, k.
    
    # We search for rank r < 24.
    # Use scaled integers to allow half-integers.
    # Let U, V, W be integers. Then sum U*V*W = 8 (if we scale target by 8? No.)
    # If we want half-integers, let U, V, W be integers and target is 8?
    # No, if U, V, W are integers, the sum is integer.
    # If we want half-integers, we can multiply all by 2.
    # Let u = 2*U, v = 2*V, w = 2*W.
    # sum (u/2)(v/2)(w/2) = sum u*v*w / 8 = 1 => sum u*v*w = 8.
    
    def check_solution(U, V, W):
        # U, V, W are numpy arrays of shape (dim, rank)
        # Check if sum_t U[i,j,t] V[j,k,t] W[k,i,t] == 8 for all i,j,k
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(U.shape[1]):
                        s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    if s != 8:
                        return False
        return True

    def error(U, V, W):
        # Calculate sum of squared errors
        # U, V, W are flat arrays or matrices
        # Reshape to matrices
        r = U.shape[1]
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(r):
                        s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    err += (s - 8.0) ** 2
        return err

    def random_init(r):
        # Random integers in a small range
        U = np.random.randint(-5, 6, size=(n*m, r))
        V = np.random.randint(-5, 6, size=(m*p, r))
        W = np.random.randint(-5, 6, size=(p*n, r))
        return U, V, W

    def local_search(U, V, W, max_iter=1000):
        # Simple random walk with acceptance if error decreases
        # Or hill climbing
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = error(U, V, W)
        r = U.shape[1]
        
        for _ in range(max_iter):
            # Pick a random element in one of the matrices
            mat_idx = random.choice([0, 1, 2])
            if mat_idx == 0:
                row = random.randint(0, n*m-1)
                col = random.randint(0, r-1)
                delta = random.choice([-1, 1])
                U[row, col] += delta
                new_err = error(U, V, W)
                if new_err < best_err:
                    best_err = new_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    U[row, col] -= delta
            elif mat_idx == 1:
                row = random.randint(0, m*p-1)
                col = random.randint(0, r-1)
                delta = random.choice([-1, 1])
                V[row, col] += delta
                new_err = error(U, V, W)
                if new_err < best_err:
                    best_err = new_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    V[row, col] -= delta
            else:
                row = random.randint(0, p*n-1)
                col = random.randint(0, r-1)
                delta = random.choice([-1, 1])
                W[row, col] += delta
                new_err = error(U, V, W)
                if new_err < best_err:
                    best_err = new_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    W[row, col] -= delta
                    
            if best_err == 0:
                break
                
        return best_U, best_V, best_W, best_err

    # Try ranks from 20 to 23
    # Time limit is 120s. Let's allocate time carefully.
    start_time = time.time()
    time_limit = 110.0 # Leave margin
    
    best_solution = None
    best_rank = rank_school
    
    for r in range(20, 24):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few restarts for this rank
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            U, V, W = random_init(r)
            U, V, W, err = local_search(U, V, W, max_iter=200)
            
            if err == 0:
                # Found a solution!
                # Convert back to half-integers (divide by 2)
                U_out = [[U[i, t] / 2.0 for t in range(r)] for i in range(n*m)]
                V_out = [[V[i, t] / 2.0 for t in range(r)] for i in range(m*p)]
                W_out = [[W[i, t] / 2.0 for t in range(r)] for i in range(p*n)]
                
                # Verify
                if check_solution(U, V, W):
                    return {"U": U_out, "V": V_out, "W": W_out}
                
    # If no better solution found, return schoolbook
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
