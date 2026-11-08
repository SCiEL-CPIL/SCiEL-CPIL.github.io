import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0  # Safety margin

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school

    random.seed(42)
    
    # Precompute index arrays
    # For U update: for each (i,j), W indices are [k*n+i for k in range(p)]
    U_indices = [(i, j) for i in range(n) for j in range(m)]
    V_indices = [(j, k) for j in range(m) for k in range(p)]
    W_indices = [(k, i) for k in range(p) for i in range(n)]
    
    # Vectorized verification
    def verify(U, V, W, tol=1e-10):
        r = U.shape[1]
        i_vals = np.repeat(np.arange(n), m*p)
        j_vals = np.tile(np.repeat(np.arange(m), p), n)
        k_vals = np.tile(np.arange(p), n*m)
        
        u_idx = i_vals * m + j_vals
        v_idx = j_vals * p + k_vals
        w_idx = k_vals * n + i_vals
        
        U_sel = U[u_idx, :]
        V_sel = V[v_idx, :]
        W_sel = W[w_idx, :]
        
        products = U_sel * V_sel * W_sel
        sums = products.sum(axis=1)
        
        return np.all(np.abs(sums - 1) < tol)
    
    # Solve for a given rank using alternating least squares
    def solve_rank(r, num_trials=5, max_iters=150, tol=1e-10):
        if time.time() > deadline:
            return None
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U
                for (i, j) in U_indices:
                    w_idx = np.array([k*n+i for k in range(p)])
                    M = V[j*p:(j+1)*p, :] * W[w_idx, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
                
                # Update V
                for (j, k) in V_indices:
                    u_idx = np.array([i*m+j for i in range(n)])
                    w_idx = np.array([k*n+i for i in range(n)])
                    M = U[u_idx, :] * W[w_idx, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
                
                # Update W
                for (k, i) in W_indices:
                    u_idx = np.array([i*m+j for j in range(m)])
                    v_idx = np.array([j*p+k for j in range(m)])
                    M = U[u_idx, :] * V[v_idx, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
                
                # Check convergence
                if it % 20 == 0 or it == max_iters - 1:
                    if verify(U, V, W, tol):
                        return U, V, W
        
        return None
    
    # Try ranks from low to high
    for r in range(10, best_rank):
        if time.time() > deadline - 5:
            break
        
        result = solve_rank(r, num_trials=3, max_iters=100)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_U = U
            best_V = V
            best_W = W
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
