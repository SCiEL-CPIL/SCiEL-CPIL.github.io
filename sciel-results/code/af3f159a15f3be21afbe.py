import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 4, 4
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Target Tensor
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T is 1 if (a,b,c) corresponds to a valid multiplication step, 0 otherwise.
    # Actually, we want sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
    
    # Build the target tensor T (sparse)
    # We only care about the 48 non-zero entries and ensuring others are 0.
    # For speed, we will compute the error only on a subset or the whole tensor if fast enough.
    # Tensor size: 12 * 16 * 12 = 2304 entries. This is small.
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def error_func(params, rank):
        # params is a flat array of size rank * (n*m + m*p + p*n)
        # But we can optimize U, V, W separately or jointly.
        # Let's flatten U, V, W into one vector.
        
        dim_u = n * m
        dim_v = m * p
        dim_w = p * n
        
        U = params[:rank * dim_u].reshape(rank, dim_u).T # (dim_u, rank)
        V = params[rank * dim_u : rank * dim_u + rank * dim_v].reshape(rank, dim_v).T
        W = params[rank * dim_u + rank * dim_v :].reshape(rank, dim_w).T
        
        # Compute T_reconstructed
        # T_reconstructed[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # We can compute this efficiently using einsum
        # T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        
        # To save memory and time, we might not want to build the full tensor if rank is high,
        # but 46 is small.
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Error: sum (T_rec - T_target)^2
        # We can restrict to non-zero entries of T_target for the '1' part,
        # and all entries for the '0' part?
        # No, we need T_rec to be 0 where T_target is 0.
        # So error = sum((T_rec - T_target)**2)
        
        err = np.sum((T_rec - T_target)**2)
        return err

    def try_rank(rank):
        if time.time() > deadline - 20:
            return None
            
        dim_u = n * m
        dim_v = m * p
        dim_w = p * n
        total_params = rank * (dim_u + dim_v + dim_w)
        
        # Multiple restarts
        best_err = np.inf
        best_params = None
        
        num_restarts = 10
        for i in range(num_restarts):
            if time.time() > deadline - 20:
                break
                
            # Initialization
            # Strategy: Start with schoolbook-like structure if possible, or random
            # For rank 46, random is risky.
            # Let's try a "greedy" init: take first 46 columns of schoolbook, 
            # then randomize the last 2 slightly? No, that won't work for rank reduction.
            
            # Better init: Random normal
            np.random.seed(i)
            x0 = np.random.normal(0, 1, total_params)
            
            # Use Nelder-Mead or Powell
            # Powell is often better for this kind of problem
            try:
                res = minimize(error_func, x0, args=(rank,), method='Powell', 
                               options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-8})
                if res.fun < best_err:
                    best_err = res.fun
                    best_params = res.x
            except Exception:
                pass
                
        if best_params is not None and best_err < 1e-6:
            # Reconstruct U, V, W
            U = best_params[:rank * dim_u].reshape(rank, dim_u).T
            V = best_params[rank * dim_u : rank * dim_u + rank * dim_v].reshape(rank, dim_v).T
            W = best_params[rank * dim_u + rank * dim_v :].reshape(rank, dim_w).T
            return U, V, W, best_err
        return None

    # Try rank 46
    result = try_rank(46)
    if result:
        U, V, W, err = result
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
        
    # Try rank 47
    result = try_rank(47)
    if result:
        U, V, W, err = result
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
        
    # Fallback to schoolbook rank 48
    rank = 48
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
