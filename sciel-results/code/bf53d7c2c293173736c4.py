import time
import numpy as np

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 130
    
    # Build target tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition (rank 36)
    rank_fb = n * m * p
    U_fb = np.zeros((n*m, rank_fb), dtype=int)
    V_fb = np.zeros((m*p, rank_fb), dtype=int)
    W_fb = np.zeros((p*n, rank_fb), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    best_result = (U_fb, V_fb, W_fb, rank_fb)
    best_rank = rank_fb
    
    # Try to find lower rank decompositions using random search
    # The key insight: for 3x3x4, we can potentially do better than 36
    
    # First, let's try a structured approach:
    # For each (i,j) pair in the 3x3 input, we need to compute 4 outputs
    # The standard approach uses 3*3*4 = 36 terms
    # But we can potentially share computations
    
    # Let's try random search with small integer entries
    for rank in range(20, 36):
        if time.time() > deadline - 15:
            break
        if rank >= best_rank:
            continue
        
        # Number of attempts based on remaining time and rank
        remaining = deadline - 15 - time.time()
        if remaining < 5:
            break
        
        # More attempts for lower ranks
        base_attempts = 10000 if rank <= 25 else 5000 if rank <= 30 else 2000
        attempts = int(base_attempts * remaining / 100)
        attempts = max(100, min(attempts, 50000))
        
        for attempt in range(attempts):
            if time.time() > deadline - 10:
                break
            
            # Generate random matrices with entries in {-1, 0, 1}
            # Use sparse random matrices for efficiency
            U = np.random.choice([-1, 0, 1], size=(n*m, rank), p=[0.3, 0.4, 0.3])
            V = np.random.choice([-1, 0, 1], size=(m*p, rank), p=[0.3, 0.4, 0.3])
            W = np.random.choice([-1, 0, 1], size=(p*n, rank), p=[0.3, 0.4, 0.3])
            
            # Compute decomposed tensor efficiently
            # T_decomp[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
            T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
            
            if np.allclose(T_decomp, T, atol=1e-10):
                best_result = (U, V, W, rank)
                best_rank = rank
                break
        else:
            continue
    
    # Try with half-integer entries for some ranks
    for rank in range(24, 32):
        if time.time() > deadline - 10:
            break
        if rank >= best_rank:
            continue
        
        remaining = deadline - 10 - time.time()
        if remaining < 5:
            break
        
        attempts = int(2000 * remaining / 100)
        attempts = max(100, min(attempts, 20000))
        
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
            
            vals = [-1, -0.5, 0, 0.5, 1]
            U = np.random.choice(vals, size=(n*m, rank))
            V = np.random.choice(vals, size=(m*p, rank))
            W = np.random.choice(vals, size=(p*n, rank))
            
            T_decomp = np.einsum('at,bt,ct->abc', U, V, W, optimize=True)
            
            if np.allclose(T_decomp, T, atol=1e-10):
                best_result = (U, V, W, rank)
                best_rank = rank
                break
    
    U_best, V_best, W_best, _ = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
