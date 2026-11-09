import numpy as np
import time

def construct():
    N = 24
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Precompute evaluation matrix for dense sampling
    M = 10000
    thetas = np.linspace(0, 2*np.pi, M, endpoint=False)
    # Z[i, k] = exp(i * theta_i * k)
    Z = np.exp(1j * thetas[:, None] * np.arange(N)[None, :])
    
    def fast_sup_norm(c):
        # c is numpy array of shape (N,)
        vals = Z @ c
        return np.max(np.abs(vals))
    
    def popcount(x):
        return bin(x).count('1')
    
    best_c = None
    best_score = np.inf
    
    # Generate Rudin-Shapiro sequence for initial guess
    rs = np.array([(-1) ** popcount(k) for k in range(N)])
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        if restart == 0:
            c = rs.astype(float)
        else:
            c = np.random.choice([-1.0, 1.0], N)
        
        current_score = fast_sup_norm(c)
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                old_val = c[i]
                c[i] = -old_val
                new_score = fast_sup_norm(c)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    c[i] = old_val
                    
        if current_score < best_score:
            best_score = current_score
            best_c = c.copy()
            
        # Early exit if we hit the theoretical lower bound or close to it?
        # Optimal is ~6.154.
        if best_score < 6.16:
            break

    if best_c is None:
        best_c = rs.astype(float)
        
    return list(best_c.astype(int))
