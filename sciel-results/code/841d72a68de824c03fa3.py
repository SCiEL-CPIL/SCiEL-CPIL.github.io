import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _proxy_score(h):
    # h is a vector of non-negative numbers
    # Normalize to sum 1 to handle scale invariance
    s = np.sum(h)
    if s < 1e-9:
        return 0.0
    h = h / s
    
    # Discrete linear autoconvolution
    # Efficient way: use FFT or just np.convolve for small N
    c = np.convolve(h, h, mode='full')
    
    # L2 norm squared
    l2_sq = np.sum(h**2)
    
    # L1 norm
    l1 = np.sum(h)
    
    # Supremum norm of autoconvolution
    max_c = np.max(c)
    
    if max_c < 1e-9:
        return 0.0
        
    return (l2_sq / (l1 * max_c))

def construct():
    t0 = time.time()
    deadline = t0 + 140  # Safety margin
    
    # Initial guess: known best [2/3, 1/3, 0...]
    h0 = np.zeros(N)
    h0[0] = 2.0 / 3.0
    h0[1] = 1.0 / 3.0
    
    # Define bounds: h_i >= 0
    bounds = [(0.0, None)] * N
    
    # Objective: minimize negative proxy score
    def objective(h):
        return -_proxy_score(h)
    
    # Constraints: sum h = 1 (to fix scale)
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}]
    
    best_h = h0.copy()
    best_score = _proxy_score(best_h)
    
    # Strategy: Multi-start local optimization with diverse initializations
    # 1. The seed [2/3, 1/3]
    # 2. Random sparse supports (Sidon-like sets)
    # 3. Random dense supports
    # 4. Random smooth profiles (sine waves, etc.)
    
    rng = np.random.default_rng(42)
    
    initializations = []
    
    # 1. Seed
    initializations.append(h0.copy())
    
    # 2. Random sparse supports (pick K indices, random weights)
    for k in range(3, 11):
        if time.time() > deadline:
            break
        idx = rng.choice(N, k, replace=False)
        w = rng.random(k)
        w = w / np.sum(w)
        h_init = np.zeros(N)
        h_init[idx] = w
        initializations.append(h_init)
        
    # 3. Random dense supports
    for _ in range(5):
        if time.time() > deadline:
            break
        h_init = rng.random(N)
        h_init = h_init / np.sum(h_init)
        initializations.append(h_init)
        
    # 4. Random smooth profiles
    for _ in range(5):
        if time.time() > deadline:
            break
        x = np.linspace(0, 1, N)
        # Random sum of sines
        h_init = np.abs(np.sin(x * rng.random() * 10 + rng.random())) + 0.1
        h_init = h_init / np.sum(h_init)
        initializations.append(h_init)
        
    # 5. Try specific Sidon-like sets
    # Sidon set: all pairwise sums are distinct.
    # For small N, we can just pick random sets and check, or use known constructions.
    # A simple Sidon set is squares: 0, 1, 4, 9, 16, 25, 36, 49... but N=50.
    # Let's just use the random sparse ones as a proxy for now, or add a few known ones.
    # Squares: 0, 1, 4, 9, 16, 25, 36, 49 (8 elements)
    idx_sq = [0, 1, 4, 9, 16, 25, 36, 49]
    w_sq = np.ones(len(idx_sq))
    w_sq = w_sq / np.sum(w_sq)
    h_sq = np.zeros(N)
    h_sq[idx_sq] = w_sq
    initializations.append(h_sq)
    
    # Optimize each initialization
    for h_init in initializations:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                objective, 
                h_init, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints, 
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            if res.success:
                h_opt = res.x
                h_opt[h_opt < 0] = 0 # Clean up small negatives
                s = np.sum(h_opt)
                if s > 1e-9:
                    h_opt = h_opt / s
                    score_opt = _proxy_score(h_opt)
                    if score_opt > best_score:
                        best_h = h_opt
                        best_score = score_opt
        except Exception as e:
            pass
            
    # Final refinement with a higher iteration limit on the best found so far
    if time.time() < deadline - 10:
        try:
            res = minimize(
                objective, 
                best_h, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints, 
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            if res.success:
                h_opt = res.x
                h_opt[h_opt < 0] = 0
                s = np.sum(h_opt)
                if s > 1e-9:
                    h_opt = h_opt / s
                    score_opt = _proxy_score(h_opt)
                    if score_opt > best_score:
                        best_h = h_opt
                        best_score = score_opt
        except Exception as e:
            pass
            
    return [float(x) for x in best_h]
