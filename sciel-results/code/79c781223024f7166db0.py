import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.special import eval_chebyt
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    # 1. Precompute Chebyshev basis for generating diverse starting points
    x = np.linspace(-1, 1, N)
    K_basis = 15
    basis_cheb = np.zeros((N, K_basis + 1))
    for k in range(K_basis + 1):
        basis_cheb[:, k] = eval_chebyt(k, x)

    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return float('inf')
        if not np.any(np.isfinite(h)):
            return float('inf')
        # np.convolve is O(N^2), which is 160,000 ops, very fast.
        # But we want to be safe.
        z = np.convolve(h, h)
        return abs(2 * N * np.max(z) / (s * s))

    # 2. Generate a pool of diverse "extreme" candidates
    # Strategy: Use random Chebyshev combinations, then refine locally.
    # This creates a set of vectors that are likely in different "basins" of the landscape.
    
    pool = []
    rng = np.random.default_rng(42)
    
    # Add the seed
    pool.append(np.ones(N))
    
    num_pool_candidates = 20
    time_per_candidate = (deadline - time.time()) / (num_pool_candidates * 2)
    if time_per_candidate < 0.5:
        # If time is tight, reduce pool size
        num_pool_candidates = 5

    for i in range(num_pool_candidates):
        if time.time() > deadline - 10:
            break
        
        # Random Chebyshev coefficients
        # c0 is kept positive to ensure non-zero sum roughly
        c0 = rng.uniform(0.5, 2.0)
        c_rest = rng.uniform(-1.0, 1.0, size=K_basis)
        coeffs = np.concatenate(([c0], c_rest))
        
        h_init = coeffs @ basis_cheb.T
        
        # Local refinement using Nelder-Mead on the coefficients
        # We optimize the coefficients, not the 400-dim h directly,
        # to maintain the smooth structure.
        
        def obj_cheb(c):
            h = c @ basis_cheb.T
            if not np.all(np.isfinite(h)):
                return 1e10
            s = np.sum(h)
            if abs(s) < 1e-9:
                return 1e10
            z = np.convolve(h, h)
            return abs(2 * N * np.max(z) / (s * s))
        
        # Simple local optimization
        try:
            res = minimize(obj_cheb, coeffs, method='Nelder-Mead', 
                           options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.fun < compute_value(h_init):
                h_opt = res.x @ basis_cheb.T
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-9:
                    pool.append(h_opt)
        except Exception:
            pass

    # Ensure pool is not empty and valid
    valid_pool = [h for h in pool if np.all(np.isfinite(h)) and abs(np.sum(h)) > 1e-9]
    if not valid_pool:
        return np.ones(N)

    # Normalize pool vectors? 
    # The objective is scale-invariant: value(ah) = value(h).
    # However, convex combinations of unscaled vectors might result in cancellation.
    # It's better to work with the vectors as is, but ensure the combination doesn't sum to zero.
    
    M = len(valid_pool)
    
    # 3. Search for the best convex combination of these pool vectors
    # We want to find weights w >= 0, sum(w) = 1 such that h = sum(w_i * pool_i) minimizes value(h).
    # This is a non-convex optimization problem because value(h) is not convex.
    # But the dimension is M (e.g., 20), which is small.
    
    # We will use Differential Evolution on the simplex of weights.
    # DE requires bounds. We can use bounds [0, 1] for each weight and enforce sum=1 
    # via normalization inside the objective, or use a transformation.
    # Simpler: Optimize M-1 weights, the last is 1 - sum(others).
    # To handle the simplex constraint, we can just bound all weights to [0,1] 
    # and normalize in the objective function.
    
    def objective_weights(w):
        # w is a vector of length M
        if not np.all(np.isfinite(w)):
            return float('inf')
        # Normalize to simplex
        s_w = np.sum(w)
        if s_w < 1e-9:
            return float('inf')
        w_norm = w / s_w
        
        # Construct h
        h = np.zeros(N)
        for i in range(M):
            h += w_norm[i] * valid_pool[i]
            
        if not np.all(np.isfinite(h)):
            return float('inf')
            
        s_h = np.sum(h)
        if abs(s_h) < 1e-9:
            return float('inf')
            
        z = np.convolve(h, h)
        return abs(2 * N * np.max(z) / (s_h * s_h))

    # Initial best
    best_h = valid_pool[0]
    best_val = compute_value(best_h)
    
    # Check if any single pool element is better
    for h in valid_pool[1:]:
        v = compute_value(h)
        if v < best_val:
            best_val = v
            best_h = h

    # DE Optimization over weights
    if M > 1 and time.time() < deadline - 15:
        bounds = [(0.0, 1.0)] * M
        # popsize and maxiter scaled by dimension
        popsize = max(10, M)
        maxiter = 100
        
        try:
            res = differential_evolution(
                objective_weights, 
                bounds, 
                seed=42,
                maxiter=maxiter,
                popsize=popsize,
                tol=0.01,
                polish=True
            )
            if res.fun < best_val:
                # Recover the weights
                w_opt = res.x / np.sum(res.x)
                h_new = np.zeros(N)
                for i in range(M):
                    h_new += w_opt[i] * valid_pool[i]
                
                # Final verification
                if np.all(np.isfinite(h_new)) and abs(np.sum(h_new)) > 1e-9:
                    v_new = compute_value(h_new)
                    if v_new < best_val:
                        best_val = v_new
                        best_h = h_new
        except Exception:
            pass

    # If time remains, try a second round with a different seed or more pool elements?
    # Given the budget, one round of DE on the hull is a strong structural change.
    
    return best_h

if __name__ == "__main__":
    h = construct()
    print(len(h))
    print(np.sum(h))
    z = np.convolve(h, h)
    print(abs(2 * N * np.max(z) / np.sum(h)**2))
