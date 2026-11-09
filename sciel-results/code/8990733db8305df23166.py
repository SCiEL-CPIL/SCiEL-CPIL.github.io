import numpy as np
import time
import sys
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    # Start with uniform distribution
    P0 = np.ones((n, n)) / (n * n)
    
    # Time limit
    start_time = time.time()
    deadline = start_time + 580  # 20s margin
    
    best_P = P0
    best_score = -1.0
    
    def get_projected_entropies(P):
        # P is 10x10
        # Slopes: -1, 0, 1, 2
        # Values: x + r*y
        # x is col index, y is row index
        xs = np.arange(n)
        ys = np.arange(n)
        
        entropies = {}
        for r in [-1, 0, 1, 2]:
            # Create grid of values
            # X is 0..9, Y is 0..9
            # Value = X + r*Y
            X_grid, Y_grid = np.meshgrid(xs, ys)
            vals = X_grid + r * Y_grid
            
            # Sum probabilities for each unique value
            # Unique values range from min to max
            min_val = np.min(vals)
            max_val = np.max(vals)
            
            # Initialize counts
            counts = np.zeros(int(max_val - min_val + 1))
            
            # Vectorized sum
            # vals are integers
            indices = vals - min_val
            np.add.at(counts, indices.ravel(), P.ravel())
            
            # Filter non-zero
            probs = counts[counts > 1e-12]
            if len(probs) == 0:
                entropies[r] = 0.0
            else:
                # Normalize to ensure sum 1 (floating point)
                probs = probs / probs.sum()
                entropies[r] = -np.sum(probs * np.log2(probs))
        
        return entropies

    def calculate_score(P):
        ents = get_projected_entropies(P)
        D = ents[-1]
        M = max(ents[0], ents[1], ents[2])
        if M < 1e-9:
            return -1.0
        return D / M

    # Initial score
    best_score = calculate_score(P0)
    
    # Optimization
    # Parameterize P via logit Z
    # P = exp(Z) / sum(exp(Z))
    
    def softmax(Z):
        Z = Z.reshape(n, n)
        Z_max = np.max(Z)
        exp_Z = np.exp(Z - Z_max)
        P = exp_Z / np.sum(exp_Z)
        return P

    def objective(Z_flat):
        P = softmax(Z_flat)
        ents = get_projected_entropies(P)
        D = ents[-1]
        H0 = ents[0]
        H1 = ents[1]
        H2 = ents[2]
        
        # Smooth max
        # Use log-sum-exp for max
        # max(a,b,c) approx tau * log(exp(a/tau) + exp(b/tau) + exp(c/tau))
        tau = 0.5 # Smoothness parameter
        max_val = np.max([H0, H1, H2])
        if max_val == 0:
            M_smooth = 0
        else:
            # To avoid overflow, subtract max
            a, b, c = H0/tau, H1/tau, H2/tau
            m = np.max([a, b, c])
            M_smooth = tau * (m + np.log(np.exp(a-m) + np.exp(b-m) + np.exp(c-m)))
            
        if M_smooth < 1e-9:
            return 0.0 # Avoid division by zero, return 0 (bad score)
            
        return -D / M_smooth

    # Run multiple random restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        Z_init = np.random.normal(0, 1, (n, n))
        
        # Use L-BFGS-B
        try:
            res = minimize(objective, Z_init.flatten(), method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-8})
            
            P_opt = softmax(res.x)
            score_opt = calculate_score(P_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_P = P_opt
                
        except Exception:
            continue
            
    # Final Polish: Use the best P found, maybe refine with smaller tau or more iterations?
    # The L-BFGS-B with tau=0.5 might be stuck in local optima.
    # Let's try a few more iterations with the best P as start, reducing tau.
    
    if time.time() < deadline - 10:
        Z_best = np.log(best_P + 1e-12) # Recover Z from P (approx)
        # Actually, better to keep Z from the best run. 
        # But we didn't store Z. We can reconstruct or just re-optimize from P_best.
        
        # Let's do a quick gradient step with smaller tau on the best P
        # Or just trust the L-BFGS-B result.
        
    # Ensure valid output
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s <= 1e-9:
        return P0
    best_P /= s
    
    return best_P
