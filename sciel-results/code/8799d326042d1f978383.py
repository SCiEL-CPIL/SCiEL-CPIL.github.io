import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 14
D = 3

def _pairwise_sq_dists(P):
    """Compute squared pairwise distances for N points in D dimensions."""
    # P is (N, D)
    # dist2[i, j] = ||P[i] - P[j]||^2
    # Using broadcasting
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist2 = np.sum(diff ** 2, axis=-1)
    return dist2

def _soft_min_max_ratio(P, tau):
    """
    Compute the soft ratio of max to min squared distance.
    Returns (value, gradient)
    """
    dist2 = _pairwise_sq_dists(P)
    # Mask upper triangle to avoid self-distances and double counting
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    vals = dist2[mask]
    
    # Soft max: log(sum(exp(vals/tau))) * tau
    # Soft min: -log(sum(exp(-vals/tau))) * tau
    # To avoid overflow/underflow, we can shift vals, but since we take ratio, 
    # let's be careful.
    # Actually, minimizing ratio is equivalent to minimizing log(max) - log(min).
    # log(soft_max) = log(tau * log(sum(exp(v/tau)))) ... no.
    # Let M = soft_max = tau * log(sum(exp(v/tau)))
    # Let m = soft_min = -tau * log(sum(exp(-v/tau)))
    # We want to minimize M/m.
    # Gradient of M/m is (m * grad_M - M * grad_m) / m^2.
    
    # Stable computation
    v_max = np.max(vals)
    v_min = np.min(vals)
    
    # Soft max
    exp_vals = np.exp((vals - v_max) / tau)
    sum_exp = np.sum(exp_vals)
    soft_max = v_max + tau * np.log(sum_exp)
    
    # Soft min
    exp_vals_min = np.exp((vals - v_min) / -tau) # exp(-(vals-v_min)/tau)
    sum_exp_min = np.sum(exp_vals_min)
    soft_min = v_min - tau * np.log(sum_exp_min)
    
    ratio = soft_max / soft_min
    
    # Gradient
    # d(soft_max)/dv_i = exp((v_i - v_max)/tau) / sum_exp
    grad_max = exp_vals / sum_exp
    
    # d(soft_min)/dv_i = -exp(-(v_i - v_min)/tau) / sum_exp_min
    grad_min = -exp_vals_min / sum_exp_min
    
    # d(ratio)/dv_i = (soft_min * grad_max_i - soft_max * grad_min_i) / soft_min^2
    grad_vals = (soft_min * grad_max - soft_max * grad_min) / (soft_min ** 2)
    
    # Map grad_vals back to P
    # dist2[i,j] = sum_k (P[i,k] - P[j,k])^2
    # d(dist2[i,j])/dP[i,k] = 2*(P[i,k] - P[j,k])
    # d(dist2[i,j])/dP[j,k] = -2*(P[i,k] - P[j,k])
    
    grad_P = np.zeros_like(P)
    idx_i, idx_j = np.where(mask)
    for i, j in zip(idx_i, idx_j):
        g = grad_vals[i * N + j] # This indexing is wrong, need to map from flat vals to pairs
        
    # Let's do it properly
    grad_P = np.zeros_like(P)
    for i in range(N):
        for j in range(i+1, N):
            g = grad_vals[i * N + j] # Wait, vals is extracted from mask. 
            # The order in vals corresponds to np.where(mask)
            # Let's precompute indices
            pass
            
    # Better:
    idx_i, idx_j = np.where(mask)
    for k in range(len(idx_i)):
        i = idx_i[k]
        j = idx_j[k]
        g = grad_vals[k]
        diff = P[i] - P[j]
        grad_P[i] += 2 * g * diff
        grad_P[j] -= 2 * g * diff
        
    return ratio, grad_P

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = float('inf')
    best_P = None
    
    # Try different initializations and tau schedules
    # We will use L-BFGS-B which is fast for smooth problems
    
    n_restarts = 0
    while time.time() < deadline - 5.0:
        n_restarts += 1
        
        # Initialization: Random points in a sphere
        # Or better: Fibonacci sphere for better spread
        if n_restarts % 3 == 0:
            # Fibonacci sphere
            P = np.zeros((N, D))
            golden_angle = np.pi * (3. - np.sqrt(5.))
            for i in range(N):
                y = 1 - (i / (N - 1)) * 2
                r = np.sqrt(1 - y**2)
                theta = golden_angle * i
                P[i, 0] = np.cos(theta) * r
                P[i, 1] = np.sin(theta) * r
                P[i, 2] = y
        else:
            # Random in unit ball
            P = np.random.randn(N, D)
            norms = np.linalg.norm(P, axis=1, keepdims=True)
            P = P / norms * np.random.rand(N, 1)
            
        # Center the points
        P -= np.mean(P, axis=0)
        
        # Optimization loop with annealing tau
        current_P = P.copy()
        best_local_score = float('inf')
        best_local_P = current_P.copy()
        
        # Start with larger tau, decrease
        taus = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
        
        for tau in taus:
            if time.time() > deadline - 2.0:
                break
                
            # Define objective for scipy
            def obj(x):
                P_flat = x.reshape(N, D)
                val, _ = _soft_min_max_ratio(P_flat, tau)
                return val
            
            def grad(x):
                P_flat = x.reshape(N, D)
                _, g = _soft_min_max_ratio(P_flat, tau)
                return g.flatten()
            
            try:
                res = minimize(
                    obj, 
                    current_P.flatten(), 
                    jac=grad, 
                    method='L-BFGS-B', 
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                current_P = res.x.reshape(N, D)
                
                # Evaluate true score
                dist2 = _pairwise_sq_dists(current_P)
                mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                vals = dist2[mask]
                d_min = np.min(vals)
                d_max = np.max(vals)
                if d_min > 1e-9:
                    score = (d_max / d_min) ** 2
                    if score < best_local_score:
                        best_local_score = score
                        best_local_P = current_P.copy()
            except Exception:
                pass
        
        if best_local_score < best_score:
            best_score = best_local_score
            best_P = best_local_P.copy()
            
    # Final check: if best_P is None (should not happen), return random
    if best_P is None:
        return np.random.randn(N, D)
        
    return best_P

# EVOLVE-BLOCK-START
import itertools
import numpy as np


def construct():
    side = int(np.ceil(N ** (1 / D)))
    grid = list(itertools.product(range(side), repeat=D))[:N]
    return np.array(grid, dtype=float)
# EVOLVE-BLOCK-END
