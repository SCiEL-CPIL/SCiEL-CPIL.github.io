import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import sys

N = 50
# Grid points for h
# h is defined on equal intervals of [-1/4, 1/4]
# The problem says "Choose 50 nonnegative step heights h on equal intervals of [-1/4, 1/4]"
# This usually means h is a vector of length 50.
# The convolution g = np.convolve(h, h) has length 2*50 - 1 = 99.
# The evaluator computes:
# m = len(g) = 99
# y = [0, g[0], ..., g[m-1], 0] -> length m+2 = 101
# Points are np.linspace(-0.5, 0.5, m + 2)
# Spacing w = 1.0 / (m+1) = 1.0 / 100 = 0.01

def compute_value(h):
    h = np.array(h)
    if np.any(h < 0) or np.any(~np.isfinite(h)):
        return -np.inf
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    
    # l2 = sum over i in 0..m of (w / 3) * (y[i]^2 + y[i] * y[i+1] + y[i+1]^2)
    # y has indices 0 to m+1
    # The sum is i=0 to m
    # Terms: y[0]^2 + y[0]y[1] + y[1]^2
    #        y[1]^2 + y[1]y[2] + y[2]^2
    #        ...
    #        y[m]^2 + y[m]y[m+1] + y[m+1]^2
    #
    # Sum of squares: y[0]^2 + 2*sum(y[1:m+1]^2) + y[m+1]^2 ? No.
    # Let's just compute it directly or vectorize.
    # y_sq = y**2
    # cross = y[:m+1] * y[1:m+2]
    # l2 = (w/3) * (sum(y_sq[:m+1]) + sum(cross) + sum(y_sq[1:m+2]))
    
    y_sq = y**2
    cross = y[:m+1] * y[1:m+2]
    l2 = (w / 3.0) * (np.sum(y_sq[:m+1]) + np.sum(cross) + np.sum(y_sq[1:m+2]))
    
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
        
    value = l2 / (norm_1 * norm_inf)
    return value

def objective(params):
    # params: [a1_log, mu1_tanh, sigma1_log, a2_log, mu2_tanh, sigma2_log, ...]
    K = len(params) // 3
    h = np.zeros(N)
    
    # Grid for h
    # "50 nonnegative step heights h on equal intervals of [-1/4, 1/4]"
    # This implies the domain of h is [-0.25, 0.25].
    # The step heights are values at 50 points.
    # Let's assume the points are centered or include endpoints?
    # "on equal intervals of [-1/4, 1/4]" usually implies the support is [-0.25, 0.25].
    # If we use np.linspace(-0.25, 0.25, 50), the spacing is 0.5/49.
    # However, the convolution result g is then sampled on a grid.
    # The evaluator uses np.linspace(-0.5, 0.5, m+2) for y.
    # This implies the convolution is done on the discrete grid of h.
    # The physical interpretation is that h represents a function on [-0.25, 0.25].
    
    # Let's generate the x-coordinates for the 50 points of h.
    # Standard practice for such "step height" problems on an interval [a,b] with N points:
    # x = np.linspace(-0.25, 0.25, N)
    
    x_h = np.linspace(-0.25, 0.25, N)
    
    for k in range(K):
        a_k = np.exp(params[3*k])
        mu_k = 0.25 * np.tanh(params[3*k+1])
        sigma_k = np.exp(params[3*k+2])
        
        h += a_k * np.exp(-0.5 * ((x_h - mu_k) / sigma_k)**2)
        
    return -compute_value(h) # Minimize negative value

def polish(h_init):
    # Use L-BFGS-B to optimize the 50-dimensional h directly
    # Constraint: h >= 0
    bounds = [(0, None)] * N
    
    def obj(h):
        return -compute_value(h)
    
    res = minimize(obj, h_init, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10})
    return res.x

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Global Search Parameters
    K = 4 # Number of Gaussian components
    n_params = 3 * K
    
    # Bounds for differential evolution
    # a_log: log(a). a can be large. Let's say 0 to 10 (a up to 22000)
    # mu_tanh: tanh(mu/0.25). Range -1 to 1.
    # sigma_log: log(sigma). sigma should be small enough to fit in interval. 0.01 to 0.5. log: -4.6 to -0.7
    bounds = []
    for _ in range(K):
        bounds.append((0, 10)) # a_log
        bounds.append((-2, 2)) # mu_tanh (covers most of range)
        bounds.append((-4.0, -0.5)) # sigma_log
    
    best_val = -np.inf
    best_h = None
    
    # Run Differential Evolution
    try:
        de_res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15,
            tol=1e-4,
            polish=False, # We will polish manually
            init='latinhypercube',
            update_interval=1
        )
        
        if de_res.success or de_res.fun < best_val:
            best_params = de_res.x
            best_h_candidate = None
            # Reconstruct h from params
            x_h = np.linspace(-0.25, 0.25, N)
            h_candidate = np.zeros(N)
            for k in range(K):
                a_k = np.exp(best_params[3*k])
                mu_k = 0.25 * np.tanh(best_params[3*k+1])
                sigma_k = np.exp(best_params[3*k+2])
                h_candidate += a_k * np.exp(-0.5 * ((x_h - mu_k) / sigma_k)**2)
            
            val = compute_value(h_candidate)
            if val > best_val:
                best_val = val
                best_h = h_candidate
    except Exception as e:
        print(f"DE failed: {e}")
    
    # If DE didn't run or failed, try a few random starts
    if best_h is None:
        # Random Gaussian mixture starts
        for _ in range(20):
            if time.time() > deadline:
                break
            params = np.random.uniform(bounds[0][0], bounds[0][1], size=n_params)
            # Adjust bounds properly for random gen
            params[0::3] = np.random.uniform(0, 10, K)
            params[1::3] = np.random.uniform(-2, 2, K)
            params[2::3] = np.random.uniform(-4, -0.5, K)
            
            x_h = np.linspace(-0.25, 0.25, N)
            h_candidate = np.zeros(N)
            for k in range(K):
                a_k = np.exp(params[3*k])
                mu_k = 0.25 * np.tanh(params[3*k+1])
                sigma_k = np.exp(params[3*k+2])
                h_candidate += a_k * np.exp(-0.5 * ((x_h - mu_k) / sigma_k)**2)
            
            val = compute_value(h_candidate)
            if val > best_val:
                best_val = val
                best_h = h_candidate

    if best_h is None:
        # Fallback to seed
        return [1.0] * N

    # Polish with L-BFGS-B
    if time.time() < deadline - 10.0:
        try:
            polished_h = polish(best_h)
            val_polished = compute_value(polished_h)
            if val_polished > best_val:
                best_h = polished_h
                best_val = val_polished
        except Exception as e:
            print(f"Polish failed: {e}")

    # Ensure non-negative and finite
    best_h = np.clip(best_h, 0, None)
    
    return list(best_h)
