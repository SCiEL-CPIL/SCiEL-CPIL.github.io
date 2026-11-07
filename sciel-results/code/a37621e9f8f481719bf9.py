"""
Strategy: Parameterized Basis Expansion (Gaussian Mixture Model)
Instead of optimizing 50 raw heights, we model h as a sum of K Gaussians.
h[i] = sum_{k=1}^K A_k * exp(-0.5 * ((x_i - mu_k)/sigma_k)^2)
This reduces the dimensionality from 50 to 3*K.
We use a coarse grid search over the parameters to find a good basin,
then refine with L-BFGS-B. This is robust against local minima in the high-dim space.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# Pre-compute x positions
# The problem statement says "equal intervals of [-1/4, 1/4]" for the 50 points.
# However, the evaluator description says:
# "Choose 50 nonnegative step heights h on equal intervals of [-1/4, 1/4]."
# Wait, the evaluator code uses:
# y = [0, g[0], ..., g[m-1], 0] on the equally spaced points np.linspace(-0.5, 0.5, m + 2)
# where m = len(g) = 2*N - 1 = 99.
# So the grid for y is np.linspace(-0.5, 0.5, 101).
# The heights h are defined on [-1/4, 1/4].
# Let's verify the mapping.
# Usually, if h is on [-1/4, 1/4], the convolution g = h*h will be on [-1/2, 1/2].
# The evaluator places g on linspace(-0.5, 0.5, m+2) which has m+1=100 points?
# No, np.linspace(-0.5, 0.5, m + 2) has m+2 points.
# m = 99. So 101 points.
# Spacing w = 1.0 / (m + 1) = 1/100 = 0.01.
# The points are -0.5, -0.49, ..., 0.5.
# The heights h are 50 points.
# If h is on [-1/4, 1/4], the spacing is (0.5) / 49 = 0.0102...
# But the evaluator's w is 0.01.
# Let's look at the seed: "50 equal heights".
# The value is scale invariant.
# The key is the SHAPE.
# The "x" for the heights h should correspond to the support of h.
# If g is on [-0.5, 0.5], h is likely on [-0.25, 0.25].
# Let's assume the 50 points of h are centered and roughly map to [-0.25, 0.25].
# For the parameterization, we just need a consistent x-grid for the Gaussians.
# We will use x_h = np.linspace(-0.25, 0.25, N) for constructing h.

x_h = np.linspace(-0.25, 0.25, N)

def compute_value(h):
    """Compute the value metric for given heights."""
    h = np.asarray(h, dtype=float)
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        return 0.0

    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    
    y_sq_terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq_terms)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    
    value = l2 / (norm_1 * norm_inf)
    return value

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Parameters for Gaussian Mixture Model:
    # K components
    # Each has: Amplitude A (positive), Mean mu, Sigma sigma (positive)
    # We optimize in log-space for A and sigma to ensure positivity.
    # A = exp(logA), sigma = exp(logSigma)
    
    def get_h_from_params(params, K):
        """
        params: array of shape (3*K,)
        [logA_1, mu_1, logSigma_1, logA_2, mu_2, logSigma_2, ...]
        """
        h = np.zeros(N)
        for k in range(K):
            logA = params[3*k]
            mu = params[3*k + 1]
            logSigma = params[3*k + 2]
            
            A = np.exp(logA)
            sigma = np.exp(logSigma)
            
            # Clip sigma to avoid division by zero or extreme values
            sigma = np.clip(sigma, 1e-3, 1.0)
            
            h += A * np.exp(-0.5 * ((x_h - mu) / sigma) ** 2)
        
        return h

    def objective(params, K):
        h = get_h_from_params(params, K)
        # Scale invariance: we can fix the scale of h to reduce noise,
        # but the optimizer might drift. Since value is scale invariant,
        # we can just compute value.
        val = compute_value(h)
        if val <= 0:
            return 1e10
        return -val

    best_val = compute_value(np.ones(N))
    best_h = np.ones(N)
    
    # Try different numbers of Gaussian components
    K_values = [1, 2, 3, 4]
    
    for K in K_values:
        if time.time() > deadline - 10:
            break
            
        # 1. Coarse Grid Search for Initial Parameters
        # This is much more effective in low dimensions (3*K <= 12)
        
        # Ranges:
        # logA: -2 to 2 (A: 0.13 to 7.4)
        # mu: -0.25 to 0.25
        # logSigma: -2 to 0.5 (Sigma: 0.13 to 1.65)
        
        # We will do a random sampling + local refinement approach
        # which is safer than a full grid in 12D.
        
        n_random_starts = 200
        
        for i in range(n_random_starts):
            if time.time() > deadline - 5:
                break
                
            # Generate random params
            logA = np.random.uniform(-2, 2, K)
            mu = np.random.uniform(-0.25, 0.25, K)
            logSigma = np.random.uniform(-2, 0.5, K)
            
            params = np.zeros(3 * K)
            for k in range(K):
                params[3*k] = logA[k]
                params[3*k + 1] = mu[k]
                params[3*k + 2] = logSigma[k]
            
            # Evaluate start
            val_start = -objective(params, K)
            if val_start > best_val:
                best_val = val_start
                best_h = get_h_from_params(params, K)
            
            # Local Refinement with L-BFGS-B
            # L-BFGS-B is good for smooth functions.
            # The objective is smooth in params (sum of exponentials).
            try:
                res = minimize(
                    lambda p: objective(p, K),
                    params,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 200,
                        'ftol': 1e-8,
                        'gtol': 1e-6
                    }
                )
                
                if res.fun < -best_val:
                    val_opt = -res.fun
                    h_opt = get_h_from_params(res.x, K)
                    # Double check with direct compute to avoid numerical issues in objective wrapper
                    val_direct = compute_value(h_opt)
                    if val_direct > best_val:
                        best_val = val_direct
                        best_h = h_opt
            except Exception:
                pass

    # Final Polish on the best solution found so far
    # We can try to optimize the raw h directly from best_h using Nelder-Mead
    # to fix any discretization errors or shape deviations from the Gaussian model.
    if time.time() < deadline - 5:
        try:
            # Map best_h to log space for Nelder-Mead? 
            # Or just Nelder-Mead on h directly with bounds?
            # Nelder-Mead doesn't handle bounds well.
            # Let's use SLSQP or L-BFGS-B on h directly with bounds [0, inf).
            # But h has 50 dims. L-BFGS-B might work if we start from a good point.
            
            def obj_h(h):
                val = compute_value(h)
                if val <= 0: return 1e10
                return -val
            
            res = minimize(
                obj_h,
                best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={
                    'maxiter': 100,
                    'ftol': 1e-8
                }
            )
            
            if res.fun < -best_val:
                val_opt = -res.fun
                h_opt = res.x
                val_direct = compute_value(h_opt)
                if val_direct > best_val:
                    best_val = val_direct
                    best_h = h_opt
        except Exception:
            pass

    # Ensure non-negativity and finiteness
    best_h = np.maximum(0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    
    return best_h.tolist()

if __name__ == "__main__":
    # Quick test
    h = construct()
    print(f"Score: {compute_value(h)}")
