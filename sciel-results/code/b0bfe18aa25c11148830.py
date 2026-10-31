import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _calculate_ratio(h):
    """
    Calculate the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    h is a numpy array of nonnegative heights.
    """
    # Ensure non-negative
    h = np.maximum(h, 0)
    
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
        
    l2_sq = np.sum(h**2)
    
    # Autoconvolution
    # np.convolve with mode='full' gives length 2N-1
    conv = np.convolve(h, h, mode='full')
    
    max_conv = np.max(conv)
    
    if max_conv == 0:
        return 0.0
        
    return l2_sq / (l1 * max_conv)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_score = 0.0
    best_h = None
    
    # Candidate 1: Spike (Theoretical Max for Interpretation A)
    h_spike = np.zeros(N)
    h_spike[0] = 1.0
    score_spike = _calculate_ratio(h_spike)
    if score_spike > best_score:
        best_score = score_spike
        best_h = h_spike.copy()
        
    # Candidate 2: Uniform (Seed)
    h_uniform = np.ones(N)
    score_uniform = _calculate_ratio(h_uniform)
    if score_uniform > best_score:
        best_score = score_uniform
        best_h = h_uniform.copy()
        
    # Candidate 3: Triangular
    h_tri = np.arange(1, N+1)
    h_tri = h_tri / np.sum(h_tri)
    score_tri = _calculate_ratio(h_tri)
    if score_tri > best_score:
        best_score = score_tri
        best_h = h_tri.copy()
        
    # Candidate 4: Random Search + Local Optimization
    # We try to maximize the ratio.
    # Since the ratio is scale invariant, we can fix sum(h)=1.
    # Objective to minimize: -log(ratio) or just -ratio.
    # Ratio is non-smooth. We use a smooth approximation for optimization, 
    # but evaluate the true ratio for selection.
    
    def objective_smooth(h):
        # h is expected to be non-negative.
        # We minimize -log(sum(h^2)) + log(max(h*h)) approx
        # Actually, let's just minimize -ratio using a smooth max.
        # To use gradients, we need differentiability.
        # Let's minimize: logsumexp(h*h) - log(sum(h^2))
        # This is equivalent to minimizing log(max) - log(norm2_sq)
        # Which is -log(ratio * l1). Since l1 is fixed to 1, it's -log(ratio).
        
        conv = np.convolve(h, h, mode='full')
        # Soft max approximation
        # Temperature tau
        tau = 0.05 
        soft_max = tau * np.log(np.sum(np.exp(conv / tau)))
        
        l2_sq = np.sum(h**2)
        if l2_sq < 1e-10:
            return 1e10
            
        return soft_max - np.log(l2_sq)
    
    def gradient_smooth(h):
        # Numerical gradient or analytic?
        # Analytic is better.
        # f = logsumexp(conv) - log(l2_sq)
        # d(logsumexp)/dh = (h * softmax(conv)) ... complex.
        # Let's use finite differences or just rely on L-BFGS-B numerical gradients.
        # L-BFGS-B with approx_grad=True is slower but robust.
        # Given N=50, it's small.
        pass

    # We will run a few random restarts with L-BFGS-B
    np.random.seed(42)
    
    for i in range(20):
        if time.time() > deadline:
            break
            
        # Random initial condition, normalized to sum 1
        h_init = np.random.exponential(1.0, N)
        h_init = h_init / np.sum(h_init)
        
        # Bounds: [0, inf)
        bounds = [(0, None)] * N
        
        try:
            res = minimize(
                objective_smooth, 
                h_init, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            
            h_opt = res.x
            # Ensure non-negative
            h_opt = np.maximum(h_opt, 0)
            
            # Normalize to sum 1 for fair comparison? 
            # The ratio is scale invariant, so normalization doesn't change the value.
            # But we should check the raw value.
            score_opt = _calculate_ratio(h_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_h = h_opt.copy()
                
        except Exception:
            continue
            
    # Final check: Return the best found
    if best_h is None:
        # Fallback to spike if something went wrong
        return [1.0] + [0.0] * (N - 1)
        
    # The task asks for a sequence of nonnegative step heights.
    # We return the list of floats.
    return list(best_h)

# EVOLVE-BLOCK-END
