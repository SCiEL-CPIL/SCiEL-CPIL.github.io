import numpy as np
from scipy.optimize import minimize

N = 600

def _score(h):
    """Compute the normalized maximum autoconvolution value."""
    h = np.asarray(h, dtype=float)
    total = h.sum()
    if total <= 0:
        return float('inf')
    # Linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    max_conv = conv.max()
    return max_conv / (total * total)

def _objective_smoothed(h, epsilon=0.01):
    """
    Smoothed objective function to allow gradient-based optimization.
    We minimize the log-sum-exp of the normalized autocorrelation coefficients.
    This approximates the max, which is the actual metric.
    """
    h = np.asarray(h, dtype=float)
    total = h.sum()
    if total <= 1e-10:
        return 1e10
    
    # Normalize h to have sum 1 to keep numbers stable, 
    # the score is scale invariant anyway.
    h_norm = h / total
    
    # Compute autocorrelation
    # We only need the positive lags and zero lag for the max (due to symmetry and non-negativity? 
    # Actually for non-negative h, the max is likely at k=0 or small k. 
    # But let's compute all to be safe.
    conv = np.convolve(h_norm, h_norm, mode='full')
    
    # The score is max(conv). 
    # We approximate max(conv) with log-sum-exp(conv / epsilon) * epsilon.
    # To avoid overflow, we subtract the max.
    c_max = conv.max()
    if c_max <= 0:
        return 0.0
        
    # log-sum-exp stability
    shifted = (conv - c_max) / epsilon
    # Clip to avoid exp underflow/overflow issues in log
    # np.log(np.sum(np.exp(shifted))) + c_max/epsilon
    # We want the value that minimizes the max.
    # The score is c_max / 1 (since total=1).
    # So we just need to minimize c_max.
    # Using log-sum-exp gives a smooth approximation to c_max.
    # val = epsilon * (c_max/epsilon + log(sum(exp((conv-c_max)/epsilon))))
    # val = c_max + epsilon * log(sum(exp((conv-c_max)/epsilon)))
    
    # Since we want to minimize the max, and log-sum-exp is an upper bound on max,
    # minimizing it is a good strategy.
    
    # However, L-BFGS-B needs gradients. 
    # The gradient of log-sum-exp is easy.
    # But implementing it manually might be error-prone.
    # Let's just use the raw max and hope the optimizer handles it, or use a simpler smooth max.
    # Actually, a common smooth max is:
    # M = epsilon * log(sum(exp(x/epsilon)))
    # This is differentiable.
    
    # Let's compute the smooth max value.
    # We return this value. The optimizer will minimize it.
    
    # To prevent numerical issues with large exponents:
    # max_val = c_max
    # sum_exp = np.sum(np.exp((conv - max_val) / epsilon))
    # smooth_max = max_val + epsilon * np.log(sum_exp)
    
    # If we use this as objective, we minimize the smooth max of the normalized autocorrelation.
    
    # One catch: The problem asks for the max of the *normalized* autoconvolution.
    # Since we normalized h to sum 1, the score is exactly max(conv).
    
    return c_max + epsilon * np.log(np.sum(np.exp((conv - c_max) / epsilon)))

def construct():
    best_h = None
    best_score = float('inf')
    
    # Define bounds: h_i >= 0
    bounds = [(0, None)] * N
    
    # We will try several initializations
    # 1. Constant
    # 2. Random
    # 3. Sinusoidal
    
    inits = []
    inits.append(np.ones(N))
    
    # Add some random initializations
    np.random.seed(42)
    for _ in range(5):
        h_rand = np.random.rand(N)
        inits.append(h_rand)
        
    # Add sinusoidal initializations
    for freq in [1, 2, 5]:
        x = np.arange(N)
        h_sin = 1.0 + 0.5 * np.sin(2 * np.pi * freq * x / N)
        inits.append(np.maximum(h_sin, 0))
        
    for h0 in inits:
        # Ensure non-negative
        h0 = np.maximum(h0, 0)
        if h0.sum() == 0:
            continue
            
        # Optimize
        try:
            res = minimize(
                _objective_smoothed, 
                h0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            h_opt = res.x
            # Ensure non-negative
            h_opt = np.maximum(h_opt, 0)
            
            # Calculate the actual (unsmoothed) score
            s = _score(h_opt)
            
            if s < best_score:
                best_score = s
                best_h = h_opt.copy()
                
        except Exception as e:
            pass
            
    if best_h is None:
        return np.ones(N).tolist()
        
    return best_h.tolist()
