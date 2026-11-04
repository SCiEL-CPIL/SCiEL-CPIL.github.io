import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 110.0 # Safety margin

def _score(h):
    h = np.maximum(h, 0)
    s = np.sum(h)
    if s == 0:
        return 0.0
    h = h / s # Normalize L1 to 1 (discrete sum)
    # Note: The actual score formula in the current program was:
    # score = (sum h^2 * dx) / (max conv * dx) = sum h^2 / max conv
    # If we normalize discrete sum to 1, then sum h = 1.
    # The ratio is scale invariant.
    # So we can just work with sum h = 1.
    
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    # Use log-sum-exp to approximate max
    # To avoid overflow, subtract max
    max_val = np.max(conv)
    if max_val == 0:
        return 0.0
    # beta for log-sum-exp. Higher beta -> sharper max approximation.
    beta = 100.0
    # log-sum-exp: max + log(sum(exp(beta*(x-max))))/beta
    # But we want to maximize l2_sq / max_conv.
    # This is equivalent to minimizing -log(l2_sq) + log(max_conv).
    # Or just minimize -score.
    # Let's just compute score directly for the final evaluation,
    # but for optimization, we need a differentiable surrogate.
    
    # Surrogate for max(conv):
    # m = max(conv)
    # m_approx = (1/beta) * log(sum(exp(beta * conv)))
    # This is stable if we shift.
    shifted = beta * (conv - max_val)
    m_approx = max_val + (1.0/beta) * np.log(np.sum(np.exp(shifted)))
    
    return l2_sq / m_approx

def _negative_score(h):
    return -_score(h)

def construct():
    start_time = time.time()
    best_h = None
    best_score = 0.0
    
    # Initial guess: Asymmetric power law
    # Try a few random restarts
    
    for restart in range(10):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Generate initial guess
        # Random asymmetric shape
        # h_i = (i+1)^alpha * noise
        alpha = np.random.uniform(0.5, 3.0)
        h0 = np.zeros(N)
        # Use first K elements
        K = np.random.randint(5, 25)
        x = np.arange(1, K+1)
        h0[:K] = x ** alpha
        # Add small noise to break symmetry/degeneracy
        h0[:K] *= (1 + 0.1 * np.random.randn(K))
        h0 = np.maximum(h0, 0)
        
        if np.sum(h0) == 0:
            continue
            
        # Optimize
        # Bounds: h_i >= 0
        bounds = [(0, None) for _ in range(N)]
        
        try:
            res = minimize(_negative_score, h0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 50, 'ftol': 1e-6})
            h_opt = res.x
            score_opt = _score(h_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_h = h_opt
        except Exception:
            pass

    if best_h is None:
        # Fallback to simple asymmetric shape
        h = np.zeros(N)
        h[:10] = np.arange(1, 11)
        return h.tolist()
        
    # Normalize to match the expected interface if necessary?
    # The evaluator likely normalizes or the ratio is scale invariant.
    # The current program returns heights that sum to 1/dx in continuous sense?
    # The prompt says "50 nonnegative heights".
    # The score is scale invariant.
    # So we can return h_opt directly.
    
    return best_h.tolist()
