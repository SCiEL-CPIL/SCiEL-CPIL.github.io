"""Rewritten program for Erdos minimum overlap."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint, SLSQP
from scipy.linalg import eigh

def _overlap_max(h, shifts=None):
    """Compute max overlap of h with shifted 1-h over all cyclic shifts."""
    if shifts is None:
        shifts = range(N)
    ones_minus_h = 1.0 - h
    max_ov = 0.0
    for s in shifts:
        # overlap = sum_i h[i] * (1-h)[(i-s) mod N]
        # = sum_i h[i] * (1 - h[(i-s) mod N])
        # = sum_i h[i] - sum_i h[i]*h[(i-s) mod N]
        # = 0.5 - autocorr[s]
        # where autocorr[s] = sum_i h[i]*h[(i-s) mod N]
        # So we want to minimize max_s (0.5 - autocorr[s])
        # equivalently maximize min_s autocorr[s]
        shifted = np.roll(h, s)
        ov = np.sum(h * (1.0 - shifted))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _objective_and_grad(h):
    """Smooth surrogate: log-sum-exp of overlaps."""
    ones_minus_h = 1.0 - h
    ov = np.zeros(N)
    for s in range(N):
        shifted = np.roll(h, s)
        ov[s] = np.sum(h * (1.0 - shifted))
    # Use log-sum-exp for smooth max
    m = np.max(ov)
    if m < -1e10:
        return 0.0, np.zeros_like(h)
    lse = m + np.log(np.sum(np.exp(ov - m)))
    # Gradient of log-sum-exp
    weights = np.exp(ov - m) / np.sum(np.exp(ov - m))
    grad = np.zeros_like(h)
    for s in range(N):
        shifted = np.roll(h, s)
        # d/dh_i [sum_j h[j]*(1-h[(j-s) mod N])] = (1 - h[(i-s) mod N]) - h[i]
        d = (1.0 - np.roll(h, -s)) - h
        grad += weights[s] * d
    return lse, grad

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s limit
    
    # Start with uniform 0.5
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = _overlap_max(h)
    
    # Try to improve using gradient-based optimization on smooth surrogate
    # Constraint: sum(h) = N/2, 0 <= h <= 1
    
    # Use SLSQP with bounds and equality constraint
    bounds = [(0.0, 1.0)] * N
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0, 
         'jac': lambda x: np.ones(N)}
    ]
    
    def obj(x):
        val, grad = _objective_and_grad(x)
        return val
    
    def obj_grad(x):
        val, grad = _objective_and_grad(x)
        return grad
    
    # First pass: try a few random initializations
    rng = np.random.RandomState(42)
    
    for trial in range(10):
        if time.time() > deadline - 30:
            break
            
        if trial == 0:
            x0 = best_h.copy()
        else:
            # Random perturbation around 0.5
            x0 = 0.5 + 0.3 * rng.randn(N)
            x0 = np.clip(x0, 0.0, 1.0)
            # Adjust to satisfy sum constraint
            diff = N/2.0 - np.sum(x0)
            x0 += diff / N
            x0 = np.clip(x0, 0.0, 1.0)
            # Re-adjust
            diff = N/2.0 - np.sum(x0)
            x0 += diff / N
            x0 = np.clip(x0, 0.0, 1.0)
        
        try:
            res = minimize(
                obj, x0, jac=obj_grad, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            if res.success or res.fun < best_score:
                h_candidate = np.clip(res.x, 0.0, 1.0)
                # Enforce sum constraint
                diff = N/2.0 - np.sum(h_candidate)
                h_candidate += diff / N
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
                
                score = _overlap_max(h_candidate)
                if score < best_score:
                    best_score = score
                    best_h = h_candidate.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 30:
            break
    
    # Final polish: use the best solution and try a few more iterations
    if time.time() < deadline - 10:
        try:
            res = minimize(
                obj, best_h, jac=obj_grad, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            h_candidate = np.clip(res.x, 0.0, 1.0)
            diff = N/2.0 - np.sum(h_candidate)
            h_candidate += diff / N
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
            
            score = _overlap_max(h_candidate)
            if score < best_score:
                best_score = score
                best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Ensure exact sum
    diff = N/2.0 - np.sum(best_h)
    best_h += diff / N
    best_h = np.clip(best_h, 0.0, 1.0)
    
    # Final check and return
    return best_h
# EVOLVE-BLOCK-END
