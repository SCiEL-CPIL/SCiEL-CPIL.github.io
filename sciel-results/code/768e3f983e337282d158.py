"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h (adjacent shift)."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _overlap_max(h):
    """Compute max of h[i]*(1-h[i+1]) with zero-padding at the end."""
    # h[i] * (1 - h[i+1]) for i=0..N-2, and h[N-1] * (1 - 0) = h[N-1]
    overlaps = h[:-1] * (1.0 - h[1:])
    # Last element: h[N-1] * (1 - 0) = h[N-1]
    overlaps = np.append(overlaps, h[-1])
    return overlaps.max()

def _smooth_max(values, alpha):
    """Smooth approximation of max using log-sum-exp."""
    m = np.max(values)
    if m > 1e10:
        return m
    shifted = alpha * (values - m)
    shifted = np.clip(shifted, -50, 50)
    return m + np.log(np.sum(np.exp(shifted))) / alpha

def _objective(h, alpha):
    """Smooth surrogate objective: smooth max of adjacent overlaps."""
    overlaps = h[:-1] * (1.0 - h[1:])
    overlaps = np.append(overlaps, h[-1])
    return _smooth_max(overlaps, alpha)

def _objective_grad(h, alpha):
    """Gradient of the smooth max objective."""
    overlaps = h[:-1] * (1.0 - h[1:])
    overlaps = np.append(overlaps, h[-1])
    m = np.max(overlaps)
    shifted = alpha * (overlaps - m)
    shifted = np.clip(shifted, -50, 50)
    exp_vals = np.exp(shifted)
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient of overlaps with respect to h
    # overlap[i] = h[i]*(1-h[i+1]) for i < N-1
    # overlap[N-1] = h[N-1]
    grad = np.zeros(N)
    # Vectorized gradient calculation
    # d(overlap[i])/dh[i] = (1 - h[i+1])
    # d(overlap[i])/dh[i+1] = -h[i]
    
    # Contribution to grad[i] from overlap[i]: weights[i] * (1 - h[i+1])
    # Contribution to grad[i] from overlap[i-1]: weights[i-1] * (-h[i-1])
    
    # For i=0: only from overlap[0]
    grad[0] = weights[0] * (1.0 - h[1])
    
    # For i=1..N-2: from overlap[i] and overlap[i-1]
    if N > 2:
        grad[1:N-1] = weights[1:N-1] * (1.0 - h[2:N]) - weights[0:N-2] * h[0:N-2]
        
    # For i=N-1: from overlap[N-1] and overlap[N-2]
    # overlap[N-1] = h[N-1], grad is 1.0
    # overlap[N-2] = h[N-2]*(1-h[N-1]), grad is -h[N-2]
    grad[N-1] = weights[N-1] * 1.0 - weights[N-2] * h[N-2]
    
    return grad

def _enforce_constraints(h):
    """Project h onto feasible set: [0,1]^N with sum = N/2."""
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    target = N / 2.0
    if abs(s - target) > 1e-12:
        diff = target - s
        # Distribute residual to elements not at boundary
        can_add = h < 1.0
        can_sub = h > 0.0
        if diff > 0:
            mask = can_add
            if mask.sum() > 0:
                h[mask] += diff / mask.sum()
            else:
                h += diff / N
        else:
            mask = can_sub
            if mask.sum() > 0:
                h[mask] += diff / mask.sum()
            else:
                h += diff / N
        h = np.clip(h, 0.0, 1.0)
    # Final correction
    diff = target - h.sum()
    if abs(diff) > 1e-12:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _optimize_from_start(h0, alpha, max_iter=200):
    """Optimize from a starting point using SLSQP."""
    n = len(h0)
    target = n / 2.0
    
    bounds = [(0.0, 1.0)] * n
    constraints = [{'type': 'eq', 'fun': lambda x: x.sum() - target, 
                    'jac': lambda x: np.ones(n)}]
    
    def obj(x):
        return _objective(x, alpha)
    
    def obj_grad(x):
        return _objective_grad(x, alpha)
    
    try:
        res = minimize(obj, h0, jac=obj_grad, method='SLSQP', bounds=bounds, 
                      constraints=constraints,
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False})
        if res.success or (hasattr(res, 'fun') and res.fun < _objective(h0, alpha)):
            return res.x
    except Exception:
        pass
    return h0

def _generate_initial_guesses(deadline):
    """Generate diverse initial guesses."""
    guesses = []
    
    # 1. Uniform
    guesses.append(np.full(N, 0.5))
    
    # 2. Ramp up
    h = np.linspace(0, 1, N)
    guesses.append(_enforce_constraints(h))
    
    # 3. Ramp down
    h = np.linspace(1, 0, N)
    guesses.append(_enforce_constraints(h))
    
    # 4. Sine wave (mean 0.5)
    t = np.linspace(0, 2*np.pi, N)
    h = 0.5 + 0.5 * np.sin(t)
    guesses.append(_enforce_constraints(h))
    
    # 5. Cosine wave
    t = np.linspace(0, 2*np.pi, N)
    h = 0.5 + 0.5 * np.cos(t)
    guesses.append(_enforce_constraints(h))
    
    # 6. Block pattern (high in middle)
    h = np.zeros(N)
    h[N//4 : 3*N//4] = 1.0
    guesses.append(_enforce_constraints(h))
    
    # 7. Random noise
    np.random.seed(42)
    h = np.random.rand(N)
    guesses.append(_enforce_constraints(h))
    
    # 8. Smooth noise
    np.random.seed(123)
    h = np.random.rand(N)
    # Simple smoothing
    for _ in range(5):
        h = np.convolve(h, np.ones(3)/3, mode='same')
    guesses.append(_enforce_constraints(h))
    
    return guesses

def construct():
    deadline = time.time() + 145.0  # Safety margin for 150s limit
    best_h = np.full(N, 0.5)
    best_score = _overlap_max(best_h)
    
    # Try different alpha values for smooth max
    # Lower alpha is smoother, higher alpha is closer to true max
    alphas = [50, 100, 200, 500]
    
    guesses = _generate_initial_guesses(deadline)
    
    for alpha in alphas:
        if time.time() > deadline - 20:
            break
            
        for h0 in guesses:
            if time.time() > deadline - 10:
                break
                
            h_opt = _optimize_from_start(h0, alpha, max_iter=300)
            h_opt = _enforce_constraints(h_opt)
            score = _overlap_max(h_opt)
            
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
                
    # Final polish with higher alpha and more iterations on the best solution
    if time.time() < deadline - 5:
        h_polish = _optimize_from_start(best_h, 1000, max_iter=500)
        h_polish = _enforce_constraints(h_polish)
        score_polish = _overlap_max(h_polish)
        
        if score_polish < best_score:
            best_score = score_polish
            best_h = h_polish
    
    # Ensure exact sum and bounds
    best_h = _enforce_constraints(best_h)
    
    return best_h
