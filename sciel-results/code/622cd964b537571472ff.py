"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _compute_overlaps(h):
    """Compute overlap at all shifts for step function h."""
    n = len(h)
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    # autocorr[k] = sum_i h[i] * h[(i-k) % n]
    overlap = 1.0 - (2.0 / n) * autocorr
    return overlap

def _smooth_max(values, alpha):
    """Smooth approximation of max using log-sum-exp."""
    m = np.max(values)
    if m > 1e10:
        return m
    shifted = alpha * (values - m)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -50, 50)
    return m + np.log(np.sum(np.exp(shifted))) / alpha

def _objective(h, alpha):
    """Smooth surrogate objective: smooth max of overlaps."""
    overlap = _compute_overlaps(h)
    return _smooth_max(overlap, alpha)

def _enforce_constraints(h):
    """Project h onto feasible set: [0,1]^N with sum = N/2."""
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    target = N / 2.0
    if s > 0 and s < target:
        # Scale up
        h = h * (target / s)
        h = np.clip(h, 0.0, 1.0)
    elif s > target:
        # Scale down
        h = h * (target / s)
        h = np.clip(h, 0.0, 1.0)
    # Final correction: distribute residual
    diff = target - h.sum()
    if abs(diff) > 1e-12:
        # Add/subtract uniformly
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _kkt_polish(h0, alpha, max_iter=200):
    """Polish using SLSQP with smooth objective."""
    n = len(h0)
    target = n / 2.0
    
    bounds = [(0.0, 1.0)] * n
    constraints = [{'type': 'eq', 'fun': lambda x: x.sum() - target, 
                    'jac': lambda x: np.ones(n)}]
    
    def obj(x):
        return _objective(x, alpha)
    
    try:
        res = minimize(obj, h0, method='SLSQP', bounds=bounds, 
                      constraints=constraints,
                      options={'maxiter': max_iter, 'ftol': 1e-14, 'disp': False})
        if res.success or res.fun < _objective(h0, alpha):
            return res.x
    except Exception:
        pass
    return h0

def construct():
    deadline = time.time() + 140.0
    best_h = np.full(N, 0.5)
    best_score = _compute_overlaps(best_h).max()
    
    # Multi-start optimization
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points
    n_starts = 12
    starts = []
    
    # Start 1: uniform
    starts.append(np.full(N, 0.5))
    
    # Start 2: alternating
    alt = np.zeros(N)
    alt[::2] = 1.0
    starts.append(alt)
    
    # Start 3-6: sinusoidal patterns
    for freq in [1, 2, 3, 5]:
        x = np.arange(N)
        h = 0.5 + 0.4 * np.sin(2 * np.pi * freq * x / N)
        h = _enforce_constraints(h)
        starts.append(h)
    
    # Start 7-10: random with structure
    for i in range(4):
        h = rng.uniform(0, 1, N)
        h = _enforce_constraints(h)
        starts.append(h)
    
    # Start 11: block pattern
    blocks = np.zeros(N)
    for i in range(0, N, 10):
        blocks[i:i+5] = 1.0
    blocks = _enforce_constraints(blocks)
    starts.append(blocks)
    
    # Start 12: another pattern
    h = 0.5 + 0.3 * np.cos(4 * np.pi * np.arange(N) / N) + 0.2 * np.sin(8 * np.pi * np.arange(N) / N)
    h = _enforce_constraints(h)
    starts.append(h)
    
    # Optimize each start with increasing alpha (annealing)
    alphas = [10.0, 50.0, 200.0, 1000.0]
    
    for start_idx, h0 in enumerate(starts):
        if time.time() > deadline - 15:
            break
        
        h = h0.copy()
        
        for alpha in alphas:
            if time.time() > deadline - 10:
                break
            
            h = _kkt_polish(h, alpha, max_iter=100)
            h = _enforce_constraints(h)
            
            score = _compute_overlaps(h).max()
            if score < best_score:
                best_score = score
                best_h = h.copy()
        
        # Final high-alpha polish
        if time.time() < deadline - 5:
            h = _kkt_polish(h, 5000.0, max_iter=300)
            h = _enforce_constraints(h)
            score = _compute_overlaps(h).max()
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Final global polish on best solution
    if time.time() < deadline - 2:
        h = _kkt_polish(best_h, 10000.0, max_iter=500)
        h = _enforce_constraints(h)
        score = _compute_overlaps(h).max()
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Ensure exact sum
    best_h = _enforce_constraints(best_h)
    
    return best_h
