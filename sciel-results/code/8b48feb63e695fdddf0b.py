"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 50

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|)."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(np.abs(conv))
    if max_conv <= 0:
        return 0.0
    # The continuous autoconvolution peak: for discrete heights h_i on intervals of width DELTA,
    # the piecewise-constant function has autoconvolution peak = max_conv * DELTA
    DELTA = 1.0 / (2 * N)
    max_val = max_conv * DELTA
    if max_val <= 0:
        return 0.0
    return l2_sq / (l1 * max_val)

def _greedy_bipartite_assignment(deadline):
    """
    Greedy heuristic on bipartite graph: left nodes = indices, right nodes = lags.
    Iteratively assign heights to minimize max lag contribution while maximizing sum of squares.
    """
    h = np.zeros(N)
    assigned = [False] * N
    
    # Start with a random seed index
    start_idx = random.randint(0, N - 1)
    h[start_idx] = 1.0
    assigned[start_idx] = True
    
    # Greedily add indices that minimize the increase in max convolution
    for _ in range(N - 1):
        if time.time() > deadline:
            break
        
        best_idx = -1
        best_ratio = -1
        best_h_candidate = None
        
        for idx in range(N):
            if assigned[idx]:
                continue
            
            # Try adding this index with a small weight
            h_trial = h.copy()
            h_trial[idx] = 1.0
            
            ratio = _compute_ratio(h_trial)
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = idx
                best_h_candidate = h_trial.copy()
        
        if best_idx >= 0:
            h = best_h_candidate
            assigned[best_idx] = True
        else:
            break
    
    return h

def _local_search_polish(h, deadline, max_iterations=200):
    """
    Local search: try perturbing individual heights and accept if ratio improves.
    Uses a smooth surrogate: we optimize in log-space for positivity.
    """
    h = np.array(h, dtype=np.float64)
    if np.sum(h) <= 0:
        return h
    
    # Normalize to L1=1
    h = h / np.sum(h)
    
    current_ratio = _compute_ratio(h)
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try perturbing a random index
        idx = random.randint(0, N - 1)
        old_val = h[idx]
        
        # Try different values
        for scale in [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 0.0]:
            h_trial = h.copy()
            h_trial[idx] = old_val * scale if old_val > 0 else 1.0
            
            if np.any(h_trial < 0):
                continue
            
            l1 = np.sum(h_trial)
            if l1 <= 0:
                continue
            
            h_trial = h_trial / l1
            ratio = _compute_ratio(h_trial)
            
            if ratio > current_ratio:
                h = h_trial
                current_ratio = ratio
                improved = True
                break
        
        if not improved:
            # Try a different index
            continue
    
    return h

def _gradient_polish(h, deadline, max_iterations=100):
    """
    Gradient-based polish using finite differences.
    Optimizes in log-space: let h_i = exp(x_i), then optimize over x.
    """
    h = np.array(h, dtype=np.float64)
    if np.sum(h) <= 0:
        return h
    
    h = h / np.sum(h)
    
    # Work in log-space
    x = np.log(np.maximum(h, 1e-10))
    
    current_ratio = _compute_ratio(h)
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Compute gradient via finite differences
        grad = np.zeros(N)
        eps = 1e-4
        
        for i in range(N):
            x_plus = x.copy()
            x_plus[i] += eps
            h_plus = np.exp(x_plus)
            h_plus = h_plus / np.sum(h_plus)
            r_plus = _compute_ratio(h_plus)
            
            x_minus = x.copy()
            x_minus[i] -= eps
            h_minus = np.exp(x_minus)
            h_minus = h_minus / np.sum(h_minus)
            r_minus = _compute_ratio(h_minus)
            
            grad[i] = (r_plus - r_minus) / (2 * eps)
        
        # Gradient ascent step
        lr = 0.01
        x = x + lr * grad
        h = np.exp(x)
        h = h / np.sum(h)
        
        new_ratio = _compute_ratio(h)
        if new_ratio < current_ratio:
            # Revert
            h = np.exp(x - lr * grad)
            h = h / np.sum(h)
            x = x - lr * grad
            current_ratio = _compute_ratio(h)
        else:
            current_ratio = new_ratio
    
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    best_h = None
    best_ratio = 0.0
    
    # Multiple random restarts with the greedy bipartite heuristic
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Greedy construction
        h = _greedy_bipartite_assignment(deadline)
        
        if np.sum(h) <= 0:
            continue
        
        # Local search polish
        h = _local_search_polish(h, deadline, max_iterations=100)
        
        # Gradient polish
        h = _gradient_polish(h, deadline, max_iterations=50)
        
        ratio = _compute_ratio(h)
        if ratio > best_ratio:
            best_ratio = ratio
            best_h = h.copy()
        
        # Check if we're running out of time
        if time.time() > deadline - 20:
            break
    
    # Fallback to a simple Golomb ruler if nothing found
    if best_h is None or best_ratio <= 0:
        # Simple Golomb ruler
        ruler = [0, 1, 4, 10, 18, 23, 29, 36]
        best_h = np.zeros(N)
        for idx in ruler:
            if idx < N:
                best_h[idx] = 1.0
        best_ratio = _compute_ratio(best_h)
    
    return [float(x) for x in best_h]
# EVOLVE-BLOCK-END
