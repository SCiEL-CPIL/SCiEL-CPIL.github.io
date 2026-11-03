"""Erdos minimum overlap: find h in [0,1]^N with sum N/2 minimizing max_k sum_i h_i(1-h_{i+k}).

Strategy:
- The objective is a quadratic function of h with structure: for each shift k, overlap is
  sum_i h_i * (1 - h_{i+k}).
- We use gradient descent with projected gradient (clip to [0,1]) and a repair step to
  enforce sum = N/2 exactly.
- We also try a structured initial guess: a "staircase" or block pattern that might
  have low overlap.
- Deadline guard: stop before 140s and return best found.
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 100

def _overlap_score(h):
    """Compute max over all shifts k of sum_i h[i] * (1 - h[i+k]) for linear (truncated) shift."""
    n = len(h)
    max_ov = 0.0
    for k in range(1, n):
        # h[i] for i=0..n-k-1, h[i+k] for i=0..n-k-1
        a = h[:n-k]
        b = h[k:]
        ov = np.sum(a * (1.0 - b))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _overlap_score_grad(h):
    """Compute (score, gradient) where score = max_k overlap_k, gradient w.r.t. h."""
    n = len(h)
    best_k = 0
    best_ov = -1.0
    # Precompute overlaps
    for k in range(1, n):
        a = h[:n-k]
        b = h[k:]
        ov = np.sum(a * (1.0 - b))
        if ov > best_ov:
            best_ov = ov
            best_k = k
    # Gradient of overlap_{best_k} = sum_i h[i]*(1-h[i+best_k])
    k = best_k
    grad = np.zeros(n)
    if k < n:
        idx_a = np.arange(n - k)
        idx_b = idx_a + k
        # d/dh[i] [h[i]*(1-h[i+k])] = (1-h[i+k]) for i in 0..n-k-1
        # d/dh[i+k] [h[i]*(1-h[i+k])] = -h[i] for i in 0..n-k-1
        grad[idx_a] += (1.0 - h[idx_b])
        grad[idx_b] += (-h[idx_a])
    return best_ov, grad

def _project_to_simplex(h, target_sum):
    """Project h onto {x in [0,1]^n : sum(x) = target_sum} approximately via clipping and scaling."""
    h = np.clip(h, 0.0, 1.0)
    current_sum = np.sum(h)
    if current_sum == 0:
        return np.full(N, target_sum / N)
    # Simple adjustment: scale then clip, then fix
    scale = target_sum / current_sum
    h2 = np.clip(h * scale, 0.0, 1.0)
    # If still off, do a more careful projection
    diff = target_sum - np.sum(h2)
    if abs(diff) > 1e-12:
        # Distribute diff evenly, then re-clip
        step = diff / N
        h2 = h2 + step
        h2 = np.clip(h2, 0.0, 1.0)
        # Second pass
        diff2 = target_sum - np.sum(h2)
        if abs(diff2) > 1e-12:
            step2 = diff2 / N
            h2 = h2 + step2
            h2 = np.clip(h2, 0.0, 1.0)
    return h2

def _gradient_descent(init_h, max_iter=20000, lr_start=0.05, lr_end=0.001, patience=5000):
    h = init_h.copy().astype(float)
    h = _project_to_simplex(h, N / 2.0)
    best_h = h.copy()
    best_score = _overlap_score(h)
    lr = lr_start
    no_improve = 0
    for it in range(max_iter):
        score, grad = _overlap_score_grad(h)
        if score < best_score - 1e-10:
            best_score = score
            best_h = h.copy()
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
        # Adam-like step
        h = h - lr * grad
        h = _project_to_simplex(h, N / 2.0)
        # Decay learning rate
        lr = max(lr_end, lr_start * (lr_end / lr_start) ** (it / max(1, max_iter)))
    return best_h, best_score

def _random_init():
    """Random h in [0,1]^N with sum N/2."""
    h = np.random.rand(N)
    h = _project_to_simplex(h, N / 2.0)
    return h

def _staircase_init():
    """Create a staircase-like initial guess that might have low overlap."""
    # Try: first half 0, second half 1 (step function)
    h1 = np.concatenate([np.zeros(N // 2), np.ones(N - N // 2)])
    # Try: alternating blocks
    block = N // 10
    h2 = np.tile([0, 1], N // 2)[:N]
    # Try: smooth ramp
    x = np.linspace(0, 1, N)
    h3 = x.copy()
    h3 = _project_to_simplex(h3, N / 2.0)
    return [h1, h2, h3]

def construct():
    deadline = time.time() + 140.0  # 10s safety margin
    
    best_h = None
    best_score = float('inf')
    
    # Try structured initializations
    inits = _staircase_init()
    for init in inits:
        if time.time() > deadline:
            break
        h, score = _gradient_descent(init, max_iter=5000, patience=2000)
        if score < best_score:
            best_score = score
            best_h = h
    
    # Try random initializations with decreasing learning rates
    n_random = 30
    for i in range(n_random):
        if time.time() > deadline:
            break
        init = _random_init()
        # Use shorter runs for later iterations
        max_iter = max(1000, 5000 - i * 100)
        patience = max(500, 2000 - i * 50)
        h, score = _gradient_descent(init, max_iter=max_iter, lr_start=0.03, lr_end=0.0005, patience=patience)
        if score < best_score:
            best_score = score
            best_h = h
    
    if best_h is None:
        # Fallback: step function
        best_h = np.concatenate([np.zeros(N // 2), np.ones(N - N // 2)])
    
    # Ensure exact sum
    best_h = _project_to_simplex(best_h, N / 2.0)
    
    # Convert to list of floats for return
    return list(best_h)
