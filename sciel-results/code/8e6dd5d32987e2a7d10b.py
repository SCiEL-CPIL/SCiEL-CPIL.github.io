"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 130.0  # safety margin before 150s kill

def _evaluate(h):
    """Compute the value metric for a given height vector."""
    h = np.asarray(h, dtype=np.float64)
    g = np.convolve(h, h, mode='full')
    m = len(g)
    w = 1.0 / N
    y = np.zeros(m + 2)
    y[1:-1] = g
    l2 = (w / 3.0) * (np.sum(y[:m+1]**2) + np.sum(y[:m+1] * y[1:m+2]) + np.sum(y[1:m+2]**2))
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf < 1e-15:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _tent_basis(n_knots, knot_positions, amplitudes):
    """
    Build a nonnegative piecewise-linear signal from tent basis functions.
    """
    h = np.zeros(N)
    x = np.arange(N, dtype=np.float64)
    
    for k in range(n_knots):
        pos = knot_positions[k]
        amp = amplitudes[k]
        if amp <= 0:
            continue
        width = 2.0
        left = pos - width
        right = pos + width
        
        mask_left = (x >= left) & (x < pos)
        if np.any(mask_left):
            h[mask_left] += amp * (x[mask_left] - left) / width
        
        mask_right = (x >= pos) & (x <= right)
        if np.any(mask_right):
            h[mask_right] += amp * (right - x[mask_right]) / width
    
    h = np.maximum(h, 0.0)
    return h

def _tent_optimize(n_knots, initial_knots, initial_amplitudes, max_iter=200):
    """Optimize knot positions and amplitudes using Nelder-Mead."""
    x0 = np.concatenate([initial_knots, initial_amplitudes])
    
    def neg_value(params):
        knots = np.clip(params[:n_knots], 0, N - 1)
        amps = np.maximum(params[n_knots:], 0.0)
        h = _tent_basis(n_knots, knots, amps)
        return -_evaluate(h)
    
    try:
        res = minimize(neg_value, x0, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8})
        best_params = res.x
        best_knots = np.clip(best_params[:n_knots], 0, N - 1)
        best_amps = np.maximum(best_params[n_knots:], 0.0)
        best_h = _tent_basis(n_knots, best_knots, best_amps)
        best_val = _evaluate(best_h)
        return best_h, best_val
    except:
        h = _tent_basis(n_knots, np.clip(initial_knots, 0, N-1), np.maximum(initial_amplitudes, 0))
        return h, _evaluate(h)

def _polish_with_coordinate_ascent(h, max_iter=50):
    """Final polish using coordinate ascent on individual heights."""
    h = np.array(h, dtype=np.float64)
    best_val = _evaluate(h)
    
    for it in range(max_iter):
        improved = False
        for i in range(N):
            old = h[i]
            for delta in [-0.05, -0.01, 0.01, 0.05, -0.2, 0.2]:
                h[i] = max(0.0, old + delta)
                v = _evaluate(h)
                if v > best_val + 1e-10:
                    best_val = v
                    improved = True
            h[i] = 0.0
            v = _evaluate(h)
            if v > best_val + 1e-10:
                best_val = v
                improved = True
            else:
                h[i] = old
        if not improved:
            break
    
    return h, best_val

def construct():
    start_time = time.time()
    
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    knot_configs = [
        (3, np.array([10.0, 25.0, 40.0]), np.array([1.0, 1.0, 1.0])),
        (4, np.array([5.0, 18.0, 32.0, 45.0]), np.array([1.0, 1.0, 1.0, 1.0])),
        (5, np.array([3.0, 13.0, 25.0, 37.0, 47.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0])),
        (6, np.array([2.0, 10.0, 18.0, 32.0, 40.0, 48.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
        (3, np.array([8.0, 25.0, 35.0]), np.array([0.5, 1.0, 0.5])),
        (4, np.array([5.0, 20.0, 30.0, 45.0]), np.array([0.3, 1.0, 1.0, 0.3])),
        (5, np.array([2.0, 12.0, 25.0, 38.0, 48.0]), np.array([0.2, 0.6, 1.0, 0.6, 0.2])),
        (2, np.array([20.0, 30.0]), np.array([1.0, 1.0])),
        (3, np.array([22.0, 25.0, 28.0]), np.array([0.5, 1.0, 0.5])),
    ]
    
    for n_knots, init_knots, init_amps in knot_configs:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        h_tent, val_tent = _tent_optimize(n_knots, init_knots, init_amps, max_iter=150)
        
        if val_tent > best_val:
            best_val = val_tent
            best_h = h_tent.copy()
        
        if time.time() - start_time < DEADLINE - 15 and val_tent > best_val - 0.01:
            h_polished, val_polished = _polish_with_coordinate_ascent(h_tent, max_iter=30)
            if val_polished > best_val:
                best_val = val_polished
                best_h = h_polished.copy()
    
    if time.time() - start_time < DEADLINE - 5:
        h_final, val_final = _polish_with_coordinate_ascent(best_h, max_iter=40)
        if val_final > best_val:
            best_val = val_final
            best_h = h_final.copy()
    
    best_h = np.maximum(best_h, 0.0)
    best_h = np.where(np.isfinite(best_h), best_h, 0.0)
    
    return list(best_h)

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
