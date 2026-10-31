"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_MARGIN = 15.0  # seconds to spare before 150s kill
TARGET_SUM = N / 2.0


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    This corresponds to the integral of h(x) * (1 - h(x-k)) over the domain,
    assuming h is zero outside [0, N].
    """
    n = len(h)
    cs = np.zeros(n)
    # k=0
    r0 = np.dot(h, h)
    cs[0] = (2.0 / n) * (n / 2.0 - r0)
    
    for k in range(1, n):
        # r_k = sum_{i=k}^{n-1} h[i] * h[i-k]
        # This is equivalent to np.dot(h[k:], h[:-k])
        r = np.dot(h[k:], h[:-k])
        cs[k] = (2.0 / n) * (n / 2.0 - r)
        
    return cs

def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))

def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective.
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    
    # C_k = 1 - (2/n) * r_k
    # dC_k/dh_j = -(2/n) * ( h_{j-k} (if j>=k) + h_{j+k} (if j+k<n) )
    
    for k in range(n):
        if w[k] == 0.0:
            continue
        factor = -(2.0 / n) * w[k]
        
        for j in range(n):
            g = 0.0
            # Term 1: h_{j-k} if j >= k
            if j >= k:
                g += h[j-k]
            # Term 2: h_{j+k} if j+k < n
            if j + k < n:
                g += h[j+k]
            grad[j] += factor * g
            
    return val, grad

def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    # Distribute the difference evenly, but clip
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    # If clipping changed the sum, do a second pass
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    # Final tiny correction on one element to ensure exact sum in floating point
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h

def _fast_gradient_ascent(h, deadline, beta=50.0, max_iter=2000):
    """
    Fast gradient ascent to minimize smooth max overlap.
    Uses a simple step size and line search for robustness.
    """
    h = h.copy()
    current_val = _score(h)
    
    # We want to minimize C_k, so we ascend the gradient of -C_k (or descend C_k)
    # _smooth_max_grad gives gradient of C_k. So we subtract the gradient.
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        val, grad = _smooth_max_grad(h, beta)
        
        # Gradient descent step
        # Simple adaptive step size
        step_size = 0.05
        h_trial = h - step_size * grad
        h_trial = _fix_sum(h_trial)
        
        new_val = _score(h_trial)
        
        if new_val < current_val:
            h = h_trial
            current_val = new_val
        else:
            # Halve step size
            step_size /= 2.0
            if step_size < 1e-6:
                break
                
    return current_val, h

def _multi_scale_search(h, deadline):
    """
    Try different step sizes in a local search to escape shallow local minima.
    """
    h = h.copy()
    current_val = _score(h)
    
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for ss in step_sizes:
        if time.time() > deadline:
            break
            
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
                
            # Randomly pick a few indices to perturb
            indices = np.random.choice(N, size=5, replace=False)
            for idx in indices:
                for delta in [ss, -ss]:
                    h_trial = h.copy()
                    h_trial[idx] += delta
                    if h_trial[idx] < 0 or h_trial[idx] > 1:
                        continue
                    h_trial = _fix_sum(h_trial)
                    new_val = _score(h_trial)
                    if new_val < current_val - 1e-12:
                        h = h_trial
                        current_val = new_val
                        improved = True
                        break
                if improved:
                    break
                    
    return current_val, h

def _optimize_sinusoidal_fast(deadline):
    """
    Fast optimization of sinusoidal parameters using coarse grid + local search.
    """
    best_score = float('inf')
    best_h = None

    # Coarse grid
    A_grid = np.linspace(0.05, 0.45, 20)
    phi_grid = np.linspace(0.0, 2.0 * np.pi, 30)

    best_A = 0.25
    best_phi = 0.0
    best_grid_score = float('inf')

    for A in A_grid:
        for phi in phi_grid:
            i = np.arange(N)
            h = 0.5 + A * np.sin(2.0 * np.pi * i / N + phi)
            h = np.clip(h, 0.0, 1.0)
            h = _fix_sum(h)
            s = _score(h)
            if s < best_grid_score:
                best_grid_score = s
                best_A = A
                best_phi = phi
                best_h = h.copy()
                
    # Local refinement using Nelder-Mead on 2 params (A, phi)
    def obj_sin(params):
        A, phi = params
        A = np.clip(A, 0.0, 0.5)
        i = np.arange(N)
        h = 0.5 + A * np.sin(2.0 * np.pi * i / N + phi)
        h = np.clip(h, 0.0, 1.0)
        h = _fix_sum(h)
        return _score(h)

    x0 = np.array([best_A, best_phi])
    try:
        res = minimize(obj_sin, x0, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
        if res.fun < best_grid_score:
            best_grid_score = res.fun
            best_A = float(res.x[0])
            best_phi = float(res.x[1])
            i = np.arange(N)
            h = 0.5 + best_A * np.sin(2.0 * np.pi * i / N + best_phi)
            h = np.clip(h, 0.0, 1.0)
            h = _fix_sum(h)
            best_h = h.copy()
    except Exception:
        pass

    return best_grid_score, best_h

def construct():
    start_time = time.time()
    deadline = start_time + (150.0 - DEADLINE_MARGIN)
    np.random.seed(42)

    best_score = float('inf')
    best_h = None

    # Phase 1: Analytical Start: Cosine Wave
    # h_i = 0.5 + 0.5 * cos(2 * pi * i / N)
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    s_cos = _score(h_cos)
    if s_cos < best_score:
        best_score = s_cos
        best_h = h_cos.copy()

    # Phase 2: Fast Gradient Ascent from Cosine
    if time.time() < deadline - 20:
        s_ga, h_ga = _fast_gradient_ascent(h_cos, deadline - 20, beta=50.0, max_iter=5000)
        if s_ga < best_score:
            best_score = s_ga
            best_h = h_ga.copy()

    # Phase 3: Sinusoidal Optimization (Grid + Nelder-Mead)
    if time.time() < deadline - 20:
        s_sin, h_sin = _optimize_sinusoidal_fast(deadline - 20)
        if s_sin < best_score:
            best_score = s_sin
            best_h = h_sin.copy()
            
        # Refine sinusoidal result with gradient ascent
        if h_sin is not None:
            s_sin_ga, h_sin_ga = _fast_gradient_ascent(h_sin, deadline - 10, beta=50.0, max_iter=2000)
            if s_sin_ga < best_score:
                best_score = s_sin_ga
                best_h = h_sin_ga.copy()

    # Phase 4: Multi-scale Local Search on Best Candidate
    if best_h is not None and time.time() < deadline - 10:
        s_ms, h_ms = _multi_scale_search(best_h, deadline - 10)
        if s_ms < best_score:
            best_score = s_ms
            best_h = h_ms.copy()

    # Phase 5: Final High-Precision Gradient Ascent
    if best_h is not None and time.time() < deadline - 5:
        s_final, h_final = _fast_gradient_ascent(best_h, deadline - 5, beta=100.0, max_iter=2000)
        if s_final < best_score:
            best_score = s_final
            best_h = h_final.copy()

    # Phase 6: Final Polish
    if best_h is not None and time.time() < deadline - 2:
        s_polish, h_polish = _multi_scale_search(best_h, deadline - 2)
        if s_polish < best_score:
            best_score = s_polish
            best_h = h_polish.copy()

    # Ensure valid output
    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = np.clip(best_h, 0.0, 1.0)
    best_h = _fix_sum(best_h)

    return best_h
