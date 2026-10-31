"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100
DEADLINE_MARGIN = 10.0  # seconds to spare before 150s kill


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]  # shifted[x] = h[x+1] for x in [0, n-2]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _make_step_profile(half, a):
    """
    Create a step profile: first 'half' elements are 'a', remaining are '1-a'.
    This maintains sum = N/2 exactly (in exact arithmetic).
    """
    h = np.zeros(N)
    h[:half] = a
    h[half:] = 1.0 - a
    return h


def _construct_analytic():
    """
    Theoretical optimum for 2-block structure:
    h=a on [0,1], h=b=1-a on [1,2]
    minimize max(a, (1-a)^2) subject to a+b=1
    a = (3 - sqrt(5))/2
    """
    A = (3.0 - math.sqrt(5.0)) / 2.0  # ~0.381966
    half = N // 2
    h = _make_step_profile(half, A)
    # Verify sum
    s = h.sum()
    diff = N / 2.0 - s
    if diff != 0.0:
        h[half - 1] += diff
    h = np.clip(h, 0.0, 1.0)
    return h


def _construct_sigmoid(w, c):
    """
    Construct h using a sigmoid transition.
    h[i] = 1 / (1 + exp(-(i - c) / w))
    """
    i = np.arange(N)
    if w < 1e-6:
        w = 1e-6
    h = 1.0 / (1.0 + np.exp(-(i - c) / w))
    return h


def _solve_c_for_sigmoid(w):
    """
    Find c such that sum(h) = N/2 for a given sigmoid width w.
    Uses bisection since sum is monotonically decreasing in c.
    """
    target_sum = N / 2.0
    lo = -10.0 * w
    hi = N + 10.0 * w
    
    for _ in range(60):
        mid = (lo + hi) / 2.0
        h = _construct_sigmoid(w, mid)
        s_mid = np.sum(h)
        if s_mid > target_sum:
            lo = mid
        else:
            hi = mid
            
    c_opt = (lo + hi) / 2.0
    return c_opt


def _construct_cosine():
    """
    Cosine wave construction:
    h_i = 0.5 + 0.5 * cos(2 * pi * i / N)
    """
    i = np.arange(N)
    h = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    
    # Correct for any floating point errors in the sum
    target_sum = N / 2.0
    current_sum = np.sum(h)
    diff = target_sum - current_sum
    
    if abs(diff) > 1e-12:
        # Distribute difference to interior points
        mask = (h > 1e-9) & (h < 1 - 1e-9)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
        else:
            h[-1] += diff
            
    h = np.clip(h, 0.0, 1.0)
    return h


def _refine_sum(h):
    """
    Ensure h sums exactly to N/2 and is within [0, 1].
    """
    target_sum = N / 2.0
    h = np.clip(h, 0.0, 1.0)
    
    for _ in range(100):
        current_sum = np.sum(h)
        diff = target_sum - current_sum
        if abs(diff) < 1e-14:
            break
            
        if diff > 0:
            # Need to increase sum
            mask = h < 1.0
            if not np.any(mask):
                break
            indices = np.where(mask)[0]
            # Add evenly, but cap at 1.0
            step = diff / len(indices)
            for idx in indices:
                h[idx] = min(1.0, h[idx] + step)
        else:
            # Need to decrease sum
            mask = h > 0.0
            if not np.any(mask):
                break
            indices = np.where(mask)[0]
            step = diff / len(indices)
            for idx in indices:
                h[idx] = max(0.0, h[idx] + step)
                
    return h


def _optimize_sigmoid_width(start_time, deadline):
    """
    Optimize the sigmoid width w to minimize max overlap.
    Uses grid search + local refinement.
    """
    best_score = float('inf')
    best_w = None
    best_c = None
    
    # Grid search over log(w)
    log_w_min = np.log(0.1)
    log_w_max = np.log(50.0)
    w_grid = np.linspace(log_w_min, log_w_max, 100)
    
    for lw in w_grid:
        if time.time() > deadline:
            break
            
        w = np.exp(lw)
        c = _solve_c_for_sigmoid(w)
        h = _construct_sigmoid(w, c)
        h = _refine_sum(h)
        score = _score(h)
        
        if score < best_score:
            best_score = score
            best_w = w
            best_c = c
            
    # Local refinement around best
    if best_w is not None:
        # Try a few finer points around best_w
        for factor in [0.5, 0.7, 1.0, 1.3, 2.0]:
            if time.time() > deadline:
                break
            w = best_w * factor
            c = _solve_c_for_sigmoid(w)
            h = _construct_sigmoid(w, c)
            h = _refine_sum(h)
            score = _score(h)
            if score < best_score:
                best_score = score
                best_w = w
                best_c = c
                
    if best_w is not None:
        h = _construct_sigmoid(best_w, best_c)
        h = _refine_sum(h)
        return h, best_score
    return None, float('inf')


def _optimize_step_position(start_time, deadline):
    """
    Optimize the step position for the 2-block structure.
    We fix the values a and 1-a, but vary where the step occurs.
    Actually, for a step function, the position doesn't change the overlap pattern
    if we're just shifting the whole block. But let's try optimizing 'a' slightly
    around the analytic optimum.
    """
    A = (3.0 - math.sqrt(5.0)) / 2.0
    half = N // 2
    best_score = float('inf')
    best_h = None
    
    # Try different step positions and values of 'a'
    for a_offset in [-0.02, -0.01, 0.0, 0.01, 0.02]:
        if time.time() > deadline:
            break
        a = A + a_offset
        if a < 0 or a > 1:
            continue
        for half_offset in [-2, -1, 0, 1, 2]:
            if time.time() > deadline:
                break
            h_half = half + half_offset
            if h_half < 1 or h_half > N - 1:
                continue
            h = _make_step_profile(h_half, a)
            h = _refine_sum(h)
            score = _score(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
                
    if best_h is not None:
        return best_h, best_score
    return None, float('inf')


def construct():
    start_time = time.time()
    deadline = start_time + (150.0 - DEADLINE_MARGIN)
    
    # 1. Analytic 2-block construction
    h_analytic = _construct_analytic()
    score_analytic = _score(h_analytic)
    
    # 2. Cosine construction
    h_cosine = _construct_cosine()
    score_cosine = _score(h_cosine)
    
    # 3. Sigmoid optimization
    h_sigmoid, score_sigmoid = _optimize_sigmoid_width(start_time, deadline)
    
    # 4. Step position optimization
    h_step, score_step = _optimize_step_position(start_time, deadline)
    
    # Select the best candidate
    candidates = [
        (score_analytic, h_analytic),
        (score_cosine, h_cosine),
    ]
    if h_sigmoid is not None:
        candidates.append((score_sigmoid, h_sigmoid))
    if h_step is not None:
        candidates.append((score_step, h_step))
    
    best_score, best_h = min(candidates, key=lambda x: x[0])
    
    # Final safety check
    best_h = _refine_sum(best_h)
    
    return best_h
