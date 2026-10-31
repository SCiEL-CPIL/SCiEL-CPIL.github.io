"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
TARGET_SUM = N / 2.0


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    if n == 0:
        return 0.0
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _fix_sum(h, target=None):
    """Fix sum of h to be exactly N/2, keeping elements in [0,1]."""
    if target is None:
        target = TARGET_SUM
    h = np.asarray(h, dtype=float).copy()
    diff = target - h.sum()
    if abs(diff) > 1e-15:
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
        s = h.sum()
        diff2 = target - s
        if abs(diff2) > 1e-15:
            mask = (h > 1e-10) & (h < 1.0 - 1e-10)
            if np.any(mask):
                h[mask] += diff2 / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
                s = h.sum()
                diff3 = target - s
                if abs(diff3) > 1e-15:
                    mask2 = (h > 1e-12) & (h < 1.0 - 1e-12)
                    if np.any(mask2):
                        h[mask2] += diff3 / np.sum(mask2)
                        h = np.clip(h, 0.0, 1.0)
    return h


def _smooth_max_and_grad(x, beta):
    """Soft max using log-sum-exp, returning (value, gradient)."""
    x = np.asarray(x, dtype=float)
    max_x = np.max(x)
    if max_x < 1e-12:
        return 0.0, np.zeros_like(x)
    shifted = x - max_x
    weights = np.exp(beta * shifted)
    wsum = np.sum(weights)
    if wsum == 0:
        return max_x, np.zeros_like(x)
    value = max_x + math.log(wsum) / beta
    grad = weights / wsum
    return value, grad


def _overlap_vector(h):
    """Compute the overlap vector h[i]*(1-h[i+1])."""
    n = len(h)
    overlap = np.zeros(n)
    if n > 1:
        overlap[:-1] = h[:-1] * (1.0 - h[1:])
        overlap[-1] = h[-1]
    else:
        overlap[0] = h[0]
    return overlap


def _objective_and_grad(h, beta):
    """
    Compute the smooth objective value and its gradient.
    Objective: smooth_max(overlap) + penalty * (sum(h) - N/2)^2
    """
    n = len(h)
    h = np.clip(h, 0.0, 1.0)
    
    overlap = _overlap_vector(h)
    val_smooth, grad_overlap = _smooth_max_and_grad(overlap, beta)
    
    # Gradient of overlap w.r.t h
    grad_h = np.zeros(n)
    if n > 1:
        # o_i = h_i * (1 - h_{i+1}) for i < N-1
        # d(o_i)/d(h_i) = 1 - h_{i+1}
        # d(o_i)/d(h_{i+1}) = -h_i
        grad_h[:-1] += grad_overlap[:-1] * (1.0 - h[1:])
        grad_h[1:] += grad_overlap[:-1] * (-h[:-1])
        # o_{N-1} = h_{N-1}
        # d(o_{N-1})/d(h_{N-1}) = 1
        grad_h[-1] += grad_overlap[-1] * 1.0
    else:
        grad_h[0] += grad_overlap[0] * 1.0
    
    # Penalty term for sum constraint
    penalty_weight = 200.0
    sum_error = h.sum() - TARGET_SUM
    penalty_val = penalty_weight * sum_error * sum_error
    penalty_grad = 2.0 * penalty_weight * sum_error
    
    total_val = val_smooth + penalty_val
    total_grad = grad_h + penalty_grad
    
    return total_val, total_grad


def _adaptive_lbfgsb(h_init, deadline):
    """
    Optimize h using L-BFGS-B with adaptive beta (increasing sharpness).
    Start with soft max (easy to optimize), progressively sharpen.
    """
    n = N
    h = h_init.copy()
    h = _fix_sum(h)
    current_score = _score(h)
    
    # Stages with increasing beta
    stages = [
        (30.0, 200),
        (100.0, 150),
        (300.0, 100),
        (800.0, 50),
    ]
    
    for beta, max_iter in stages:
        if time.time() > deadline - 3.0:
            break
        
        def obj_func(x):
            val, _ = _objective_and_grad(x, beta)
            return val
        
        def grad_func(x):
            _, grad = _objective_and_grad(x, beta)
            return grad
        
        try:
            res = minimize(
                obj_func,
                h,
                jac=grad_func,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * n,
                options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'iprint': 0}
            )
            h_new = res.x
            h_new = _fix_sum(h_new)
            new_score = _score(h_new)
            if new_score < current_score:
                h = h_new
                current_score = new_score
        except Exception:
            pass
    
    return current_score, h


def _cosine_init(k=1, phase=0.0, amp=0.5):
    """Generate cosine wave initialization."""
    n = N
    i = np.arange(n, dtype=float)
    h = 0.5 + amp * np.cos(2.0 * math.pi * k * i / n + phase)
    h = np.clip(h, 0.0, 1.0)
    h = _fix_sum(h)
    return h


def _sinusoidal_init(k=1, phase=0.0, amp=0.3):
    """Generate sinusoidal initialization."""
    n = N
    i = np.arange(n, dtype=float)
    h = 0.5 + amp * np.sin(2.0 * math.pi * k * i / n + phase)
    h = np.clip(h, 0.0, 1.0)
    h = _fix_sum(h)
    return h


def _optimize_2block(k, deadline):
    """Optimize 2-block construction for given k."""
    n = N
    a_min = max(0.0, (n/2.0 - n + k) / k)
    a_max = min(1.0, n / (2.0 * k))
    if a_min >= a_max:
        return None, float('inf')
    
    # Coarse grid search
    n_samples = 500
    a_vals = np.linspace(a_min + 1e-10, a_max - 1e-10, n_samples)
    
    best_score = float('inf')
    best_a = None
    
    for a_val in a_vals:
        b_val = (n/2.0 - k * a_val) / (n - k)
        if b_val < -1e-12 or b_val > 1.0 + 1e-12:
            continue
        b_val = np.clip(b_val, 0.0, 1.0)
        # Max overlap for 2-block
        vals = []
        if k >= 2:
            vals.append(a_val * (1.0 - a_val))
        vals.append(a_val * (1.0 - b_val))
        if n - k >= 2:
            vals.append(b_val * (1.0 - b_val))
        vals.append(b_val)
        s = max(vals)
        if s < best_score:
            best_score = s
            best_a = a_val
    
    if best_a is None:
        return None, float('inf')
    
    # Fine grid refinement
    step = (a_max - a_min) / n_samples
    a_ref_min = max(a_min + 1e-10, best_a - 3*step)
    a_ref_max = min(a_max - 1e-10, best_a + 3*step)
    if a_ref_min < a_ref_max:
        a_ref_vals = np.linspace(a_ref_min, a_ref_max, 1000)
        for a_val in a_ref_vals:
            b_val = (n/2.0 - k * a_val) / (n - k)
            if b_val < -1e-12 or b_val > 1.0 + 1e-12:
                continue
            b_val = np.clip(b_val, 0.0, 1.0)
            vals = []
            if k >= 2:
                vals.append(a_val * (1.0 - a_val))
            vals.append(a_val * (1.0 - b_val))
            if n - k >= 2:
                vals.append(b_val * (1.0 - b_val))
            vals.append(b_val)
            s = max(vals)
            if s < best_score:
                best_score = s
                best_a = a_val
    
    # Scipy polish
    if time.time() < deadline - 2.0:
        def obj_2b(x):
            a = x[0]
            if a < 0 or a > 1:
                return 1.0
            b = (n/2.0 - k * a) / (n - k)
            if b < -1e-6 or b > 1.0 + 1e-6:
                return 1.0
            b = np.clip(b, 0.0, 1.0)
            vals = []
            if k >= 2:
                vals.append(a * (1.0 - a))
            vals.append(a * (1.0 - b))
            if n - k >= 2:
                vals.append(b * (1.0 - b))
            vals.append(b)
            return max(vals)
        
        for frac in [0.3, 0.5, 0.7]:
            a_init = a_min + frac * (a_max - a_min)
            try:
                res = minimize(obj_2b, np.array([a_init]), method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16})
                if res.fun < best_score and 0 <= res.x[0] <= 1:
                    a_opt = np.clip(res.x[0], 0, 1)
                    b_opt = (n/2.0 - k * a_opt) / (n - k)
                    b_opt = np.clip(b_opt, 0, 1)
                    vals = []
                    if k >= 2:
                        vals.append(a_opt * (1.0 - a_opt))
                    vals.append(a_opt * (1.0 - b_opt))
                    if n - k >= 2:
                        vals.append(b_opt * (1.0 - b_opt))
                    vals.append(b_opt)
                    s_exact = max(vals)
                    if s_exact < best_score:
                        best_score = s_exact
                        best_a = a_opt
            except Exception:
                pass
    
    if best_a is None:
        return None, float('inf')
    
    b_opt = (n/2.0 - k * best_a) / (n - k)
    b_opt = np.clip(b_opt, 0, 1)
    h = np.zeros(n)
    h[:k] = best_a
    h[k:] = b_opt
    h = _fix_sum(h)
    actual = _score(h)
    return h, actual


def _local_search_polish(h, deadline):
    """Coordinate descent local search with adaptive step size."""
    n = N
    h = h.copy()
    best_score = _score(h)
    step_size = 0.02
    improved = True
    no_improve_count = 0
    
    while improved and time.time() < deadline - 2:
        improved = False
        for i in range(n):
            for delta in [-step_size, -step_size/3, step_size/3, step_size]:
                h_trial = h.copy()
                h_trial[i] = np.clip(h_trial[i] + delta, 0.0, 1.0)
                h_trial = _fix_sum(h_trial)
                s = _score(h_trial)
                if s < best_score - 1e-14:
                    best_score = s
                    h = h_trial
                    improved = True
                    no_improve_count = 0
                    break
            if time.time() > deadline - 3:
                return h, best_score
        if not improved:
            no_improve_count += 1
            step_size *= 0.5
            if step_size < 1e-8 or no_improve_count > 10:
                break
    
    # Final tiny-step polish
    step_size = 1e-6
    for _ in range(3):
        if time.time() > deadline - 1:
            break
        for i in range(n):
            for delta in [-step_size, step_size]:
                h_trial = h.copy()
                h_trial[i] = np.clip(h_trial[i] + delta, 0.0, 1.0)
                h_trial = _fix_sum(h_trial)
                s = _score(h_trial)
                if s < best_score - 1e-15:
                    best_score = s
                    h = h_trial
                    break
            if time.time() > deadline - 1:
                break
    
    return h, best_score


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Optimize 2-block constructions for all k
    if time.time() < deadline - 60:
        for k in range(1, N):
            if time.time() > deadline - 60:
                break
            h2, s2 = _optimize_2block(k, deadline - 60)
            if h2 is not None and s2 < best_score:
                best_score = s2
                best_h = h2
    
    # Phase 2: Generate diverse initializations
    inits = []
    # Cosine waves
    inits.append(_cosine_init(1, 0.0, 0.5))
    inits.append(_cosine_init(1, math.pi/2, 0.5))
    inits.append(_cosine_init(2, 0.0, 0.4))
    # Sinusoidal waves
    inits.append(_sinusoidal_init(1, 0.0, 0.3))
    inits.append(_sinusoidal_init(1, math.pi/2, 0.3))
    inits.append(_sinusoidal_init(2, 0.0, 0.25))
    # Use best 2-block as init too
    if best_h is not None:
        inits.append(best_h.copy())
    
    # Phase 3: Adaptive L-BFGS-B optimization for each init
    for h_init in inits:
        if time.time() > deadline - 30:
            break
        h_opt, s_opt = _adaptive_lbfgsb(h_init, deadline - 30)
        if s_opt < best_score:
            best_score = s_opt
            best_h = h_opt
    
    # Phase 4: Local search polish
    if time.time() < deadline - 5 and best_h is not None:
        hp, sp = _local_search_polish(best_h, deadline - 2)
        if sp < best_score:
            best_score = sp
            best_h = hp
    
    # Fallback: analytical golden ratio 2-block
    if best_h is None:
        A = (3.0 - math.sqrt(5.0)) / 2.0
        B = 1.0 - A
        h = np.zeros(N)
        h[:N//2] = A
        h[N//2:] = B
        best_h = _fix_sum(h)
    
    return best_h
