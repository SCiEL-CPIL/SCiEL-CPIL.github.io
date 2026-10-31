import numpy as np
import time
from scipy.optimize import minimize

N = 100
TARGET_SUM = N / 2.0

def _score(h):
    """Non-circular max overlap: h[i] * (1 - h[i+1]) for i<N-1, h[N-1]*1 for last."""
    n = len(h)
    if n == 0:
        return 0.0
    overlap = np.zeros(n)
    if n > 1:
        overlap[:-1] = h[:-1] * (1.0 - h[1:])
        overlap[-1] = h[-1]
    else:
        overlap[0] = h[0]
    return float(np.max(overlap))

def _fix_sum(h, target=None):
    """Adjust h so sum(h) == target, preserving [0,1] bounds."""
    if target is None:
        target = TARGET_SUM
    h = np.asarray(h, dtype=float).copy()
    diff = target - h.sum()
    if abs(diff) > 1e-12:
        h = h + diff / len(h)
        h = np.clip(h, 0.0, 1.0)
        diff2 = target - h.sum()
        if abs(diff2) > 1e-12:
            mask = (h > 1e-9) & (h < 1.0 - 1e-9)
            if np.any(mask):
                h[mask] += diff2 / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
                diff3 = target - h.sum()
                if abs(diff3) > 1e-9:
                    mask2 = (h > 0) & (h < 1.0)
                    if np.any(mask2):
                        h[mask2] += diff3 / np.sum(mask2)
                        h = np.clip(h, 0.0, 1.0)
    return h

def _smooth_max_and_grad(x, beta):
    """Soft max and its gradient via log-sum-exp."""
    x = np.asarray(x, dtype=float)
    max_x = np.max(x)
    if max_x < 1e-15:
        return 0.0, np.zeros_like(x)
    shifted = x - max_x
    weights = np.exp(beta * shifted)
    wsum = np.sum(weights)
    if wsum < 1e-300:
        return max_x, np.zeros_like(x)
    value = max_x + np.log(wsum) / beta
    grad = weights / wsum
    return value, grad

def _objective_and_grad(h, beta):
    """Smooth max of overlap + sum penalty."""
    n = len(h)
    h = np.clip(h, 0.0, 1.0)
    
    overlap = np.zeros(n)
    if n > 1:
        overlap[:-1] = h[:-1] * (1.0 - h[1:])
        overlap[-1] = h[-1]
    else:
        overlap[0] = h[0]
    
    val_smooth, grad_overlap = _smooth_max_and_grad(overlap, beta)
    
    grad_h = np.zeros(n)
    if n > 1:
        grad_h[:-1] += grad_overlap[:-1] * (1.0 - h[1:])
        grad_h[1:] += grad_overlap[:-1] * (-h[:-1])
        grad_h[-1] += grad_overlap[-1] * 1.0
    else:
        grad_h[0] += grad_overlap[0] * 1.0
    
    penalty_weight = 50.0
    sum_error = h.sum() - TARGET_SUM
    penalty_val = penalty_weight * sum_error * sum_error
    penalty_grad = 2.0 * penalty_weight * sum_error
    
    total_val = val_smooth + penalty_val
    total_grad = grad_h + penalty_grad
    return total_val, total_grad

def _optimize_direct(h0, deadline, beta_start=30.0, beta_end=300.0, max_iter=300):
    """L-BFGS-B with increasing beta."""
    h = h0.copy()
    h = _fix_sum(h)
    current_score = _score(h)
    
    stages = [
        (beta_start, max_iter // 2),
        (beta_end // 3, max_iter // 3),
        (beta_end, max_iter // 3),
    ]
    
    for beta, iters in stages:
        if time.time() > deadline - 2.0:
            break
        
        def obj_func(x):
            val, _ = _objective_and_grad(x, beta)
            return val
        
        def grad_func(x):
            _, grad = _objective_and_grad(x, beta)
            return grad
        
        try:
            res = minimize(
                obj_func, h, jac=grad_func,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': iters, 'ftol': 1e-14, 'gtol': 1e-10, 'iprint': 0}
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

def _local_search_polish(h, deadline, max_iters=100):
    """Greedy coordinate descent polish."""
    h = h.copy()
    current_score = _score(h)
    
    for _ in range(max_iters):
        if time.time() > deadline - 1.0:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline - 1.0:
                break
            for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                h_trial = h.copy()
                h_trial[i] += delta
                if h_trial[i] < 0 or h_trial[i] > 1:
                    continue
                h_trial = _fix_sum(h_trial)
                s = _score(h_trial)
                if s < current_score - 1e-12:
                    h = h_trial
                    current_score = s
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return current_score, h

def _generate_recurrence_h(a, b, c, h0, h1):
    """Generate h from linear recurrence h[i] = a*h[i-1] + b*h[i-2] + c."""
    h = np.zeros(N)
    h[0] = h0
    h[1] = h1
    for i in range(2, N):
        h[i] = a * h[i-1] + b * h[i-2] + c
    return h

def _optimize_recurrence(params, deadline):
    """Optimize recurrence coefficients (a, b, c, h0, h1)."""
    a, b, c, h0, h1 = params
    
    def rec_obj(p):
        a, b, c, h0, h1 = p
        h = _generate_recurrence_h(a, b, c, h0, h1)
        h = _fix_sum(h)
        return _score(h)
    
    # Use Nelder-Mead on the 5D parameter space
    try:
        res = minimize(
            rec_obj, params,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        best_params = res.x
        best_a, best_b, best_c, best_h0, best_h1 = best_params
        best_h = _generate_recurrence_h(best_a, best_b, best_c, best_h0, best_h1)
        best_h = _fix_sum(best_h)
        best_score = _score(best_h)
        return best_score, best_h
    except Exception:
        h = _generate_recurrence_h(a, b, c, h0, h1)
        h = _fix_sum(h)
        return _score(h), h

def _generate_initial_guesses():
    """Generate diverse initial guesses for direct optimization."""
    guesses = []
    
    # 1. Sinusoidal
    for A in [0.2, 0.3, 0.4, 0.5]:
        for phi in [0, np.pi/2, np.pi, 3*np.pi/2]:
            i = np.arange(N)
            h = 0.5 + A * np.sin(2.0 * np.pi * i / N + phi)
            h = np.clip(h, 0.0, 1.0)
            h = _fix_sum(h)
            guesses.append(h)
    
    # 2. Block patterns
    for k in [10, 20, 25, 30, 50]:
        for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
            h = np.zeros(N)
            n_high = int(k * frac)
            if n_high > 0:
                h[:n_high] = 1.0
            h = _fix_sum(h)
            guesses.append(h)
    
    # 3. Alternating high-low
    for period in [2, 3, 4, 5]:
        for offset in range(period):
            h = np.zeros(N)
            for i in range(N):
                if (i + offset) % period == 0:
                    h[i] = 1.0
            h = _fix_sum(h)
            guesses.append(h)
    
    # 4. Ramp
    for direction in [1, -1]:
        h = np.linspace(0, 1, N) if direction == 1 else np.linspace(1, 0, N)
        h = _fix_sum(h)
        guesses.append(h)
    
    # 5. Random
    np.random.seed(42)
    for _ in range(5):
        h = np.random.rand(N)
        h = _fix_sum(h)
        guesses.append(h)
    
    return guesses

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    # Phase 1: Direct optimization from diverse starts
    guesses = _generate_initial_guesses()
    
    for guess in guesses:
        if time.time() > deadline - 20.0:
            break
        s, h = _optimize_direct(guess, deadline)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Phase 2: Recurrence-based search
    if time.time() < deadline - 30.0:
        # Try several initial parameter sets for recurrence
        rec_starts = [
            [0.5, 0.0, 0.25, 0.5, 0.5],   # gentle decay
            [0.8, 0.1, 0.1, 0.5, 0.5],   # stronger persistence
            [0.3, 0.3, 0.3, 0.5, 0.5],   # more averaging
            [0.6, -0.1, 0.3, 0.5, 0.5],  # slight oscillation
            [0.9, 0.0, 0.05, 0.5, 0.5],  # near-constant
        ]
        
        for params in rec_starts:
            if time.time() > deadline - 15.0:
                break
            s, h = _optimize_recurrence(params, deadline)
            if s < best_score:
                best_score = s
                best_h = h.copy()
    
    # Phase 3: Final polish on best solution
    if time.time() < deadline - 5.0:
        s, h = _local_search_polish(best_h, deadline)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Ensure valid output
    best_h = _fix_sum(best_h)
    best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
