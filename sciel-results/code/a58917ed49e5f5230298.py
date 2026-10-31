"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_MARGIN = 10.0
TARGET_SUM = N / 2.0
TIME_LIMIT = 150.0

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
    """Adjust h so that sum(h) == target, preserving [0,1] bounds."""
    if target is None:
        target = TARGET_SUM
    h = np.asarray(h, dtype=float)
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
    """Compute soft max of x and its gradient using log-sum-exp."""
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

def _objective_and_grad(h, beta, penalty_weight=100.0):
    """
    Compute the smooth objective value and its gradient.
    """
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
        
    penalty_weight = 100.0
    sum_error = h.sum() - TARGET_SUM
    penalty_val = penalty_weight * sum_error * sum_error
    penalty_grad = 2.0 * penalty_weight * sum_error
    
    total_val = val_smooth + penalty_val
    total_grad = grad_h + penalty_grad
    
    return total_val, total_grad

def _optimize_direct(h0, deadline, beta_start=50.0, beta_end=500.0, max_iter=500):
    """
    Optimize h using L-BFGS-B with analytic gradients and adaptive beta.
    """
    h = h0.copy()
    h = _fix_sum(h)
    current_score = _score(h)
    
    stages = [
        (beta_start, max_iter // 2),
        (beta_end // 2, max_iter // 4),
        (beta_end, max_iter // 4)
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
                obj_func, 
                h, 
                jac=grad_func,
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
        except Exception as e:
            pass
            
    return current_score, h

def _local_search_polish(h, max_iters=200, deadline=None):
    """
    Deterministic local search to polish the solution.
    """
    h = h.copy()
    current_score = _score(h)
    
    if deadline is None:
        deadline = time.time() + 5.0
        
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        improved = False
        for i in range(N):
            for delta in [0.005, -0.005, 0.01, -0.01]:
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
            for i in range(N - 1):
                h_trial = h.copy()
                h_trial[i], h_trial[i+1] = h_trial[i+1], h_trial[i]
                h_trial = _fix_sum(h_trial)
                s = _score(h_trial)
                if s < current_score - 1e-12:
                    h = h_trial
                    current_score = s
                    improved = True
                    break
        if not improved:
            break
            
    return current_score, h

def _generate_initial_guesses():
    """Generate diverse initial guesses."""
    guesses = []
    
    # 1. Sinusoidal
    for A in [0.2, 0.3, 0.4]:
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
            
    # 3. Random
    np.random.seed(42)
    for _ in range(5):
        h = np.random.uniform(0.0, 1.0, N)
        h = _fix_sum(h)
        guesses.append(h)
        
    return guesses

def _optimize_epigraph(h0, deadline, max_iter=200):
    """
    Optimize using epigraph formulation:
    minimize t
    subject to: h_i * (1 - h_{i+1}) <= t for all i
    h_i in [0,1], sum(h) = N/2
    
    We use a penalty method on the constraints.
    """
    h = h0.copy()
    h = _fix_sum(h)
    t = _score(h)
    
    def epigraph_obj(params):
        h_vec = params[:-1]
        t_val = params[-1]
        
        h_vec = np.clip(h_vec, 0.0, 1.0)
        overlap = np.zeros(N)
        overlap[:-1] = h_vec[:-1] * (1.0 - h_vec[1:])
        overlap[-1] = h_vec[-1]
        
        # Penalty for constraint violations
        violations = np.maximum(0.0, overlap - t_val)
        penalty = 1000.0 * np.sum(violations**2)
        
        # Penalty for sum constraint
        sum_err = h_vec.sum() - TARGET_SUM
        sum_penalty = 100.0 * sum_err**2
        
        return t_val + penalty + sum_penalty
    
    def epigraph_grad(params):
        h_vec = params[:-1]
        t_val = params[-1]
        
        h_vec = np.clip(h_vec, 0.0, 1.0)
        overlap = np.zeros(N)
        overlap[:-1] = h_vec[:-1] * (1.0 - h_vec[1:])
        overlap[-1] = h_vec[-1]
        
        violations = np.maximum(0.0, overlap - t_val)
        
        # Gradient w.r.t t
        grad_t = 1.0 + 1000.0 * np.sum(violations)
        
        # Gradient w.r.t h
        grad_h = np.zeros(N)
        for i in range(N):
            if violations[i] > 0:
                # d(violation_i)/d(h_vec)
                if i < N - 1:
                    grad_h[i] += 1000.0 * 2.0 * violations[i] * (1.0 - h_vec[i+1])
                    grad_h[i+1] += 1000.0 * 2.0 * violations[i] * (-h_vec[i])
                else:
                    grad_h[i] += 1000.0 * 2.0 * violations[i] * 1.0
        
        # Sum penalty gradient
        sum_err = h_vec.sum() - TARGET_SUM
        grad_h += 2.0 * 100.0 * sum_err
        
        return np.concatenate([grad_h, [grad_t]])
    
    params = np.concatenate([h, [t]])
    bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)]
    
    try:
        res = minimize(
            epigraph_obj,
            params,
            jac=epigraph_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'iprint': 0}
        )
        h_new = res.x[:-1]
        h_new = _fix_sum(h_new)
        new_score = _score(h_new)
        if new_score < t:
            return new_score, h_new
    except Exception:
        pass
    
    return t, h

def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - DEADLINE_MARGIN)
    
    best_score = float('inf')
    best_h = None
    
    guesses = _generate_initial_guesses()
    
    scored_guesses = []
    for h in guesses:
        s = _score(h)
        scored_guesses.append((s, h))
        
    scored_guesses.sort(key=lambda x: x[0])
    
    num_starts_to_try = 8
    
    for i, (init_score, h_init) in enumerate(scored_guesses[:num_starts_to_try]):
        if time.time() > deadline - 15.0:
            break
            
        if init_score < best_score:
            best_score = init_score
            best_h = h_init.copy()
            
        # Run direct optimization
        s_opt, h_opt = _optimize_direct(h_init, deadline)
        
        if s_opt < best_score:
            best_score = s_opt
            best_h = h_opt.copy()
        
        # Run epigraph optimization
        if time.time() < deadline - 10.0:
            s_epi, h_epi = _optimize_epigraph(h_opt, deadline, max_iter=100)
            if s_epi < best_score:
                best_score = s_epi
                best_h = h_epi.copy()
            
    # Final polish
    if best_h is not None and time.time() < deadline - 2.0:
        s_polish, h_polish = _local_search_polish(best_h, max_iters=150, deadline=deadline - 1.0)
        if s_polish < best_score:
            best_score = s_polish
            best_h = h_polish.copy()
            
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = np.clip(best_h, 0.0, 1.0)
    best_h = _fix_sum(best_h)
    
    return best_h

if __name__ == "__main__":
    h = construct()
    print(f"Score: {_score(h)}")
    print(f"Sum: {np.sum(h)}")
