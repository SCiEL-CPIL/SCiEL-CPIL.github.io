"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize, fmin_l_bfgs_b

N = 100
TARGET_SUM = N / 2.0
DEADLINE_MARGIN = 15.0  # seconds to spare before 150s kill


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    if n == 0:
        return 0.0
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _overlap_vector(h):
    """Compute the overlap vector h[i]*(1-h[i+1])."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    return h * (1.0 - shifted)


def _fix_sum(h, target=None):
    """Fix sum of h to be exactly N/2, keeping elements in [0,1]."""
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


def _objective_and_grad(h, beta):
    """Compute smooth objective and its analytic gradient."""
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


def _optimize_adaptive_beta(h0, deadline):
    """Optimize h using L-BFGS-B with adaptive beta schedule."""
    h = h0.copy()
    h = _fix_sum(h)
    current_score = _score(h)
    
    stages = [
        (50.0, 100),
        (150.0, 80),
        (300.0, 60),
        (500.0, 40)
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
        except Exception:
            pass
            
    return current_score, h


def _local_search_polish(h, deadline):
    """Deterministic local search to polish the solution."""
    h = h.copy()
    current_score = _score(h)
    
    for _ in range(300):
        if time.time() > deadline:
            break
            
        improved = False
        for i in range(N):
            for delta in [0.002, -0.002, 0.005, -0.005, 0.01, -0.01]:
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
    
    # 1. 2-block theoretical (golden ratio)
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h1 = np.zeros(N)
    h1[:N//2] = A
    h1[N//2:] = B
    guesses.append(_fix_sum(h1))
    
    # 2. Sinusoidal patterns
    for k in [1, 2, 3]:
        for a in [0.4, 0.5, 0.6]:
            for b in [0.1, 0.2, 0.3]:
                for phase in [0.0, math.pi/4, math.pi/2]:
                    i = np.arange(N, dtype=float)
                    h = a + b * np.cos(2.0 * math.pi * k * i / N + phase)
                    h = np.clip(h, 0.0, 1.0)
                    h = _fix_sum(h)
                    guesses.append(h)
    
    # 3. Random
    np.random.seed(42)
    h_rand = np.random.uniform(0.3, 0.7, N)
    guesses.append(_fix_sum(h_rand))
    
    return guesses


def _optimize_2block(deadline):
    """Optimize 2-block construction."""
    n = N
    best_score = float('inf')
    best_h = None
    
    for k in range(10, n, 5):
        if time.time() > deadline - 5:
            break
        a_min = max(0.0, (n/2.0 - n + k) / k)
        a_max = min(1.0, n / (2.0 * k))
        if a_min >= a_max:
            continue
        
        for frac in [0.2, 0.4, 0.6, 0.8]:
            a_init = a_min + frac * (a_max - a_min)
            b_init = (n/2.0 - k * a_init) / (n - k)
            if b_init < 0 or b_init > 1:
                continue
            
            h = np.zeros(n)
            h[:k] = a_init
            h[k:] = b_init
            h = _fix_sum(h)
            
            s, h_opt = _optimize_adaptive_beta(h, deadline - 3)
            if s < best_score:
                best_score = s
                best_h = h_opt
    
    return best_h, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: 2-block optimization
    if time.time() < deadline - 40:
        h2, s2 = _optimize_2block(deadline - 40)
        if h2 is not None and s2 < best_score:
            best_score = s2
            best_h = h2
    
    # Phase 2: Full vector optimization from diverse starts
    inits = _generate_initial_guesses()
    for h_init in inits[:8]:  # Limit to 8 starts for time
        if time.time() > deadline - 20:
            break
        s, h_opt = _optimize_adaptive_beta(h_init, deadline - 15)
        if s < best_score:
            best_score = s
            best_h = h_opt
    
    # Phase 3: Local search polish
    if time.time() < deadline - 5 and best_h is not None:
        hp, sp = _local_search_polish(best_h, deadline - 2)
        if sp < best_score:
            best_score = sp
            best_h = hp
    
    # Fallback
    if best_h is None:
        A = (3.0 - math.sqrt(5.0)) / 2.0
        B = 1.0 - A
        h = np.zeros(N)
        h[:N//2] = A
        h[N//2:] = B
        best_h = _fix_sum(h)
    
    return best_h
