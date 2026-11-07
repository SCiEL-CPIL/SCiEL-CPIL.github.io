"""Erdos minimum overlap: Direct 100D optimization with smooth penalties and SA."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0  # Safety margin

def _score(h):
    """Compute the objective value: max correlation / N * 2."""
    if not np.all(np.isfinite(h)):
        return np.inf
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _is_valid(h):
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
        return False
    if abs(np.sum(h) - N / 2) > 1e-9 * N:
        return False
    return True

def _fix_sum(h):
    """Project h onto the affine hyperplane sum(h)=N/2 and box [0,1]."""
    h = h.copy()
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(100):
        diff = target - np.sum(h)
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            mask = h < 1.0 - 1e-9
            if not np.any(mask):
                break
            room = 1.0 - h[mask]
            total_room = np.sum(room)
            if total_room < 1e-9:
                break
            add_factor = min(1.0, diff / total_room)
            h[mask] += room * add_factor
        else:
            mask = h > 1e-9
            if not np.any(mask):
                break
            val = h[mask]
            total_val = np.sum(val)
            if total_val < 1e-9:
                break
            sub_factor = min(1.0, -diff / total_val)
            h[mask] -= val * sub_factor
    
    # Final check and correction
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        if diff > 0:
            idx = np.argmin(h)
            if h[idx] < 1.0:
                h[idx] += diff
        else:
            idx = np.argmax(h)
            if h[idx] > 0.0:
                h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h

def _objective_smooth(x, mu):
    """
    Smooth objective function with quadratic penalties for constraints.
    x: vector of length N (heights)
    mu: penalty parameter
    """
    h = x
    # Base objective
    corr = np.correlate(h, 1 - h, mode="full")
    val = np.max(corr) / N * 2
    
    # Penalty for sum constraint: sum(h) = N/2
    sum_err = np.sum(h) - N / 2.0
    pen_sum = mu * sum_err**2
    
    # Penalty for box constraints: 0 <= h <= 1
    # Smooth approximation: mu * (max(0, h-1)^2 + max(0, -h)^2)
    upper_viol = np.maximum(0, h - 1.0)
    lower_viol = np.maximum(0, -h)
    pen_box = mu * (np.sum(upper_viol**2) + np.sum(lower_viol**2))
    
    return val + pen_sum + pen_box

def construct():
    start_time = time.time()
    
    def deadline_check():
        return time.time() > start_time + DEADLINE
    
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    rng = np.random.RandomState(42)
    
    # Strategy: 
    # 1. Generate diverse initial candidates (random, sinusoidal, block)
    # 2. For each candidate, run a penalty method optimization with increasing mu
    # 3. Polish the best results with Simulated Annealing
    
    candidates = []
    
    # 1. Random uniform
    for _ in range(10):
        h_rand = rng.uniform(0, 1, N)
        candidates.append(_fix_sum(h_rand))
        
    # 2. Sinusoidal patterns
    for freq in [1, 2, 3, 5, 10]:
        for phase in [0, np.pi/2, np.pi]:
            x = np.arange(N)
            h_sin = 0.5 + 0.4 * np.sin(2 * np.pi * freq * x / N + phase)
            candidates.append(_fix_sum(h_sin))
            
    # 3. Block patterns (alternating high/low)
    for block_size in [2, 5, 10, 20, 25]:
        h_block = np.zeros(N)
        for i in range(N):
            h_block[i] = 1.0 if (i // block_size) % 2 == 0 else 0.0
        candidates.append(_fix_sum(h_block))
        
    # 4. Gradient patterns
    h_grad = np.linspace(0, 1, N)
    candidates.append(_fix_sum(h_grad))
    h_grad_rev = np.linspace(1, 0, N)
    candidates.append(_fix_sum(h_grad_rev))
    
    # Optimization loop
    mu_values = [10, 100, 1000, 10000]
    
    for cand in candidates:
        if deadline_check():
            break
            
        h_curr = cand.copy()
        
        # Penalty method: increase mu iteratively
        for mu in mu_values:
            if deadline_check():
                break
                
            try:
                res = minimize(
                    _objective_smooth,
                    h_curr,
                    args=(mu,),
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * N,
                    options={'maxiter': 50, 'ftol': 1e-10}
                )
                
                if res.success:
                    h_curr = res.x
                    # Project to ensure validity for next step
                    h_curr = _fix_sum(h_curr)
            except Exception:
                pass
        
        # Evaluate final candidate
        if _is_valid(h_curr):
            s_curr = _score(h_curr)
            if s_curr < best_score:
                best_score = s_curr
                best_h = h_curr.copy()
        
        # Simulated Annealing Polish
        if not deadline_check():
            h_sa = best_h.copy()
            current_score = _score(h_sa)
            T = 0.01
            step_size = 0.05
            
            for _ in range(2000):
                if deadline_check():
                    break
                    
                # Random move: change 1-3 elements
                num_changes = rng.randint(1, 4)
                indices = rng.choice(N, num_changes, replace=False)
                deltas = rng.uniform(-step_size, step_size, num_changes)
                
                h_new = h_sa.copy()
                for idx, delta in zip(indices, deltas):
                    h_new[idx] += delta
                    
                h_new = _fix_sum(h_new)
                
                if not _is_valid(h_new):
                    continue
                    
                s_new = _score(h_new)
                delta_s = s_new - current_score
                
                if delta_s < 0 or rng.random() < np.exp(-delta_s / T):
                    h_sa = h_new
                    current_score = s_new
                    if current_score < best_score:
                        best_score = current_score
                        best_h = h_sa.copy()
                
                # Cool down
                T *= 0.99
                if T < 1e-4:
                    break
                    
    # Final validation
    if not _is_valid(best_h):
        best_h = np.full(N, 0.5)
    
    return best_h
