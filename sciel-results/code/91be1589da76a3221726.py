import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _compute_overlaps(h):
    """Compute linear overlap of h with shifted (1-h) for all shifts."""
    g = 1.0 - h
    c = np.correlate(h, g, mode='full')
    # Exclude the zero shift (center)
    c1 = c[:N]      # negative shifts
    c2 = c[N+1:]    # positive shifts
    return np.concatenate([c1, c2])

def _max_overlap(h):
    """Return the maximum overlap value."""
    h = np.clip(h, 0, 1)
    overlaps = _compute_overlaps(h)
    return np.max(overlaps)

def _project_to_feasible(h):
    """Project h onto [0,1]^N with sum = N/2."""
    h = np.clip(h, 0, 1)
    target = N / 2.0
    current = np.sum(h)
    if abs(current - target) > 1e-10:
        for _ in range(20):
            diff = target - np.sum(h)
            if abs(diff) < 1e-12:
                break
            if diff > 0:
                room = 1.0 - h
                room[h >= 1.0] = 0
                total_room = room.sum()
                if total_room < 1e-15:
                    break
                h += room * (diff / total_room)
                h = np.clip(h, 0, 1)
            else:
                avail = h
                avail[h <= 0.0] = 0
                total_avail = avail.sum()
                if total_avail < 1e-15:
                    break
                h -= avail * (-diff / total_avail)
                h = np.clip(h, 0, 1)
    return h

def _objective_lse(x, beta, lam_sum):
    """Objective: minimize max overlap using LogSumExp approximation."""
    h = np.clip(x, 0, 1)
    overlaps = _compute_overlaps(h)
    
    max_ov = np.max(overlaps)
    if max_ov > 500:
        max_ov = 500
    lse = max_ov + np.log(np.sum(np.exp(beta * (overlaps - max_ov)))) / beta
    
    sum_penalty = lam_sum * (np.sum(h) - N / 2.0) ** 2
    return lse + sum_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Generate diverse initial candidates
    rng = np.random.default_rng(42)
    candidates = []
    
    # Uniform
    candidates.append(np.full(N, 0.5))
    
    # Alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 0.6
    h_alt[1::2] = 0.4
    candidates.append(h_alt)
    
    # Sinusoidal modulations
    t = np.arange(N)
    for freq in range(1, 15):
        for amp in [0.1, 0.2, 0.3, 0.4]:
            for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                trial = 0.5 + amp * np.sin(2*np.pi*freq*t/N + phase)
                candidates.append(trial.copy())
    
    # Cosine variants
    for freq in range(1, 15):
        for amp in [0.1, 0.2, 0.3, 0.4]:
            trial = 0.5 + amp * np.cos(2*np.pi*freq*t/N)
            candidates.append(trial.copy())
    
    # Sawtooth
    h_saw = np.linspace(0.1, 0.9, N)
    candidates.append(h_saw.copy())
    
    # Random restarts
    for i in range(30):
        h_rand = rng.uniform(0.1, 0.9, size=N)
        candidates.append(h_rand.copy())
    
    # Evaluate all candidates first
    for h_cand in candidates:
        h_proj = _project_to_feasible(h_cand)
        score = _max_overlap(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj
    
    # Optimization with multiple betas
    beta_schedule = [5, 10, 20, 50, 100]
    
    for beta in beta_schedule:
        if time.time() > deadline - 10.0:
            break
        
        # Start from best so far
        x0 = best_h.copy()
        
        def obj_func(x):
            return _objective_lse(x, beta, 100.0)
        
        try:
            result = minimize(
                obj_func,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
    
    # Try random restarts with optimization
    for i in range(15):
        if time.time() > deadline - 20.0:
            break
        
        x0 = rng.uniform(0.2, 0.8, size=N)
        x0 = _project_to_feasible(x0)
        
        beta = 20.0 + i * 5.0
        
        def obj_rand(x):
            return _objective_lse(x, beta, 100.0)
        
        try:
            result = minimize(
                obj_rand,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
    
    # Final polish with very high beta
    if time.time() < deadline - 5.0:
        def obj_polish(x):
            return _objective_lse(x, 200.0, 1000.0)
        
        try:
            result = minimize(
                obj_polish,
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            h_polished = _project_to_feasible(result.x)
            score_polished = _max_overlap(h_polished)
            if score_polished < best_score:
                best_score = score_polished
                best_h = h_polished
        except Exception:
            pass
    
    return best_h
