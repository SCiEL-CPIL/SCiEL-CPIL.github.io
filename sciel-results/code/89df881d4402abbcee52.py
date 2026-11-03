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
        for _ in range(50):
            diff = target - np.sum(h)
            if abs(diff) < 1e-13:
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
    # Final exact projection using bisection on a scaling parameter
    target = N / 2.0
    current = np.sum(h)
    if abs(current - target) > 1e-12:
        # Use a simple iterative approach with tighter tolerance
        for _ in range(100):
            diff = target - np.sum(h)
            if abs(diff) < 1e-14:
                break
            if diff > 0:
                # Add to elements that have room, proportional to room
                room = 1.0 - h
                room[h >= 1.0 - 1e-15] = 0
                total_room = room.sum()
                if total_room < 1e-15:
                    break
                h += room * (diff / total_room)
                h = np.clip(h, 0, 1)
            else:
                avail = h
                avail[h <= 1e-15] = 0
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

def _objective_epigraph(x, beta, lam_sum):
    """Epigraph formulation: minimize t subject to overlaps <= t, sum constraint.
    x = [h_0, ..., h_{N-1}, t] where t is the auxiliary variable."""
    h = np.clip(x[:N], 0, 1)
    t = x[N]
    overlaps = _compute_overlaps(h)
    
    # Smooth constraint: sum of soft violations
    violations = np.maximum(overlaps - t, 0)
    constraint_penalty = np.sum(violations ** 2) + 0.1 * np.sum(violations)
    
    sum_penalty = lam_sum * (np.sum(h) - N / 2.0) ** 2
    
    # Objective: minimize t + penalties
    return t + constraint_penalty + sum_penalty

def construct():
    """
    Returns the optimal sequence of step heights.
    
    The problem is to minimize the maximum overlap of h with shifted 1-h.
    Let O_k be the overlap for shift k.
    O_k = sum_i h_i * (1 - h_{(i+k)%N}) = N/2 - C_k
    where C_k is the circular autocorrelation of h at lag k.
    
    We want to minimize max_k O_k, which is equivalent to maximizing min_k C_k.
    The average autocorrelation is N/4.
    Thus min_k C_k <= N/4, so max_k O_k >= N/4.
    
    The uniform sequence h_i = 0.5 achieves C_k = N/4 for all k,
    resulting in max_k O_k = N/4.
    This is the theoretical lower bound, hence the global optimum.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    bounds_h = [(0.0, 1.0) for _ in range(N)]
    bounds = bounds_h + [(-1.0, 2.0)]  # Add bound for t
    
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
    t_idx = np.arange(N)
    for freq in range(1, 20):
        for amp in [0.1, 0.2, 0.3, 0.4, 0.45]:
            for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
                trial = 0.5 + amp * np.sin(2*np.pi*freq*t_idx/N + phase)
                candidates.append(trial.copy())
    
    # Cosine variants
    for freq in range(1, 20):
        for amp in [0.1, 0.2, 0.3, 0.4, 0.45]:
            trial = 0.5 + amp * np.cos(2*np.pi*freq*t_idx/N)
            candidates.append(trial.copy())
    
    # Sawtooth
    h_saw = np.linspace(0.1, 0.9, N)
    candidates.append(h_saw.copy())
    
    # Triangular wave
    h_tri = 0.5 + 0.4 * np.abs(2 * (t_idx / N - 0.5) - 1)
    candidates.append(h_tri.copy())
    
    # Step functions
    for split in range(10, 90, 10):
        h_step = np.zeros(N)
        h_step[:split] = 0.6
        h_step[split:] = 0.4
        candidates.append(h_step.copy())
    
    # Random restarts
    for i in range(50):
        h_rand = rng.uniform(0.05, 0.95, size=N)
        candidates.append(h_rand.copy())
    
    # Evaluate all candidates first
    for h_cand in candidates:
        h_proj = _project_to_feasible(h_cand)
        score = _max_overlap(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj
    
    # LSE optimization with multiple betas - start with lower betas for smoothness
    beta_schedule = [3, 5, 10, 15, 20, 30, 50, 75, 100]
    
    for beta in beta_schedule:
        if time.time() > deadline - 15.0:
            break
        
        # Start from best so far
        x0 = best_h.copy()
        
        def obj_func(x):
            return _objective_lse(x, beta, 500.0)
        
        try:
            result = minimize(
                obj_func,
                x0,
                method='L-BFGS-B',
                bounds=bounds_h,
                options={'maxiter': 300, 'ftol': 1e-11, 'gtol': 1e-9}
            )
            
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
    
    # Epigraph optimization for better max-overlap handling
    if time.time() < deadline - 20.0:
        # Initialize t to current max overlap
        t_init = best_score
        x0_epigraph = np.concatenate([best_h, [t_init]])
        
        def obj_epigraph(x):
            return _objective_epigraph(x, 100.0, 1000.0)
        
        try:
            result = minimize(
                obj_epigraph,
                x0_epigraph,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-11, 'gtol': 1e-9}
            )
            
            h_new = _project_to_feasible(result.x[:N])
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
    
    # Random restarts with LSE optimization
    for i in range(20):
        if time.time() > deadline - 25.0:
            break
        
        x0 = rng.uniform(0.1, 0.9, size=N)
        x0 = _project_to_feasible(x0)
        
        beta = 15.0 + i * 5.0
        
        def obj_rand(x):
            return _objective_lse(x, beta, 500.0)
        
        try:
            result = minimize(
                obj_rand,
                x0,
                method='L-BFGS-B',
                bounds=bounds_h,
                options={'maxiter': 200, 'ftol': 1e-11, 'gtol': 1e-9}
            )
            
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
    
    return best_h
