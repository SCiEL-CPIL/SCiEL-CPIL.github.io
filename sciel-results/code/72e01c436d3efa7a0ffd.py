import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _compute_overlaps(h):
    """Compute linear overlap of h with shifted (1-h) for all shifts."""
    g = 1.0 - h
    c = np.correlate(h, g, mode='full')
    c1 = c[:N]
    c2 = c[N+1:]
    return np.concatenate([c1, c2])

def _max_overlap(h):
    """Return the maximum overlap value."""
    h = np.clip(h, 0, 1)
    overlaps = _compute_overlaps(h)
    return np.max(overlaps)

def _project_to_feasible(h):
    """Project h onto [0,1]^N with sum = N/2 using bisection on Lagrange multiplier."""
    h = np.clip(h, 0, 1)
    target = N / 2.0
    
    # If sum is already correct, return
    if abs(np.sum(h) - target) < 1e-10:
        return h

    # Bisection on lambda
    # h_i = clip(x_i - lambda, 0, 1)
    # We want sum(clip(h - lambda, 0, 1)) = target
    
    # Bounds for lambda
    # If lambda is very negative, h_i -> 1, sum -> N
    # If lambda is very positive, h_i -> 0, sum -> 0
    # We need sum = N/2
    
    low = -1.0
    high = 2.0
    
    for _ in range(50):
        mid = (low + high) / 2.0
        h_proj = np.clip(h - mid, 0.0, 1.0)
        s = np.sum(h_proj)
        if s > target:
            low = mid
        else:
            high = mid
    
    h_proj = np.clip(h - (low + high) / 2.0, 0.0, 1.0)
    
    # Final correction to ensure exact sum (floating point)
    diff = target - np.sum(h_proj)
    if abs(diff) > 1e-12:
        # Distribute diff to elements not at bounds
        mask = (h_proj > 1e-9) & (h_proj < 1.0 - 1e-9)
        if np.sum(mask) > 0:
            h_proj[mask] += diff / np.sum(mask)
            h_proj = np.clip(h_proj, 0, 1)
            # Re-project if clipping changed sum
            # Simple loop
            for _ in range(10):
                diff = target - np.sum(h_proj)
                if abs(diff) < 1e-14: break
                mask = (h_proj > 1e-9) & (h_proj < 1.0 - 1e-9)
                if np.sum(mask) == 0: break
                h_proj[mask] += diff / np.sum(mask)
                h_proj = np.clip(h_proj, 0, 1)
                
    return h_proj

def _objective_lse(x, beta, lam_sum):
    """Objective: minimize max overlap using LogSumExp approximation."""
    h = np.clip(x, 0, 1)
    overlaps = _compute_overlaps(h)
    max_ov = np.max(overlaps)
    if max_ov > 500:
        max_ov = 500
    # Add small epsilon to avoid log(0) issues if overlaps are very negative (though they are non-negative here)
    # overlaps are sums of products of [0,1], so they are >= 0
    lse = max_ov + np.log(np.sum(np.exp(beta * (overlaps - max_ov)))) / beta
    sum_penalty = lam_sum * (np.sum(h) - N / 2.0) ** 2
    return lse + sum_penalty

def _generate_m_sequence(n):
    """Generate a maximal length sequence (M-sequence) of length 2^n - 1."""
    # Simple LFSR implementation
    # For N=100, we can use n=7 (length 127) and truncate
    n = 7
    length = (1 << n) - 1
    # Feedback polynomial for maximal length: x^7 + x^6 + 1 (or similar)
    # Let's use a known one: 0b1000011 (135) for n=7?
    # Actually, simpler to just use random binary if M-seq generation is complex
    # But M-seqs have good autocorrelation.
    # Let's use a standard maximal length LFSR
    # State is 7 bits
    state = 0x7F # All 1s
    seq = []
    for _ in range(length):
        seq.append(state & 1)
        feedback = (state & 1) ^ ((state >> 6) & 1) # x^7 + x^6 + 1
        state = (state >> 1) | (feedback << 6)
    return np.array(seq)

def construct():
    """
    Returns the optimal sequence of step heights.
    """
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    bounds_h = [(0.0, 1.0) for _ in range(N)]
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    rng = np.random.default_rng(42)
    candidates = []
    
    # 1. Uniform
    candidates.append(np.full(N, 0.5))
    
    # 2. M-Sequence based
    m_seq = _generate_m_sequence(7)
    # Truncate or pad to N
    m_seq_N = np.tile(m_seq, int(np.ceil(N / len(m_seq))))[:N]
    # Map 0/1 to 0/1
    h_m = m_seq_N.astype(float)
    candidates.append(h_m)
    
    # 3. Random Binary
    for _ in range(10):
        h_rand_bin = rng.integers(0, 2, size=N).astype(float)
        candidates.append(h_rand_bin)
        
    # 4. Sinusoidal
    t_idx = np.arange(N)
    for freq in [1, 2, 3, 5, 10]:
        for amp in [0.2, 0.4, 0.49]:
            trial = 0.5 + amp * np.sin(2*np.pi*freq*t_idx/N)
            candidates.append(trial.copy())
            
    # 5. Random Continuous
    for _ in range(20):
        h_rand = rng.uniform(0, 1, size=N)
        candidates.append(h_rand)
        
    # Evaluate and project all candidates
    for h_cand in candidates:
        h_proj = _project_to_feasible(h_cand)
        score = _max_overlap(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj
            
    # Optimization Loop
    # Use a schedule of betas for LSE
    beta_schedule = [5, 10, 20, 50, 100, 200]
    
    for beta in beta_schedule:
        if time.time() > deadline - 15.0:
            break
        
        x0 = best_h.copy()
        
        def obj_func(x):
            return _objective_lse(x, beta, 500.0)
        
        try:
            result = minimize(
                obj_func,
                x0,
                method='L-BFGS-B',
                bounds=bounds_h,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
            
    # Random Restarts with Optimization
    for i in range(10):
        if time.time() > deadline - 20.0:
            break
            
        # Generate new random start
        x0 = rng.uniform(0, 1, size=N)
        x0 = _project_to_feasible(x0)
        
        beta = 50.0
        
        def obj_rand(x):
            return _objective_lse(x, beta, 500.0)
            
        try:
            result = minimize(
                obj_rand,
                x0,
                method='L-BFGS-B',
                bounds=bounds_h,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
            
    # Final Polish
    if time.time() < deadline - 10.0:
        beta = 500.0
        x0 = best_h.copy()
        
        def obj_polish(x):
            return _objective_lse(x, beta, 1000.0)
            
        try:
            result = minimize(
                obj_polish,
                x0,
                method='L-BFGS-B',
                bounds=bounds_h,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
            
    return best_h
