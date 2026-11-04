import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _compute_linear_overlaps(h):
    """Compute linear cross-correlation of h with (1-h) for all non-zero shifts."""
    g = 1.0 - h
    c = np.correlate(h, g, mode='full')
    # c has length 2N-1, center at N-1
    # Exclude zero shift
    c_neg = c[:N-1]
    c_pos = c[N:]
    return np.concatenate([c_neg, c_pos])

def _max_overlap(h):
    """Return the maximum linear overlap value."""
    overlaps = _compute_linear_overlaps(h)
    if len(overlaps) == 0:
        return 0.0
    return np.max(overlaps)

def _project_to_feasible(h):
    """Project h onto feasible set: 0 <= h_i <= 1 and sum(h) = N/2."""
    h = np.clip(h, 0, 1).copy()
    target = N / 2.0
    current = np.sum(h)
    
    if abs(current - target) < 1e-14:
        return h
    
    # Bisection on Lagrange multiplier
    lo, hi = -1.0, 1.0
    
    def sum_at_lambda(lam):
        return np.sum(np.clip(h - lam, 0, 1))
    
    for _ in range(80):
        mid = (lo + hi) / 2
        s = sum_at_lambda(mid)
        if abs(s - target) < 1e-14:
            break
        if s > target:
            lo = mid
        else:
            hi = mid
    
    lam = (lo + hi) / 2
    h = np.clip(h - lam, 0, 1)
    
    # Final adjustment
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 1e-10) & (h < 1 - 1e-10)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
    
    return h

def _objective(x, beta, lam_sum):
    """Objective: LSE-smoothed max overlap + sum constraint penalty."""
    h = np.clip(x, 0, 1)
    overlaps = _compute_linear_overlaps(h)
    
    if len(overlaps) == 0:
        max_ov = 0.0
    else:
        max_ov = np.max(overlaps)
    
    # Smooth max via LSE
    if beta > 0 and len(overlaps) > 0:
        shifted = overlaps - max_ov
        shifted = np.clip(shifted, -50, 50)
        lse = max_ov + np.log(np.sum(np.exp(beta * shifted))) / beta
    else:
        lse = max_ov
    
    sum_penalty = lam_sum * (np.sum(h) - N / 2.0) ** 2
    return lse + sum_penalty

def _spectral_init(rng):
    """Generate candidates using spectral flatness design."""
    candidates = []
    t = np.arange(N)
    
    # Uniform baseline
    candidates.append(np.full(N, 0.5))
    
    # Alternating patterns
    for base in [0.3, 0.4, 0.45, 0.5]:
        h_alt = np.zeros(N)
        h_alt[::2] = base + 0.1
        h_alt[1::2] = base - 0.1
        candidates.append(h_alt.copy())
    
    # Multi-harmonic with spectral flatness
    for n_harm in range(1, 8):
        for total_amp in [0.2, 0.3, 0.4, 0.45, 0.49]:
            # Distribute amplitude across harmonics with some randomness
            amps = rng.dirichlet(np.ones(n_harm)) * total_amp
            phases = rng.uniform(0, 2*np.pi, size=n_harm)
            h = np.full(N, 0.5)
            for k in range(n_harm):
                freq = (k + 1)
                h += amps[k] * np.sin(2*np.pi*freq*t/N + phases[k])
            candidates.append(h.copy())
    
    # Pure sinusoids
    for freq in range(1, 20):
        for amp in [0.1, 0.2, 0.3, 0.4, 0.45]:
            for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                h = 0.5 + amp * np.sin(2*np.pi*freq*t/N + phase)
                candidates.append(h.copy())
    
    # Random
    for _ in range(50):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        candidates.append(h_rand)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    bounds = [(0.0, 1.0) for _ in range(N)]
    rng = np.random.default_rng(42)
    
    # Generate and evaluate initial candidates
    candidates = _spectral_init(rng)
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    for h_cand in candidates:
        h_proj = _project_to_feasible(h_cand)
        score = _max_overlap(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj
    
    # Multi-phase optimization with increasing beta
    beta_schedule = [3, 8, 20, 50, 100, 200, 400, 800]
    
    for beta in beta_schedule:
        if time.time() > deadline - 15.0:
            break
        
        # Optimize from best solution
        x0 = best_h.copy()
        
        def obj_func(x):
            return _objective(x, beta, 500.0)
        
        try:
            result = minimize(
                obj_func, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 800, 'ftol': 1e-13, 'gtol': 1e-11}
            )
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
        
        # Random restarts
        num_restarts = 6 if beta <= 50 else 3
        for _ in range(num_restarts):
            if time.time() > deadline - 15.0:
                break
            
            x0 = rng.uniform(0.1, 0.9, size=N)
            x0 = _project_to_feasible(x0)
            
            def obj_rand(x):
                return _objective(x, beta, 500.0)
            
            try:
                result = minimize(
                    obj_rand, x0, method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 400, 'ftol': 1e-11, 'gtol': 1e-9}
                )
                h_new = _project_to_feasible(result.x)
                score_new = _max_overlap(h_new)
                if score_new < best_score:
                    best_score = score_new
                    best_h = h_new
            except Exception:
                pass
    
    # Coordinate descent polish
    if time.time() < deadline - 5.0:
        improved = True
        max_cd = 80
        cd_iter = 0
        while improved and cd_iter < max_cd and time.time() < deadline - 5.0:
            improved = False
            cd_iter += 1
            for i in range(N):
                old_val = best_h[i]
                for delta in [0.02, -0.02, 0.01, -0.01, 0.005, -0.005, 0.002, -0.002]:
                    trial = best_h.copy()
                    trial[i] = np.clip(old_val + delta, 0, 1)
                    trial = _project_to_feasible(trial)
                    score = _max_overlap(trial)
                    if score < best_score - 1e-13:
                        best_h = trial
                        best_score = score
                        improved = True
                        break
    
    # Final high-precision polish
    if time.time() < deadline - 3.0:
        def obj_final(x):
            return _objective(x, 1500.0, 5000.0)
        
        try:
            result = minimize(
                obj_final, best_h, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            h_polished = _project_to_feasible(result.x)
            score_polished = _max_overlap(h_polished)
            if score_polished < best_score:
                best_score = score_polished
                best_h = h_polished
        except Exception:
            pass
    
    # Ensure validity
    best_h = _project_to_feasible(best_h)
    return best_h
