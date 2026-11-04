import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _compute_cross_corr(h):
    """
    Compute linear cross-correlation of h with (1-h) for all non-zero shifts.
    Returns array of length 2*(N-1).
    """
    g = 1.0 - h
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    G = np.fft.rfft(g, n=n_fft)
    c_full = np.fft.irfft(H * G, n=n_fft)
    c_pos = c_full[1:N]
    c_neg = c_full[n_fft-(N-1):n_fft]
    return np.concatenate([c_neg, c_pos])

def _get_max_overlap(h):
    """Return the maximum absolute overlap value."""
    corr = _compute_cross_corr(h)
    if len(corr) == 0:
        return 0.0
    return np.max(np.abs(corr))

def _project_to_feasible(h_raw):
    """
    Project h_raw to the set {h | 0 <= h_i <= 1, sum(h) = N/2}.
    Uses bisection on the Lagrange multiplier lambda for h_proj = clip(h_raw - lambda, 0, 1).
    """
    target_sum = N / 2.0
    h_clipped = np.clip(h_raw, 0.0, 1.0)
    current_sum = np.sum(h_clipped)
    if abs(current_sum - target_sum) < 1e-12:
        return h_clipped
    
    lo = -1.0 - np.max(h_raw)
    hi = 1.0 + np.max(h_raw)
    
    for _ in range(60):
        mid = (lo + hi) / 2.0
        h_mid = np.clip(h_raw - mid, 0.0, 1.0)
        s_mid = np.sum(h_mid)
        if abs(s_mid - target_sum) < 1e-13:
            break
        if s_mid > target_sum:
            lo = mid
        else:
            hi = mid
            
    lambda_star = (lo + hi) / 2.0
    h_proj = np.clip(h_raw - lambda_star, 0.0, 1.0)
    
    residual = target_sum - np.sum(h_proj)
    if abs(residual) > 1e-14:
        free_mask = (h_proj > 1e-10) & (h_proj < 1.0 - 1e-10)
        if np.any(free_mask):
            h_proj[free_mask] += residual / np.sum(free_mask)
            h_proj = np.clip(h_proj, 0.0, 1.0)
    
    return h_proj

def _objective_function(x, beta=100.0):
    """
    Objective: minimize t + penalty + smooth_max(|corr| - t)
    Using log-sum-exp for smooth max.
    """
    h = x[:N]
    t = x[N]
    
    sum_h = np.sum(h)
    penalty = 1000.0 * (sum_h - N/2.0)**2
    
    corr = _compute_cross_corr(h)
    if len(corr) == 0:
        return t + penalty
    
    # Use absolute value of overlaps
    abs_corr = np.abs(corr)
    
    # Smooth max of (abs_corr - t) using log-sum-exp
    # max(z) ≈ (1/beta) * log(sum(exp(beta*z)))
    z = beta * (abs_corr - t)
    z = np.clip(z, -50, 50)
    max_z = np.max(z)
    log_sum_exp = max_z + np.log(np.sum(np.exp(z - max_z)))
    smooth_max = log_sum_exp / beta
    
    return t + penalty + smooth_max

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    rng = np.random.default_rng(42)
    
    # Initial candidates
    candidates = []
    candidates.append(np.full(N, 0.5))
    
    h_alt = np.zeros(N)
    h_alt[::2] = 0.5 + 0.1
    h_alt[1::2] = 0.5 - 0.1
    candidates.append(h_alt)
    
    t = np.arange(N)
    for freq in range(1, 10):
        for phase in [0, np.pi/4, np.pi/2, np.pi]:
            for amp in [0.1, 0.2, 0.3, 0.4]:
                h_sin = 0.5 + amp * np.sin(2*np.pi*freq*t/N + phase)
                candidates.append(h_sin)
                
    for _ in range(20):
        candidates.append(rng.uniform(0, 1, N))
        
    best_h = np.full(N, 0.5)
    best_score = float('inf')
    
    for h_cand in candidates:
        h_proj = _project_to_feasible(h_cand)
        score = _get_max_overlap(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj
    
    bounds_full = [(0.0, 1.0)] * N + [(0.0, 2.0)]
    beta_schedule = [20, 50, 100, 200, 500]
    
    for beta in beta_schedule:
        if time.time() > deadline - 30.0:
            break
            
        x0 = np.concatenate([best_h, [best_score]])
        
        try:
            res = minimize(
                _objective_function, 
                x0, 
                args=(beta,),
                method='L-BFGS-B', 
                bounds=bounds_full,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            h_new = _project_to_feasible(res.x[:N])
            score_new = _get_max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
        
        if beta < 200:
            for _ in range(3):
                if time.time() > deadline - 30.0:
                    break
                h_rand = rng.uniform(0, 1, N)
                h_rand = _project_to_feasible(h_rand)
                x0 = np.concatenate([h_rand, [best_score]])
                try:
                    res = minimize(
                        _objective_function, 
                        x0, 
                        args=(beta,),
                        method='L-BFGS-B', 
                        bounds=bounds_full,
                        options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    h_new = _project_to_feasible(res.x[:N])
                    score_new = _get_max_overlap(h_new)
                    if score_new < best_score:
                        best_score = score_new
                        best_h = h_new
                except Exception:
                    pass
                    
    # Coordinate Descent with line search
    if time.time() < deadline - 15.0:
        for _ in range(30):
            if time.time() > deadline - 15.0:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 15.0:
                    break
                old_val = best_h[i]
                # Try several step sizes with line search
                for step in [0.02, -0.02, 0.01, -0.01, 0.005, -0.005, 0.002, -0.002]:
                    trial_h = best_h.copy()
                    trial_h[i] = np.clip(old_val + step, 0, 1)
                    trial_h = _project_to_feasible(trial_h)
                    score_trial = _get_max_overlap(trial_h)
                    if score_trial < best_score - 1e-13:
                        best_h = trial_h
                        best_score = score_trial
                        improved = True
                        break
    
    # Final high-precision polish using SLSQP on a reduced set of active constraints
    if time.time() < deadline - 10.0:
        # Identify which overlaps are near the maximum
        corr = _compute_cross_corr(best_h)
        abs_corr = np.abs(corr)
        max_val = best_score
        # Find indices where overlap is close to max
        active_mask = abs_corr > (max_val - 0.01)
        
        if np.any(active_mask):
            # Try a final optimization focusing on reducing the peak
            x0 = np.concatenate([best_h, [best_score]])
            try:
                res = minimize(
                    _objective_function, 
                    x0, 
                    args=(1000.0,),
                    method='L-BFGS-B', 
                    bounds=bounds_full,
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14}
                )
                h_new = _project_to_feasible(res.x[:N])
                score_new = _get_max_overlap(h_new)
                if score_new < best_score:
                    best_score = score_new
                    best_h = h_new
            except Exception:
                pass
    
    best_h = _project_to_feasible(best_h)
    return best_h
