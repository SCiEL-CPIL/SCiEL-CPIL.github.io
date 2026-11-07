import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 100
DEADLINE = 150.0
SAFETY = 5.0

def _correlation_value(h):
    """Compute the evaluator's value: max(corr) / N * 2."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _project_to_feasible(h):
    """Project h onto feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    for _ in range(100):
        diff = np.sum(h) - N / 2.0
        if abs(diff) < 1e-13:
            break
        if diff > 0:
            mask = h > 1e-14
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
        else:
            mask = h < 1.0 - 1e-14
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    return h

def _soft_max_objective(h, T):
    """Soft-max approximation of max correlation, scaled for stability."""
    corr = np.correlate(h, 1 - h, mode="full")
    x = corr / T
    x_max = np.max(x)
    shifted = x - x_max
    sum_exp = np.sum(np.exp(shifted))
    lse = x_max + np.log(sum_exp)
    return T * lse

def _soft_max_grad(h, T):
    """Gradient of soft-max objective w.r.t. h."""
    corr = np.correlate(h, 1 - h, mode="full")
    x = corr / T
    x_max = np.max(x)
    shifted = x - x_max
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    
    # Gradient of corr[k] w.r.t. h[i]
    # corr = correlate(h, 1-h, full)
    # corr[k] = sum_j h[j] * (1-h[j+k-N+1]) for appropriate ranges
    # d(corr[k])/d(h[i]) = (1-h[i+k-N+1]) - h[i] * delta... 
    # Actually, let's use the fact that corr = h * (1-h) cross-correlation
    # This is complex analytically, so use finite differences or skip gradient.
    # For L-BFGS-B, we can use numerical gradient.
    return None  # Let scipy handle numerical gradient

def _anneal_step(h, T, rng):
    """One simulated annealing step."""
    n_trials = 5
    best_h = h.copy()
    best_val = _correlation_value(h)
    
    for _ in range(n_trials):
        # Pick random indices to perturb
        n_perturb = rng.integers(1, 5)
        indices = rng.choice(N, size=n_perturb, replace=False)
        
        h_new = h.copy()
        for idx in indices:
            delta = rng.uniform(-0.1, 0.1)
            h_new[idx] = np.clip(h_new[idx] + delta, 0, 1)
        
        h_new = _project_to_feasible(h_new)
        val_new = _correlation_value(h_new)
        
        if val_new < best_val or rng.random() < np.exp(-(val_new - best_val) / T):
            best_h = h_new
            best_val = val_new
    
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    rng = np.random.default_rng(100)
    
    # Generate diverse initial guesses
    initial_guesses = [np.full(N, 0.5)]
    
    # Alternating patterns
    for period in [2, 3, 4, 5, 7, 10]:
        h_pat = np.zeros(N)
        for i in range(N):
            phase = (i % period) / period
            h_pat[i] = 0.5 + 0.45 * np.cos(2 * np.pi * phase)
        initial_guesses.append(_project_to_feasible(h_pat))
    
    # Block patterns
    for block_size in [2, 3, 4, 5, 7, 10, 14, 20, 25]:
        h_block = np.zeros(N)
        for i in range(N):
            if (i // block_size) % 2 == 0:
                h_block[i] = 1.0
        initial_guesses.append(_project_to_feasible(h_block))
    
    # Sine waves
    for f in [1, 2, 3, 4, 5, 7, 10, 14, 20]:
        h_sin = 0.5 + 0.45 * np.sin(2 * np.pi * f * np.arange(N) / N)
        initial_guesses.append(_project_to_feasible(h_sin))
    
    # Random
    for _ in range(10):
        h_rand = rng.uniform(0, 1, N)
        initial_guesses.append(_project_to_feasible(h_rand))
    
    # Evaluate all initial guesses
    initial_vals = [_correlation_value(h) for h in initial_guesses]
    sorted_indices = np.argsort(initial_vals)
    
    # Update best
    for idx in sorted_indices[:3]:
        if initial_vals[idx] < best_val:
            best_val = initial_vals[idx]
            best_h = initial_guesses[idx].copy()
    
    # Phase 1: Differential Evolution (global search)
    if time.time() < deadline - 30:
        def de_objective(x):
            x = _project_to_feasible(x)
            return _correlation_value(x)
        
        bounds = [(0.0, 1.0)] * N
        try:
            de_result = differential_evolution(
                de_objective,
                bounds=bounds,
                seed=100,
                maxiter=30,
                popsize=15,
                tol=1e-8,
                polish=False,
                init='sobol'
            )
            if de_result.success or de_result.fun < best_val:
                h_de = _project_to_feasible(de_result.x)
                val_de = _correlation_value(h_de)
                if val_de < best_val:
                    best_val = val_de
                    best_h = h_de.copy()
        except Exception:
            pass
    
    # Phase 2: Simulated Annealing from best
    if time.time() < deadline - 20:
        h_sa = best_h.copy()
        T_start = 0.5
        T_end = 0.01
        n_sa_steps = 200
        
        for step in range(n_sa_steps):
            if time.time() > deadline - 5:
                break
            T = T_start * (T_end / T_start) ** (step / n_sa_steps)
            h_sa, val_sa = _anneal_step(h_sa, T, rng)
            if val_sa < best_val:
                best_val = val_sa
                best_h = h_sa.copy()
    
    # Phase 3: L-BFGS-B with soft-max objective
    # Use a parameterization: h = sigmoid(z) * scale, then project
    # Actually, let's just use L-BFGS-B on the soft-max with the constraint handled by projection
    if time.time() < deadline - 10:
        for T in [50.0, 100.0, 200.0, 500.0]:
            if time.time() > deadline - 5:
                break
            
            def lbfgs_obj(x):
                x = _project_to_feasible(x)
                return _soft_max_objective(x, T)
            
            try:
                res = minimize(
                    lbfgs_obj,
                    best_h,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-8, 'disp': False}
                )
                if res.fun < _soft_max_objective(best_h, T):
                    h_opt = _project_to_feasible(res.x)
                    val_opt = _correlation_value(h_opt)
                    if val_opt < best_val:
                        best_val = val_opt
                        best_h = h_opt.copy()
            except Exception:
                pass
    
    # Phase 4: Fine-tuning with coordinate descent
    if time.time() < deadline - 3:
        improved = True
        max_rounds = 50
        for _ in range(max_rounds):
            if time.time() > deadline - 2:
                break
            if not improved:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 2:
                    break
                h_trial = best_h.copy()
                # Try moving h[i] to optimize
                # Binary search for optimal h[i] given others fixed
                # The correlation depends on h[i] in a specific way
                # Just try a few values
                for trial_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    h_trial[i] = trial_val
                    h_trial = _project_to_feasible(h_trial)
                    val_trial = _correlation_value(h_trial)
                    if val_trial < best_val - 1e-12:
                        best_val = val_trial
                        best_h = h_trial.copy()
                        improved = True
    
    # Final projection and validation
    best_h = _project_to_feasible(best_h)
    diff = np.sum(best_h) - N / 2.0
    if abs(diff) > 1e-9:
        mask = (best_h > 1e-8) & (best_h < 1.0 - 1e-8)
        if np.any(mask):
            best_h[mask] -= diff / np.sum(mask)
        else:
            best_h -= diff / N
        best_h = np.clip(best_h, 0, 1)
    
    # Final check
    if not np.all(np.isfinite(best_h)):
        best_h = np.full(N, 0.5)
    
    return best_h
