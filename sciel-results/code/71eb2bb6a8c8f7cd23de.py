import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, linprog

N = 100
DEADLINE = 150.0
SAFETY = 5.0

def _correlation_value(h):
    """Compute the evaluator's value: max(corr) / N * 2."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _correlation_full(h):
    """Compute full correlation array for epigraph formulation."""
    return np.correlate(h, 1 - h, mode="full")

def _project_to_feasible(h):
    """Project h onto feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    for _ in range(200):
        diff = np.sum(h) - N / 2.0
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h > 1e-15
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
        else:
            mask = h < 1.0 - 1e-15
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    return h

def _epigraph_objective(x):
    """
    Epigraph formulation: x = [h (100 vars), t (1 var)]
    Minimize t subject to:
      corr_i <= t for all i
      sum(h) = N/2
      0 <= h_i <= 1
    """
    h = x[:N]
    t = x[N]
    corr = _correlation_full(h)
    # Penalty for constraint violations
    violations = np.maximum(corr - t, 0)
    penalty = 1e6 * np.sum(violations ** 2)
    # Also penalize sum constraint
    sum_pen = 1e4 * (np.sum(h) - N / 2.0) ** 2
    return t + penalty + sum_pen

def _epigraph_grad(x):
    """Gradient of epigraph objective (approximate)."""
    h = x[:N]
    t = x[N]
    corr = _correlation_full(h)
    # Gradient of t w.r.t. h_i
    # corr_k = sum_j h_j * (1-h_{j+k}) for various k
    # This is complex; use numerical gradient for now
    eps = 1e-7
    grad = np.zeros_like(x)
    for i in range(N):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad[i] = (_epigraph_objective(x_plus) - _epigraph_objective(x_minus)) / (2 * eps)
    # Gradient of t
    x_plus_t = x.copy()
    x_plus_t[N] += eps
    x_minus_t = x.copy()
    x_minus_t[N] -= eps
    grad[N] = (_epigraph_objective(x_plus_t) - _epigraph_objective(x_minus_t)) / (2 * eps)
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    rng = np.random.default_rng(100)
    
    # Phase 1: Differential Evolution on low-dimensional space (10 control points)
    # This allows for more effective global exploration than 100-dim DE.
    if time.time() < deadline - 60:
        def _low_dim_to_high_dim(params):
            """Interpolate 10 control points to 100 heights."""
            indices = np.linspace(0, 99, len(params))
            h = np.interp(np.arange(N), indices, params)
            return h

        def de_objective_low_dim(params):
            h = _low_dim_to_high_dim(params)
            h = _project_to_feasible(h)
            return _correlation_value(h)
        
        bounds_low_dim = [(0.0, 1.0)] * 10
        try:
            de_result = differential_evolution(
                de_objective_low_dim,
                bounds=bounds_low_dim,
                seed=100,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                polish=False,
                init='sobol',
                workers=1
            )
            if de_result.fun < best_val:
                h_de = _low_dim_to_high_dim(de_result.x)
                h_de = _project_to_feasible(h_de)
                val_de = _correlation_value(h_de)
                if val_de < best_val:
                    best_val = val_de
                    best_h = h_de.copy()
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B with epigraph formulation
    if time.time() < deadline - 30:
        # Initialize x = [h, t] where t is initial max correlation
        h_init = best_h.copy()
        t_init = np.max(_correlation_full(h_init))
        x_init = np.concatenate([h_init, [t_init]])
        
        bounds_epigraph = [(0.0, 1.0)] * N + [(0.0, 10.0)]
        
        try:
            res = minimize(
                _epigraph_objective,
                x_init,
                method='L-BFGS-B',
                bounds=bounds_epigraph,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-6, 'disp': False}
            )
            h_opt = _project_to_feasible(res.x[:N])
            val_opt = _correlation_value(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Phase 3: Soft-max based L-BFGS-B refinement with decreasing temperature
    if time.time() < deadline - 15:
        for T in [200.0, 100.0, 50.0, 20.0, 10.0]:
            if time.time() > deadline - 5:
                break
            
            def soft_max_obj(h_raw):
                h = _project_to_feasible(h_raw)
                corr = _correlation_full(h)
                x = corr / T
                x_max = np.max(x)
                shifted = x - x_max
                sum_exp = np.sum(np.exp(shifted))
                lse = x_max + np.log(sum_exp)
                return T * lse
            
            try:
                res = minimize(
                    soft_max_obj,
                    best_h,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-8, 'disp': False}
                )
                h_opt = _project_to_feasible(res.x)
                val_opt = _correlation_value(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # Phase 4: Fine-grained coordinate descent polish
    if time.time() < deadline - 5:
        improved = True
        max_rounds = 30
        for _ in range(max_rounds):
            if time.time() > deadline - 3:
                break
            if not improved:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline - 2:
                    break
                # Try a range of values around current
                current = best_h[i]
                trial_vals = np.clip(
                    current + np.array([-0.15, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.15]),
                    0.0, 1.0
                )
                for trial_val in trial_vals:
                    h_trial = best_h.copy()
                    h_trial[i] = trial_val
                    h_trial = _project_to_feasible(h_trial)
                    val_trial = _correlation_value(h_trial)
                    if val_trial < best_val - 1e-12:
                        best_val = val_trial
                        best_h = h_trial.copy()
                        improved = True
    
    # Phase 5: Final high-precision polish with SLSQP
    if time.time() < deadline - 2:
        def slsqp_obj(h):
            return _correlation_value(_project_to_feasible(h))
        
        try:
            res = minimize(
                slsqp_obj,
                best_h,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}],
                options={'maxiter': 50, 'ftol': 1e-12, 'disp': False}
            )
            h_opt = _project_to_feasible(res.x)
            val_opt = _correlation_value(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
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
    
    if not np.all(np.isfinite(best_h)):
        best_h = np.full(N, 0.5)
    
    return best_h
