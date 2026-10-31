"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _project_to_feasible(h):
    """Project h to satisfy sum=N/2 and 0<=h<=1."""
    target = N / 2.0
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if s < 1e-10:
        h = np.full(N, 0.5)
        return h
    h = h * (target / s)
    h = np.clip(h, 0.0, 1.0)
    # Fix residual iteratively
    for _ in range(20):
        diff = target - h.sum()
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h < 1.0
            if mask.any():
                h[mask] += diff / mask.sum()
            else:
                break
        else:
            mask = h > 0.0
            if mask.any():
                h[mask] += diff / mask.sum()
            else:
                break
        h = np.clip(h, 0.0, 1.0)
    return h

def _two_block_init():
    """Analytic 2-block initial guess."""
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    return h0

def _objective_epigraph(params, alpha=200.0):
    """Objective with epigraph formulation."""
    h = params[:N]
    t = params[N]
    shifted = np.zeros(N)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    smax = _smooth_max(overlap, alpha)
    penalty = 1000.0 * np.maximum(0.0, smax - t)**2
    return t + penalty

def _slsqp_optimize(h_init, deadline, alpha=200.0, maxiter=1500):
    """Run SLSQP optimization from h_init, return best h found."""
    current_score = _score(h_init)
    
    try:
        x0 = np.concatenate([h_init, [current_score + 0.05]])
        bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]
        
        def constraint_sum(params):
            return np.sum(params[:N]) - N/2.0
        
        def constraint_epigraph(params):
            h = params[:N]
            t = params[N]
            shifted = np.zeros(N)
            shifted[:-1] = h[1:]
            overlap = h * (1.0 - shifted)
            return t - np.max(overlap)
        
        constraints_ext = [
            {'type': 'eq', 'fun': constraint_sum},
            {'type': 'ineq', 'fun': constraint_epigraph}
        ]
        
        result = minimize(
            lambda p: _objective_epigraph(p, alpha),
            x0,
            method='SLSQP',
            bounds=bounds_ext,
            constraints=constraints_ext,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'disp': False}
        )
        
        if result.fun < 100.0:
            h_opt = result.x[:N].copy()
            h_opt = _project_to_feasible(h_opt)
            new_score = _score(h_opt)
            if new_score < current_score and np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
                return h_opt, new_score
    except Exception:
        pass
    
    return h_init.copy(), current_score

def construct():
    deadline = time.time() + 140.0
    
    # Start with 2-block solution
    h0 = _two_block_init()
    h_final = h0.copy()
    current_score = _score(h0)
    
    # Phase 1: SLSQP from 2-block init
    if time.time() < deadline - 60.0:
        h_opt, sc_opt = _slsqp_optimize(h_final, deadline, alpha=200.0, maxiter=1500)
        if sc_opt < current_score:
            h_final = h_opt
            current_score = sc_opt
    
    # Phase 2: Differential Evolution global search
    if time.time() < deadline - 30.0:
        try:
            def de_objective(h_flat):
                h = _project_to_feasible(h_flat)
                return _score(h)
            
            bounds_de = [(0.0, 1.0)] * N
            
            result_de = differential_evolution(
                de_objective,
                bounds_de,
                seed=42,
                maxiter=30,
                popsize=15,
                tol=1e-8,
                polish=True,
                init='latinhypercube'
            )
            
            if result_de.fun < current_score:
                h_de = _project_to_feasible(result_de.x)
                sc_de = _score(h_de)
                if sc_de < current_score:
                    h_final = h_de
                    current_score = sc_de
        except Exception:
            pass
    
    # Phase 3: Polish with SLSQP from best found
    if time.time() < deadline - 15.0:
        h_opt, sc_opt = _slsqp_optimize(h_final, deadline, alpha=500.0, maxiter=2000)
        if sc_opt < current_score:
            h_final = h_opt
            current_score = sc_opt
    
    # Phase 4: Local random search refinement
    if time.time() < deadline - 5.0:
        rng = np.random.default_rng(42)
        h = h_final.copy()
        best_local = current_score
        
        for iteration in range(3000):
            if time.time() > deadline - 3.0:
                break
            
            step_size = max(0.001, 0.03 * (1.0 - iteration / 3000.0))
            
            h_trial = h.copy()
            n_perturb = rng.integers(1, 4)
            idxs = rng.choice(N, size=n_perturb, replace=False)
            for i in idxs:
                h_trial[i] = np.clip(h_trial[i] + rng.uniform(-step_size, step_size), 0.0, 1.0)
            
            h_trial = _project_to_feasible(h_trial)
            sc = _score(h_trial)
            
            if sc < best_local - 1e-12:
                best_local = sc
                h = h_trial.copy()
                h_final = h_trial.copy()
                current_score = sc
    
    # Final validation
    h_final = _project_to_feasible(h_final)
    
    if not (np.all(h_final >= -1e-8) and np.all(h_final <= 1.0 + 1e-8)):
        h_final = np.full(N, 0.5)
    
    return h_final
