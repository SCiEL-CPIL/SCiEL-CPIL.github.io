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

def _smooth_max_grad(x, alpha=100.0):
    """Gradient of smooth max with respect to x."""
    x_max = np.max(x)
    exps = np.exp(alpha * (x - x_max))
    weights = exps / np.sum(exps)
    return weights

def _objective_smooth(params, alpha=200.0):
    """Objective with epigraph: minimize t + penalty for violation.
    params = [h_0, ..., h_{N-1}, t]
    """
    h = params[:N]
    t = params[N]
    shifted = np.zeros(N)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    smax = _smooth_max(overlap, alpha)
    penalty = 1000.0 * np.maximum(0.0, smax - t)**2
    return t + penalty

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
    for _ in range(10):
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
    A = (3.0 - math.sqrt(5.0)) / 2.0  # ~0.382
    B = 1.0 - A  # ~0.618
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    return h0

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s kill
    
    # Initial best from analytic 2-block solution
    h0 = _two_block_init()
    h_final = h0.copy()
    current_score = _score(h0)
    
    # Phase 1: Global Search using Differential Evolution
    # DE is robust to local minima and doesn't need gradients.
    # We optimize h in [0,1]^N, projecting to sum constraint in the objective.
    try:
        if time.time() < deadline - 20.0:
            def de_objective(h_params):
                # h_params is in [0, 1]^N
                h = _project_to_feasible(h_params)
                return _score(h)
            
            bounds = [(0.0, 1.0)] * N
            
            # Use a moderate population and iterations to balance exploration and time
            result_de = differential_evolution(
                de_objective,
                bounds,
                maxiter=80, 
                popsize=15, 
                tol=1e-10,
                seed=42,
                polish=False,
                init='latinhypercube',
                mutation=(0.5, 1.0),
                recombination=0.7
            )
            
            if result_de.fun < current_score:
                h_de = _project_to_feasible(result_de.x)
                score_de = _score(h_de)
                if score_de < current_score:
                    h_final = h_de.copy()
                    current_score = score_de
    except Exception:
        pass
    
    # Phase 2: Local Polish using SLSQP with smooth max objective
    try:
        if time.time() < deadline - 5.0:
            def objective_polish(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                smax = _smooth_max(overlap, alpha=500.0)
                return t + 1000.0 * np.maximum(0.0, smax - t)**2
            
            x0_polish = np.concatenate([h_final, [current_score + 0.01]])
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
            
            result2 = minimize(
                objective_polish,
                x0_polish,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
            )
            
            if result2.fun < 100.0:
                h_opt2 = result2.x[:N].copy()
                h_opt2 = _project_to_feasible(h_opt2)
                score2 = _score(h_opt2)
                if score2 < current_score:
                    h_final = h_opt2.copy()
                    current_score = score2
    except Exception:
        pass
    
    # Phase 3: Final fine-tuning with L-BFGS-B on the smooth objective
    try:
        if time.time() < deadline - 2.0:
            def objective_lbfgsb(h_flat):
                h = _project_to_feasible(h_flat)
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return _smooth_max(overlap, alpha=1000.0)
            
            result3 = minimize(
                objective_lbfgsb,
                h_final,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-8}
            )
            
            h_opt3 = _project_to_feasible(result3.x)
            score3 = _score(h_opt3)
            if score3 < current_score:
                h_final = h_opt3.copy()
                current_score = score3
    except Exception:
        pass
    
    # Final: ensure exact sum
    h_final = _project_to_feasible(h_final)
    
    return h_final
