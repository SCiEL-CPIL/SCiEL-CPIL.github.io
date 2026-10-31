"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import numpy as np
import time
import math
from scipy.optimize import minimize

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

def _objective_smooth_grad(params, alpha=200.0):
    """Objective with analytic gradient."""
    h = params[:N]
    t = params[N]
    shifted = np.zeros(N)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    smax = _smooth_max(overlap, alpha)
    penalty = 1000.0 * np.maximum(0.0, smax - t)**2
    val = t + penalty
    
    # Gradient
    grad = np.zeros(N + 1)
    # d/dt
    if smax > t:
        grad[N] = 1.0 - 2000.0 * (smax - t)
    else:
        grad[N] = 1.0
    
    # d/dh_i for i in 0..N-1
    # overlap_i = h_i * (1 - h_{i+1}) for i < N-1
    # overlap_{N-1} = h_{N-1} * (1 - 0) = h_{N-1}
    # d(overlap_i)/d(h_j):
    #   if j == i: (1 - h_{i+1})
    #   if j == i+1: -h_i
    #   else: 0
    # (for i < N-1; for i = N-1: d(overlap_{N-1})/d(h_{N-1}) = 1)
    
    w = _smooth_max_grad(overlap, alpha)
    # d(smax)/d(overlap_k) = w[k]
    
    # d(overlap_i)/d(h_j) matrix is sparse
    for i in range(N - 1):
        if w[i] > 1e-15:
            grad[i] += w[i] * (1.0 - h[i+1])
            grad[i+1] += w[i] * (-h[i])
    # i = N-1: overlap = h[N-1], d/dh[N-1] = 1
    if w[N-1] > 1e-15:
        grad[N-1] += w[N-1] * 1.0
    
    # Penalty gradient
    if smax > t:
        dp = 2000.0 * (smax - t)
        # d(penalty)/d(h_j) = dp * d(smax)/d(h_j)
        # d(smax)/d(h_j) already computed above as part of grad before penalty
        # Actually we need to recompute: grad_smax_h = the h-gradient of smax
        # Let me recompute cleanly
        grad_smax_h = np.zeros(N)
        for i in range(N - 1):
            if w[i] > 1e-15:
                grad_smax_h[i] += w[i] * (1.0 - h[i+1])
                grad_smax_h[i+1] += w[i] * (-h[i])
        if w[N-1] > 1e-15:
            grad_smax_h[N-1] += w[N-1]
        
        grad[:N] += dp * grad_smax_h
        grad[N] -= dp  # d(penalty)/dt = -2000*(smax-t) when smax>t
    
    return val, grad

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
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
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
            # popsize=20 means 20*N=2000 initial points.
            # maxiter=50 means 100k evaluations.
            # Each eval is ~100 ops. 10M ops is fast.
            # We can increase this if time permits.
            
            # To make it faster, we can use a smaller popsize initially or rely on the default.
            # SciPy DE default popsize is 15.
            
            result_de = minimize(
                de_objective,
                h0, # Initial guess (center of population)
                method='DifferentialEvolution',
                bounds=bounds,
                options={
                    'maxiter': 100, 
                    'popsize': 20, 
                    'tol': 1e-12,
                    'seed': 42,
                    'polish': False # We will polish manually to control time
                }
            )
            
            if result_de.fun < current_score:
                h_de = _project_to_feasible(result_de.x)
                score_de = _score(h_de)
                if score_de < current_score:
                    h_final = h_de.copy()
                    current_score = score_de
    except Exception:
        pass
    
    # Phase 2: Local Polish using SLSQP
    # Use the best result from DE (or initial) to start a precise local search.
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
                new_score2 = _score(h_opt2)
                if new_score2 < current_score:
                    h_final = h_opt2.copy()
                    current_score = new_score2
    except Exception:
        pass
    
    # Final validation
    h_final = _project_to_feasible(h_final)
    final_score = _score(h_final)
    
    if not (np.all(h_final >= -1e-8) and np.all(h_final <= 1.0 + 1e-8)):
        h_final = np.full(N, 0.5)
    
    return h_final
