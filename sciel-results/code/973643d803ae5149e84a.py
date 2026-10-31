"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _fix_sum(h, target=None):
    """Fix the sum of h to exactly N/2 while keeping values in [0,1]."""
    if target is None:
        target = N / 2.0
    for _ in range(5):
        s = np.sum(h)
        diff = target - s
        if abs(diff) < 1e-15:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _smooth_max(x, alpha=200.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _make_2block_solution(A, B, split_frac):
    """Create a 2-block solution with values A and B split at split_frac."""
    h = np.zeros(N)
    split = int(round(split_frac * N))
    h[:split] = A
    h[split:] = B
    return _fix_sum(h)

def _make_transition_solution(A, B, split_frac, trans_width):
    """Create a solution with a smooth sigmoidal transition between A and B."""
    h = np.zeros(N)
    split = split_frac * N
    for i in range(N):
        x = (i - split) / trans_width
        # Sigmoid: 0.5 * (1 + tanh(x)) gives smooth transition from 0 to 1
        sig = 0.5 * (1.0 + np.tanh(x))
        h[i] = A + (B - A) * sig
    return _fix_sum(h)

def _score_from_params(params):
    """Score a parameterized solution.
    params: [A, B, split_frac, trans_width]
    """
    A, B, split_frac, trans_width = params
    if trans_width < 0.5:
        trans_width = 0.5
    h = _make_transition_solution(A, B, split_frac, trans_width)
    return _score(h)

def _score_and_h_from_params(params):
    """Score and return h for a parameterized solution."""
    A, B, split_frac, trans_width = params
    if trans_width < 0.5:
        trans_width = 0.5
    h = _make_transition_solution(A, B, split_frac, trans_width)
    return _score(h), h

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    # Golden ratio conjugate values
    A_opt = (3.0 - math.sqrt(5.0)) / 2.0  # ~0.382
    B_opt = 1.0 - A_opt  # ~0.618
    
    # Start with 2-block solution
    h0 = _make_2block_solution(A_opt, B_opt, 0.5)
    current_score = _score(h0)
    h_final = h0.copy()
    
    # Phase 1: Optimize low-dimensional parameters
    # params: [A, B, split_frac, trans_width]
    try:
        def objective_params(params):
            A, B, split_frac, trans_width = params
            # Penalty for invalid values
            if A < -0.1 or A > 1.1 or B < -0.1 or B > 1.1:
                return 1.0
            if trans_width < 0.1:
                return 1.0
            if split_frac < -0.1 or split_frac > 1.1:
                return 1.0
            score, _ = _score_and_h_from_params(params)
            return score
        
        # Bounds for parameters
        bounds_params = [
            (0.0, 1.0),      # A
            (0.0, 1.0),      # B
            (0.0, 1.0),      # split_frac
            (0.5, 20.0)      # trans_width
        ]
        
        # Try multiple starting points
        best_params = None
        best_score_params = current_score
        
        # Start from the analytic optimum
        initial_points = [
            [A_opt, B_opt, 0.5, 1.0],
            [A_opt, B_opt, 0.5, 2.0],
            [A_opt, B_opt, 0.5, 5.0],
            [A_opt, B_opt, 0.3, 1.0],
            [A_opt, B_opt, 0.7, 1.0],
            [0.3, 0.7, 0.5, 1.0],
            [0.4, 0.6, 0.5, 1.0],
        ]
        
        for x0 in initial_points:
            if time.time() > deadline - 30.0:
                break
            try:
                result = minimize(
                    objective_params,
                    x0,
                    method='SLSQP',
                    bounds=bounds_params,
                    options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
                )
                if result.fun < best_score_params:
                    best_score_params = result.fun
                    best_params = result.x.copy()
            except Exception:
                pass
        
        if best_params is not None:
            score, h_opt = _score_and_h_from_params(best_params)
            if score < current_score:
                h_final = h_opt.copy()
                current_score = score
    except Exception:
        pass
    
    # Phase 2: Full 100-variable SLSQP optimization from best parameterized solution
    try:
        if time.time() < deadline - 20.0:
            def objective_full(h):
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return float(np.max(overlap))
            
            constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
            bounds = [(0.0, 1.0)] * N
            
            result = minimize(
                objective_full,
                h_final.copy(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-14, 'disp': False}
            )
            
            if result.fun < current_score:
                h_opt = _fix_sum(result.x.copy())
                new_score = _score(h_opt)
                if new_score < current_score and np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
                    h_final = h_opt.copy()
                    current_score = new_score
    except Exception:
        pass
    
    # Phase 3: Smooth epigraph optimization
    try:
        if time.time() < deadline - 15.0:
            def objective_smooth(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                smax = _smooth_max(overlap, alpha=300.0)
                return t + 5000.0 * np.maximum(0.0, smax - t)**2
            
            x0 = np.concatenate([h_final, [current_score + 0.05]])
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
                objective_smooth,
                x0,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 1500, 'ftol': 1e-14, 'disp': False}
            )
            
            if result2.fun < 50.0:
                h_opt2 = _fix_sum(result2.x[:N].copy())
                new_score2 = _score(h_opt2)
                if new_score2 < current_score and np.all(h_opt2 >= 0.0) and np.all(h_opt2 <= 1.0):
                    h_final = h_opt2.copy()
                    current_score = new_score2
    except Exception:
        pass
    
    # Phase 4: High-precision polish with very tight alpha
    try:
        if time.time() < deadline - 10.0:
            def objective_polish(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                smax = _smooth_max(overlap, alpha=1000.0)
                return t + 50000.0 * np.maximum(0.0, smax - t)**2
            
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
            
            result3 = minimize(
                objective_polish,
                x0_polish,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 2000, 'ftol': 1e-16, 'disp': False}
            )
            
            if result3.fun < 50.0:
                h_opt3 = _fix_sum(result3.x[:N].copy())
                new_score3 = _score(h_opt3)
                if new_score3 < current_score and np.all(h_opt3 >= 0.0) and np.all(h_opt3 <= 1.0):
                    h_final = h_opt3.copy()
                    current_score = new_score3
    except Exception:
        pass
    
    # Phase 5: Local search - perturb and re-optimize
    try:
        if time.time() < deadline - 10.0:
            np.random.seed(123)
            for trial in range(8):
                if time.time() > deadline - 5.0:
                    break
                
                # Small perturbation of current best
                h_pert = h_final.copy()
                # Perturb a few random positions
                num_perturb = np.random.randint(2, 6)
                indices = np.random.choice(N, size=num_perturb, replace=False)
                h_pert[indices] += np.random.uniform(-0.05, 0.05, size=num_perturb)
                h_pert = np.clip(h_pert, 0.0, 1.0)
                h_pert = _fix_sum(h_pert)
                
                def objective_full(h):
                    shifted = np.zeros(N)
                    shifted[:-1] = h[1:]
                    overlap = h * (1.0 - shifted)
                    return float(np.max(overlap))
                
                constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
                bounds = [(0.0, 1.0)] * N
                
                result_r = minimize(
                    objective_full,
                    h_pert.copy(),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 300, 'ftol': 1e-14, 'disp': False}
                )
                
                if result_r.fun < current_score:
                    h_opt_r = _fix_sum(result_r.x.copy())
                    new_score_r = _score(h_opt_r)
                    if new_score_r < current_score and np.all(h_opt_r >= 0.0) and np.all(h_opt_r <= 1.0):
                        h_final = h_opt_r.copy()
                        current_score = new_score_r
    except Exception:
        pass
    
    # Final validation
    h_final = _fix_sum(h_final)
    h_final = np.clip(h_final, 0.0, 1.0)
    
    return h_final
