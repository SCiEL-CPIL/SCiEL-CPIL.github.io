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

def _construct_h_from_params(params):
    """
    Construct h from low-dimensional parameters.
    params = [h_low, h_high, center, width]
    h_low, h_high in [0, 1]
    center in [0, N-1]
    width in [1, N] (transition width)
    
    The profile is a sigmoidal transition from h_low to h_high.
    To ensure sum is exactly N/2, we will project afterwards.
    """
    h_low, h_high, center, width = params
    x = np.arange(N)
    # Sigmoid transition
    # t = (x - center) / width
    # h(x) = h_low + (h_high - h_low) * sigmoid(t)
    # Using a scaled sigmoid to keep values bounded and smooth
    t = (x - center) / width
    # Sigmoid: 1 / (1 + exp(-k*t)) with k controlling sharpness
    # We want a transition of width ~ 'width'. 
    # Let's use a polynomial or smoothstep for better control, or standard sigmoid.
    # Standard sigmoid with k=4 gives ~80% transition in 1 unit of t.
    # If width is the parameter, let's define t = (x - center) / (width/4) to make width meaningful.
    k = 4.0
    sigmoid = 1.0 / (1.0 + np.exp(-k * t))
    h = h_low + (h_high - h_low) * sigmoid
    
    # Clip to [0, 1]
    h = np.clip(h, 0.0, 1.0)
    return h

def _project_to_feasible(h):
    """Project h to satisfy sum=N/2 and 0<=h<=1 exactly."""
    target = N / 2.0
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    
    if abs(s - target) < 1e-12:
        return h

    # Iterative adjustment
    for _ in range(50):
        diff = target - h.sum()
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            # Need to increase values
            mask = h < 1.0
            if not mask.any():
                break
            h[mask] += diff / mask.sum()
        else:
            # Need to decrease values
            mask = h > 0.0
            if not mask.any():
                break
            h[mask] += diff / mask.sum()
        h = np.clip(h, 0.0, 1.0)
        
    # Final correction for floating point errors
    residual = target - h.sum()
    if abs(residual) > 1e-15:
        h += residual / N
        h = np.clip(h, 0.0, 1.0)
        
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Initial analytic 2-block solution
    A = (3.0 - math.sqrt(5.0)) / 2.0  # ~0.382
    B = 1.0 - A                        # ~0.618
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    h0 = _project_to_feasible(h0)
    current_score = _score(h0)
    h_final = h0.copy()

    # Phase 1: Global Search on Low-Dimensional Parameters
    # Parameters: h_low, h_high, center, width
    # h_low, h_high in [0, 1]
    # center in [0, N-1]
    # width in [2, N]
    
    def de_objective(params):
        h_raw = _construct_h_from_params(params)
        h_feasible = _project_to_feasible(h_raw)
        return _score(h_feasible)

    bounds_de = [
        (0.0, 1.0),  # h_low
        (0.0, 1.0),  # h_high
        (0.0, float(N - 1)),  # center
        (2.0, float(N))       # width
    ]
    
    try:
        if time.time() < deadline - 30.0:
            # Differential Evolution is robust for low-dimensional global search
            result_de = differential_evolution(
                de_objective,
                bounds_de,
                seed=42,
                maxiter=200,
                popsize=20,
                tol=1e-12,
                polish=False  # We will polish manually
            )
            
            if result_de.fun < current_score:
                h_candidate = _construct_h_from_params(result_de.x)
                h_candidate = _project_to_feasible(h_candidate)
                score_candidate = _score(h_candidate)
                if score_candidate < current_score:
                    h_final = h_candidate
                    current_score = score_candidate
    except Exception:
        pass

    # Phase 2: Local Polish using SLSQP on full h
    # This allows fine-tuning of individual steps to minimize the max overlap
    # and ensure exact constraint satisfaction.
    
    try:
        if time.time() < deadline - 10.0:
            def objective_slsqp(h):
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return float(np.max(overlap))

            constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
            bounds = [(0.0, 1.0)] * N

            # Start from the best h found
            h_start = h_final.copy()
            
            result_sl = minimize(
                objective_slsqp,
                h_start,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
            )

            if result_sl.fun < current_score:
                h_opt = result_sl.x.copy()
                h_opt = _project_to_feasible(h_opt)
                new_score = _score(h_opt)
                
                # Verify constraints and bounds
                if np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0) and abs(np.sum(h_opt) - N/2.0) < 1e-9:
                    if new_score < current_score:
                        h_final = h_opt
                        current_score = new_score
                        
    except Exception:
        pass

    # Phase 3: If time permits, try a few random restarts of the global search
    # with different seeds or initial points to escape any remaining local minima
    if time.time() < deadline - 15.0:
        for trial in range(3):
            if time.time() > deadline - 10.0:
                break
            try:
                # Perturb the best parameters found
                if hasattr(result_de, 'x'):
                    base_params = result_de.x
                else:
                    # Fallback to analytic guess parameters
                    base_params = [A, B, N/2.0, 10.0]
                
                # Add noise
                noise = np.random.normal(0, 0.1, size=4)
                new_params = base_params + noise
                # Clip to bounds
                new_params[0] = np.clip(new_params[0], 0, 1)
                new_params[1] = np.clip(new_params[1], 0, 1)
                new_params[2] = np.clip(new_params[2], 0, N-1)
                new_params[3] = np.clip(new_params[3], 2, N)
                
                # Local minimize on parameters? Or just evaluate?
                # Let's do a quick local minimize on the low-dim params
                def obj_params(p):
                    h_raw = _construct_h_from_params(p)
                    h_feasible = _project_to_feasible(h_raw)
                    return _score(h_feasible)
                
                res_local = minimize(
                    obj_params,
                    new_params,
                    method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-12, 'maxiter': 200}
                )
                
                if res_local.fun < current_score:
                    h_cand = _construct_h_from_params(res_local.x)
                    h_cand = _project_to_feasible(h_cand)
                    s_cand = _score(h_cand)
                    if s_cand < current_score:
                        h_final = h_cand
                        current_score = s_cand
                        
                        # Polish this new best
                        if time.time() < deadline - 5.0:
                            res_sl2 = minimize(
                                objective_slsqp,
                                h_final.copy(),
                                method='SLSQP',
                                bounds=bounds,
                                constraints=constraints,
                                options={'maxiter': 300, 'ftol': 1e-14, 'disp': False}
                            )
                            if res_sl2.fun < current_score:
                                h_opt2 = _project_to_feasible(res_sl2.x)
                                s_opt2 = _score(h_opt2)
                                if s_opt2 < current_score:
                                    h_final = h_opt2
                                    current_score = s_opt2
            except Exception:
                pass

    return h_final
