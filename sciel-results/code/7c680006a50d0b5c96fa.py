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

def _enforce_sum(h):
    """Enforce sum(h) = N/2 and 0<=h<=1 with iterative projection."""
    h = h.copy()
    for _ in range(50):
        s = np.sum(h)
        diff = N/2.0 - s
        if abs(diff) < 1e-16:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    s = np.sum(h)
    diff = N/2.0 - s
    if abs(diff) > 1e-16:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _generate_from_recurrence(coeffs, order):
    """
    Generate a sequence from a linear recurrence:
    h[i] = sum_{j=1}^{order} coeffs[j-1] * h[i-j] + bias
    Then map to [0,1] via sigmoid-like scaling.
    
    coeffs: array of length order+1 (last element is bias)
    Returns: sequence of length N
    """
    # Generate raw sequence
    raw = np.zeros(N)
    # Initialize with small random-ish values based on coeffs
    for i in range(order):
        raw[i] = coeffs[i] * 0.1 + 0.5
    
    for i in range(order, N):
        val = coeffs[-1]  # bias
        for j in range(order):
            val += coeffs[j] * raw[i - j - 1]
        raw[i] = val
    
    # Normalize to [0, 1] using tanh-like mapping
    # First center and scale
    raw_centered = raw - np.mean(raw)
    raw_scaled = raw_centered / (np.std(raw_centered) + 1e-10)
    
    # Map to [0,1]
    h = 0.5 + 0.5 * np.tanh(raw_scaled)
    h = np.clip(h, 0.0, 1.0)
    
    return h

def _recurrence_objective(params):
    """
    Objective for recurrence-based optimization.
    params: [coeffs (order+1 values), scale]
    """
    order = len(params) - 1
    coeffs = params[:order]
    scale = params[order]
    
    # Generate sequence
    h = _generate_from_recurrence(coeffs, order)
    
    # Apply scale to adjust variance
    h_mean = np.mean(h)
    h_std = np.std(h) + 1e-10
    h_scaled = h_mean + scale * (h - h_mean)
    h_scaled = np.clip(h_scaled, 0.0, 1.0)
    
    # Enforce sum constraint
    h_final = _enforce_sum(h_scaled)
    
    # Check validity
    if not (np.all(h_final >= -1e-10) and np.all(h_final <= 1.0 + 1e-10)):
        return 10.0
    
    s = np.sum(h_final)
    if abs(s - N/2.0) > 1e-8:
        return 10.0
    
    return _score(h_final)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    # Start with 2-block solution as initial guess
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    current_score = _score(h0)
    h_final = h0.copy()

    def _try_update(h_candidate):
        nonlocal h_final, current_score
        h_c = _enforce_sum(h_candidate)
        if np.all(h_c >= -1e-12) and np.all(h_c <= 1.0 + 1e-12):
            h_c = np.clip(h_c, 0.0, 1.0)
            s = np.sum(h_c)
            if abs(s - N/2.0) < 1e-10:
                sc = _score(h_c)
                if sc < current_score - 1e-15:
                    h_final = h_c.copy()
                    current_score = sc
                    return True
        return False

    # Phase 1: Global search using Differential Evolution on recurrence parameters
    try:
        if time.time() < deadline - 60.0:
            # Try different orders of recurrence
            best_de_score = current_score
            best_de_h = h_final.copy()
            
            for order in [2, 3, 4]:
                if time.time() > deadline - 30.0:
                    break
                
                # Parameters: coeffs (order values) + bias (1) + scale (1) = order + 2
                n_params = order + 2
                bounds = [(-2.0, 2.0)] * (order + 1) + [(0.1, 3.0)]
                
                # Run differential evolution
                try:
                    de_result = differential_evolution(
                        _recurrence_objective,
                        bounds,
                        maxiter=200,
                        popsize=15,
                        tol=1e-10,
                        seed=42,
                        polish=True,
                        init='sobol'
                    )
                    
                    if de_result.fun < best_de_score:
                        best_de_score = de_result.fun
                        # Regenerate the best h
                        params = de_result.x
                        coeffs = params[:order]
                        bias = params[order]
                        scale = params[order + 1]
                        
                        # Regenerate with bias included
                        raw = np.zeros(N)
                        for i in range(order):
                            raw[i] = coeffs[i] * 0.1 + 0.5 + bias * 0.1
                        
                        for i in range(order, N):
                            val = bias
                            for j in range(order):
                                val += coeffs[j] * raw[i - j - 1]
                            raw[i] = val
                        
                        raw_centered = raw - np.mean(raw)
                        raw_scaled = raw_centered / (np.std(raw_centered) + 1e-10)
                        h = 0.5 + 0.5 * np.tanh(raw_scaled)
                        h = np.clip(h, 0.0, 1.0)
                        
                        h_mean = np.mean(h)
                        h_std = np.std(h) + 1e-10
                        h_scaled = h_mean + scale * (h - h_mean)
                        h_scaled = np.clip(h_scaled, 0.0, 1.0)
                        h_de = _enforce_sum(h_scaled)
                        
                        if np.all(h_de >= -1e-10) and np.all(h_de <= 1.0 + 1e-10):
                            s = np.sum(h_de)
                            if abs(s - N/2.0) < 1e-8:
                                sc = _score(h_de)
                                if sc < best_de_score:
                                    best_de_score = sc
                                    best_de_h = h_de.copy()
                
                except Exception:
                    pass
            
            # Update if DE found something better
            if best_de_score < current_score:
                _try_update(best_de_h)
                
    except Exception:
        pass

    # Phase 2: Local polish on best solution using SLSQP
    try:
        if time.time() < deadline - 10.0:
            def objective_smooth(h):
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                # Use smooth max
                alpha = 500.0
                x_max = np.max(overlap)
                smax = x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (overlap - x_max))))
                return smax

            constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
            bounds = [(0.0, 1.0)] * N

            result = minimize(
                objective_smooth,
                h_final.copy(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 3000, 'ftol': 1e-14, 'disp': False}
            )

            if result.fun < current_score + 0.01:
                _try_update(result.x)

    except Exception:
        pass

    # Phase 3: Final tight polish
    try:
        if time.time() < deadline - 3.0:
            def objective_tight(h):
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return float(np.max(overlap))

            constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
            bounds = [(0.0, 1.0)] * N

            result = minimize(
                objective_tight,
                h_final.copy(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 5000, 'ftol': 1e-16, 'disp': False}
            )

            if result.fun < current_score:
                _try_update(result.x)

    except Exception:
        pass

    # Ensure final result is valid
    h_final = _enforce_sum(h_final)
    return h_final
