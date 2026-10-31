"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
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

def _enforce_sum(h):
    """Enforce sum(h) = N/2 and 0<=h<=1 with iterative projection."""
    h = h.copy()
    for _ in range(30):
        s = np.sum(h)
        diff = N/2.0 - s
        if abs(diff) < 1e-16:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    # Final precise adjustment
    s = np.sum(h)
    diff = N/2.0 - s
    if abs(diff) > 1e-16:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _kkt_polish(h, max_iter=500):
    """
    KKT-based local refinement for the constrained problem:
    minimize max_i h_i*(1-h_{i+1})
    subject to 0<=h_i<=1, sum(h)=N/2
    
    Uses active-set identification and Newton steps on the KKT conditions.
    """
    h = h.copy()
    h = _enforce_sum(h)
    
    for it in range(max_iter):
        # Compute overlap values
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        M = np.max(overlap)
        
        # Identify active constraints
        # Active overlap constraints: overlap[i] close to M
        tol_overlap = 1e-8
        active_overlap = np.where(overlap > M - tol_overlap)[0]
        
        # Active bound constraints
        tol_bound = 1e-10
        active_lower = np.where(h < tol_bound)[0]
        active_upper = np.where(h > 1.0 - tol_bound)[0]
        
        # Free variables: not at bound and not in active overlap set
        free = np.ones(N, dtype=bool)
        free[active_lower] = False
        free[active_upper] = False
        # For overlap constraint at i, both h_i and h_{i+1} are involved
        for i in active_overlap:
            free[i] = False
            if i+1 < N:
                free[i+1] = False
        
        free_indices = np.where(free)[0]
        
        if len(free_indices) == 0:
            break
        
        # Build the linear system for free variables
        # Gradient of objective w.r.t. h_i:
        # d/dh_i [h_i*(1-h_{i+1})] = 1-h_{i+1} for the term where i is first
        # d/dh_i [h_{i-1}*(1-h_i)] = -h_{i-1} for the term where i is second
        # 
        # For the smooth max, we weight by softmax probabilities
        alpha = 50.0
        weights = np.exp(alpha * (overlap - M))
        weights /= np.sum(weights)
        
        # Gradient of smooth max
        grad = np.zeros(N)
        for i in range(N):
            # Contribution from overlap[i] = h[i]*(1-h[i+1])
            grad[i] += weights[i] * (1.0 - (h[(i+1) % N] if i+1 < N else 0.0))
            # Contribution from overlap[i-1] = h[i-1]*(1-h[i])
            if i > 0:
                grad[i] -= weights[i-1] * h[i-1]
        
        # For free variables, we want grad_i = lambda (Lagrange multiplier for sum constraint)
        # So we adjust free variables to equalize gradients
        if len(free_indices) > 1:
            # Average gradient of free variables
            lambda_val = np.mean(grad[free_indices])
            # Newton step: adjust h to make gradients equal
            # This is a simplified approach
            delta = lambda_val - grad[free_indices]
            # Step size
            step = 0.1 * delta
            h[free_indices] += step
            h = np.clip(h, 0.0, 1.0)
            h = _enforce_sum(h)
        elif len(free_indices) == 1:
            break
        
        # Check convergence
        if it > 10:
            h_prev = h.copy()
            h_new = h.copy()
            if np.max(np.abs(h_new - h_prev)) < 1e-12:
                break
    
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
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
    
    # Phase 1: SLSQP on raw max-overlap from 2-block init
    try:
        def objective_raw(h):
            shifted = np.zeros(N)
            shifted[:-1] = h[1:]
            overlap = h * (1.0 - shifted)
            return float(np.max(overlap))
        
        constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
        bounds = [(0.0, 1.0)] * N
        
        result = minimize(
            objective_raw,
            h0.copy(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000, 'ftol': 1e-15, 'disp': False}
        )
        
        if result.fun < current_score:
            _try_update(result.x)
            
    except Exception:
        pass
    
    # Phase 2: KKT polish
    try:
        if time.time() < deadline - 20.0:
            h_kkt = _kkt_polish(h_final, max_iter=200)
            _try_update(h_kkt)
            
    except Exception:
        pass
    
    # Phase 3: Smooth epigraph formulation
    try:
        if time.time() < deadline - 20.0:
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
                options={'maxiter': 2000, 'ftol': 1e-15, 'disp': False}
            )
            
            if result2.fun < 100.0:
                _try_update(result2.x[:N])
                
    except Exception:
        pass
    
    # Phase 4: Frequency-domain low-rank representation
    try:
        if time.time() < deadline - 30.0:
            n_freq = 5
            
            def _h_from_freq(params):
                c0 = params[0]
                coeffs = params[1:]
                h = np.full(N, c0)
                for k in range(1, n_freq + 1):
                    a_k = coeffs[2*(k-1)]
                    b_k = coeffs[2*(k-1) + 1]
                    h += a_k * np.cos(2*np.pi*k*np.arange(N)/N) + b_k * np.sin(2*np.pi*k*np.arange(N)/N)
                return h
            
            def objective_freq(params):
                h = _h_from_freq(params)
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return float(np.max(overlap))
            
            c0_init = np.mean(h_final)
            coeff_init = np.zeros(2*n_freq)
            for k in range(1, n_freq + 1):
                coeff_init[2*(k-1)] = 2.0/N * np.sum(h_final * np.cos(2*np.pi*k*np.arange(N)/N))
                coeff_init[2*(k-1)+1] = 2.0/N * np.sum(h_final * np.sin(2*np.pi*k*np.arange(N)/N))
            
            x0_freq = np.concatenate([[c0_init], coeff_init])
            
            bounds_freq = [(-0.5, 1.5)] + [(-0.5, 0.5)] * (2*n_freq)
            
            def constraint_sum_freq(params):
                h = _h_from_freq(params)
                return np.sum(h) - N/2.0
            
            def constraint_bounds_freq(params):
                h = _h_from_freq(params)
                return np.min(np.minimum(h, 1.0 - h))
            
            constraints_freq = [
                {'type': 'eq', 'fun': constraint_sum_freq},
                {'type': 'ineq', 'fun': constraint_bounds_freq}
            ]
            
            result_freq = minimize(
                objective_freq,
                x0_freq,
                method='SLSQP',
                bounds=bounds_freq,
                constraints=constraints_freq,
                options={'maxiter': 1000, 'ftol': 1e-14, 'disp': False}
            )
            
            if result_freq.fun < current_score:
                h_freq = _h_from_freq(result_freq.x)
                _try_update(h_freq)
                
    except Exception:
        pass
    
    # Phase 5: Multi-start with diverse structured initializations
    try:
        if time.time() < deadline - 30.0:
            np.random.seed(42)
            n_starts = 8
            for trial in range(n_starts):
                if time.time() > deadline - 15.0:
                    break
                
                if trial == 0:
                    period = N // 3
                    h_rand = np.tile([0.3, 0.7, 0.5], N // period + 1)[:N]
                elif trial == 1:
                    h_rand = 0.5 + 0.3 * np.sin(2 * np.pi * np.arange(N) / (N/2.0))
                elif trial == 2:
                    h_rand = 0.5 + 0.3 * np.cos(2 * np.pi * np.arange(N) / (N/3.0))
                elif trial == 3:
                    block_size = 10
                    blocks = [0.3, 0.7, 0.5, 0.6, 0.4]
                    h_rand = np.repeat(blocks, N // len(blocks) + 1)[:N]
                elif trial == 4:
                    h_rand = np.random.rand(N)
                    kernel = np.ones(5) / 5
                    h_rand = np.convolve(h_rand, kernel, mode='same')
                    h_rand = 0.3 + 0.4 * (h_rand - h_rand.min()) / (h_rand.max() - h_rand.min() + 1e-10)
                elif trial == 5:
                    x = np.linspace(0, 2*np.pi, N)
                    h_rand = 0.5 + 0.3 * np.abs(np.sin(x) - 0.5)
                elif trial == 6:
                    h_rand = 0.5 + 0.2 * np.sin(2*np.pi*np.arange(N)/10.0) + 0.1 * np.sin(2*np.pi*np.arange(N)/3.0)
                else:
                    h_rand = np.random.rand(N)
                
                h_rand = np.clip(h_rand, 0.0, 1.0)
                h_rand = _enforce_sum(h_rand)
                
                def objective_smooth(params):
                    h = params[:N]
                    t = params[N]
                    shifted = np.zeros(N)
                    shifted[:-1] = h[1:]
                    overlap = h * (1.0 - shifted)
                    smax = _smooth_max(overlap, alpha=200.0)
                    return t + 3000.0 * np.maximum(0.0, smax - t)**2
                
                x0 = np.concatenate([h_rand, [_score(h_rand) + 0.05]])
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
                
                result_ms = minimize(
                    objective_smooth,
                    x0,
                    method='SLSQP',
                    bounds=bounds_ext,
                    constraints=constraints_ext,
                    options={'maxiter': 1000, 'ftol': 1e-14, 'disp': False}
                )
                
                if result_ms.fun < 100.0:
                    _try_update(result_ms.x[:N])
                    
    except Exception:
        pass
    
    # Phase 6: Final KKT polish + high-precision SLSQP
    try:
        if time.time() < deadline - 10.0:
            h_kkt_final = _kkt_polish(h_final, max_iter=300)
            _try_update(h_kkt_final)
            
    except Exception:
        pass
    
    try:
        if time.time() < deadline - 3.0:
            def objective_polish(params):
                h = params[:N]
                t = params[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                smax = _smooth_max(overlap, alpha=1000.0)
                return t + 100000.0 * np.maximum(0.0, smax - t)**2
            
            x0_polish = np.concatenate([h_final, [current_score + 0.001]])
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
            
            result_final = minimize(
                objective_polish,
                x0_polish,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 5000, 'ftol': 1e-18, 'disp': False}
            )
            
            if result_final.fun < 100.0:
                _try_update(result_final.x[:N])
                
    except Exception:
        pass
    
    # Ensure final result is valid
    h_final = _enforce_sum(h_final)
    return h_final
