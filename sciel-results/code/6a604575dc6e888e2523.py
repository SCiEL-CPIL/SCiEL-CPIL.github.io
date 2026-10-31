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

    # Structural Change: Fourier Domain Optimization
    # We represent h as a sum of Fourier modes.
    # h[i] = c0 + sum_{k=1}^{K} (a_k cos(2 pi k i / N) + b_k sin(2 pi k i / N))
    # This reduces dimensionality and encourages smooth solutions.
    
    K = 10 # Number of harmonics
    
    def _h_from_fourier(params):
        c0 = params[0]
        a = params[1:1+K]
        b = params[1+K:1+2*K]
        i = np.arange(N)
        h = np.full(N, c0)
        for k in range(1, K+1):
            angle = 2 * np.pi * k * i / N
            h += a[k-1] * np.cos(angle) + b[k-1] * np.sin(angle)
        return h

    def _fourier_objective(params):
        h = _h_from_fourier(params)
        # Penalty for bounds
        bound_violation = np.sum(np.maximum(0, h - 1.0)**2) + np.sum(np.maximum(0, -h)**2)
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        smax = _smooth_max(overlap, alpha=500.0)
        return smax + 100.0 * bound_violation

    def _fourier_gradient(params):
        h = _h_from_fourier(params)
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        
        # Gradient of smooth max
        alpha = 500.0
        M = np.max(overlap)
        weights = np.exp(alpha * (overlap - M))
        weights /= np.sum(weights)
        
        grad_h = np.zeros(N)
        for i in range(N):
            # d/dh_i [h_i*(1-h_{i+1})]
            grad_h[i] += weights[i] * (1.0 - h[(i+1) % N] if i+1 < N else 1.0)
            # d/dh_i [h_{i-1}*(1-h_i)]
            if i > 0:
                grad_h[i] -= weights[i-1] * h[i-1]
        
        # Gradient of bound penalty
        grad_bound = np.zeros(N)
        grad_bound[h > 1.0] += 200.0 * (h[h > 1.0] - 1.0)
        grad_bound[h < 0.0] -= 200.0 * h[h < 0.0]
        
        total_grad_h = grad_h + grad_bound
        
        # Map gradient from h-space to Fourier params
        grad_params = np.zeros(1 + 2*K)
        grad_params[0] = np.sum(total_grad_h) / N # c0 contribution
        
        i = np.arange(N)
        for k in range(1, K+1):
            angle = 2 * np.pi * k * i / N
            grad_params[1+k-1] = np.sum(total_grad_h * np.cos(angle)) * 2.0 / N
            grad_params[1+K+k-1] = np.sum(total_grad_h * np.sin(angle)) * 2.0 / N
            
        return grad_params

    # Initialize Fourier coefficients from h_final
    c0_init = np.mean(h_final)
    a_init = np.zeros(K)
    b_init = np.zeros(K)
    i = np.arange(N)
    for k in range(1, K+1):
        angle = 2 * np.pi * k * i / N
        a_init[k-1] = 2.0/N * np.sum(h_final * np.cos(angle))
        b_init[k-1] = 2.0/N * np.sum(h_final * np.sin(angle))
    
    x0_fourier = np.concatenate([[c0_init], a_init, b_init])
    
    bounds_fourier = [(-1.0, 2.0)] + [(-1.0, 1.0)] * (2*K)
    
    try:
        if time.time() < deadline - 20.0:
            result_fourier = minimize(
                _fourier_objective,
                x0_fourier,
                jac=_fourier_gradient,
                method='L-BFGS-B',
                bounds=bounds_fourier,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result_fourier.success or result_fourier.fun < current_score:
                h_fourier = _h_from_fourier(result_fourier.x)
                _try_update(h_fourier)
                
    except Exception:
        pass

    # Phase 2: KKT polish on the best h found
    try:
        if time.time() < deadline - 20.0:
            h_kkt = _kkt_polish(h_final, max_iter=200)
            _try_update(h_kkt)
            
    except Exception:
        pass

    # Phase 3: Local SLSQP on h directly for final polish
    try:
        if time.time() < deadline - 10.0:
            def objective_raw(h):
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return float(np.max(overlap))
            
            constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
            bounds = [(0.0, 1.0)] * N
            
            result_local = minimize(
                objective_raw,
                h_final.copy(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
            )
            
            if result_local.fun < current_score:
                _try_update(result_local.x)
                
    except Exception:
        pass

    # Ensure final result is valid
    h_final = _enforce_sum(h_final)
    return h_final
