"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _check_feasibility(M, N):
    """
    Check if there exists h in [0,1]^N with sum N/2 and max overlap <= M.
    Returns (is_feasible, L, R) where L, R are tight bounds for each h_i.
    """
    L = np.zeros(N)
    R = np.ones(N)
    
    # Direct constraint for last element
    R[-1] = min(R[-1], M)
    
    for _ in range(100):
        changed = False
        
        # Forward pass: Update L[i+1] based on R[i]
        # h[i+1] >= 1 - M/h[i]
        # Worst case for lower bound of h[i+1] is max h[i] = R[i]
        for i in range(N - 1):
            if R[i] > 1e-12:
                req_L = 1.0 - M / R[i]
                if req_L > L[i+1]:
                    L[i+1] = req_L
                    changed = True
        
        # Backward pass: Update R[i] based on L[i+1]
        # h[i] <= M / (1 - h[i+1])
        # Worst case for upper bound of h[i] is min h[i+1] = L[i+1]
        # Denominator is 1 - L[i+1]
        for i in range(N - 2, -1, -1):
            if L[i+1] < 1.0 - 1e-12:
                req_R = M / (1.0 - L[i+1])
                if req_R < R[i]:
                    R[i] = req_R
                    changed = True
            
            # Also check if R[i] is violated by L[i] directly? No, L[i] is lower bound.
            # But we must ensure L[i] <= R[i]
        
        # Check for infeasibility
        for i in range(N):
            if L[i] > R[i] + 1e-12:
                return False, L, R
        
        if not changed:
            break
    
    min_sum = np.sum(L)
    max_sum = np.sum(R)
    
    if N/2.0 < min_sum - 1e-9 or N/2.0 > max_sum + 1e-9:
        return False, L, R
    
    return True, L, R

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

def _sinusoidal_h(A, phi, N=N):
    """Generate sinusoidal step heights, projected to [0,1] with sum=N/2."""
    x = np.arange(N)
    h = 0.5 + A * np.sin(2.0 * np.pi * x / N + phi)
    return _project_to_feasible(h)

def _sinusoidal_score(params):
    """Score for sinusoidal parameters [A, phi]."""
    A, phi = params
    h = _sinusoidal_h(A, phi)
    return _score(h)

def _optimize_sinusoidal():
    """Optimize sinusoidal parameters A and phi via grid search + local refinement."""
    best_score = float('inf')
    best_A = 0.0
    best_phi = 0.0
    
    # Grid search over A in [0, 0.5] and phi in [0, 2*pi)
    n_A = 40
    n_phi = 80
    for i in range(n_A):
        A = i * 0.5 / (n_A - 1)
        for j in range(n_phi):
            phi = 2.0 * np.pi * j / n_phi
            h = _sinusoidal_h(A, phi)
            sc = _score(h)
            if sc < best_score:
                best_score = sc
                best_A = A
                best_phi = phi
    
    # Local refinement around best grid point
    # Use minimize on the 2D parameter space
    try:
        x0 = np.array([best_A, best_phi])
        result = minimize(
            _sinusoidal_score,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True}
        )
        if result.fun < best_score:
            best_score = result.fun
            best_A, best_phi = result.x
    except Exception:
        pass
    
    # Further refinement with tighter Nelder-Mead
    try:
        x0 = np.array([best_A, best_phi])
        result2 = minimize(
            _sinusoidal_score,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        if result2.fun < best_score:
            best_score = result2.fun
            best_A, best_phi = result2.x
    except Exception:
        pass
    
    h_best = _sinusoidal_h(best_A, best_phi)
    return h_best, best_score

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Initial guess: 2-block
    h0 = _two_block_init()
    current_score = _score(h0)
    h_final = h0.copy()
    
    # Phase 1: Binary search for optimal M using feasibility check
    try:
        if time.time() < deadline - 30.0:
            lo = 0.1
            hi = 0.5
            best_M = hi
            best_L = None
            best_R = None
            
            # Binary search
            for _ in range(60):
                if time.time() > deadline - 35.0:
                    break
                mid = (lo + hi) / 2.0
                feasible, L, R = _check_feasibility(mid, N)
                if feasible:
                    hi = mid
                    best_M = mid
                    best_L = L
                    best_R = R
                else:
                    lo = mid
            
            if best_L is not None:
                # Construct h from L, R
                # h_i = L_i + alpha * (R_i - L_i)
                sum_L = np.sum(best_L)
                sum_R = np.sum(best_R)
                target = N / 2.0
                
                if sum_R - sum_L > 1e-9:
                    alpha = (target - sum_L) / (sum_R - sum_L)
                    alpha = np.clip(alpha, 0.0, 1.0)
                    h_candidate = best_L + alpha * (best_R - best_L)
                    h_candidate = _project_to_feasible(h_candidate)
                    
                    sc = _score(h_candidate)
                    if sc < current_score:
                        h_final = h_candidate.copy()
                        current_score = sc
                else:
                    h_candidate = _project_to_feasible(best_L)
                    sc = _score(h_candidate)
                    if sc < current_score:
                        h_final = h_candidate.copy()
                        current_score = sc
    except Exception:
        pass
    
    # Phase 2: Polish with SLSQP using the found M as a constraint
    try:
        if time.time() < deadline - 10.0:
            M_target = current_score + 1e-6
            
            def objective_polish(params):
                h = params[:N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                smax = _smooth_max(overlap, alpha=1000.0)
                # Penalize exceeding M_target
                penalty = 10000.0 * np.maximum(0.0, smax - M_target)**2
                return penalty
            
            x0 = h_final.copy()
            bounds_ext = [(0.0, 1.0)] * N
            
            def constraint_sum(params):
                return np.sum(params) - N/2.0
            
            constraints_ext = [
                {'type': 'eq', 'fun': constraint_sum}
            ]
            
            result = minimize(
                objective_polish,
                x0,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            
            if result.fun < 1000.0:
                h_opt = result.x.copy()
                h_opt = _project_to_feasible(h_opt)
                new_score = _score(h_opt)
                if new_score < current_score:
                    h_final = h_opt.copy()
                    current_score = new_score
    except Exception:
        pass
    
    # Final validation
    h_final = _project_to_feasible(h_final)
    
    if not (np.all(h_final >= -1e-8) and np.all(h_final <= 1.0 + 1e-8)):
        h_final = np.full(N, 0.5)
    
    return h_final
