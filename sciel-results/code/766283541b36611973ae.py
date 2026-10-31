"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_MARGIN = 10.0
TARGET_SUM = N / 2.0

def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    This corresponds to the integral of h(x) * (1 - h(x-k)) over the domain,
    assuming h is zero outside [0, N].
    """
    n = len(h)
    cs = np.zeros(n)
    # k=0
    r0 = np.dot(h, h)
    cs[0] = (2.0 / n) * (n / 2.0 - r0)
    
    for k in range(1, n):
        # r_k = sum_{i=k}^{n-1} h[i] * h[i-k]
        # This is equivalent to np.dot(h[k:], h[:-k])
        r = np.dot(h[k:], h[:-k])
        cs[k] = (2.0 / n) * (n / 2.0 - r)
        
    return cs

def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))

def _smooth_max_and_grad(h, beta):
    """
    Gradient of the smooth maximum objective.
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    if m < 1e-12:
        return 0.0, np.zeros(n)
    
    # Shift for numerical stability
    shifted = cs - m
    exp_terms = np.exp(beta * shifted)
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    
    # C_k = 1 - (2/n) * r_k
    # dC_k/dh_j = -(2/n) * ( h_{j-k} (if j>=k) + h_{j+k} (if j+k<n) )
    
    for k in range(n):
        if w[k] < 1e-15:
            continue
        factor = -(2.0 / n) * w[k]
        
        for j in range(n):
            g = 0.0
            # Term 1: h_{j-k} if j >= k
            if j >= k:
                g += h[j-k]
            # Term 2: h_{j+k} if j+k < n
            if j + k < n:
                g += h[j+k]
            grad[j] += factor * g
            
    return val, grad

def _fix_sum(h, target=None):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    if target is None:
        target = TARGET_SUM
    h = np.array(h, dtype=np.float64, copy=True)
    diff = target - np.sum(h)
    # Distribute the difference evenly, but clip
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    # If clipping changed the sum, do a second pass
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    # Final tiny correction on one element to ensure exact sum in floating point
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h

def _objective_and_grad(h, beta, penalty_weight=100.0):
    """
    Objective function and gradient for optimization.
    """
    h = np.clip(h, 0.0, 1.0)
    val, grad = _smooth_max_and_grad(h, beta)
    
    # Penalty term for sum constraint
    sum_error = h.sum() - TARGET_SUM
    penalty_val = penalty_weight * sum_error * sum_error
    penalty_grad = 2.0 * penalty_weight * sum_error
    
    total_val = val + penalty_val
    total_grad = grad + penalty_grad
    
    return total_val, total_grad

def _optimize_direct_lbfgsb(h0, deadline, beta_start=50.0, beta_end=500.0, max_iter=500):
    """
    Optimize h using L-BFGS-B with analytic gradients and adaptive beta.
    """
    h = h0.copy()
    h = _fix_sum(h)
    current_score = _score(h)
    
    # We will run a few stages with increasing beta
    stages = [
        (beta_start, max_iter // 2),
        (beta_end // 2, max_iter // 4),
        (beta_end, max_iter // 4)
    ]
    
    for beta, iters in stages:
        if time.time() > deadline - 2.0:
            break
            
        def obj_func(x):
            val, _ = _objective_and_grad(x, beta)
            return val
        
        def grad_func(x):
            _, grad = _objective_and_grad(x, beta)
            return grad
        
        try:
            res = minimize(
                obj_func, 
                h, 
                jac=grad_func,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': iters, 'ftol': 1e-14, 'gtol': 1e-10, 'iprint': 0}
            )
            h_new = res.x
            h_new = _fix_sum(h_new)
            new_score = _score(h_new)
            if new_score < current_score:
                h = h_new
                current_score = new_score
        except Exception as e:
            pass
            
    return current_score, h

def _local_search_polish(h, max_iters=200, deadline=None):
    """
    Deterministic local search to polish the solution.
    Tries small adjustments and swaps.
    """
    h = h.copy()
    current_score = _score(h)
    
    if deadline is None:
        deadline = time.time() + 5.0
        
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        improved = False
        # Try small adjustments
        for i in range(N):
            for delta in [0.005, -0.005, 0.01, -0.01]:
                h_trial = h.copy()
                h_trial[i] += delta
                if h_trial[i] < 0 or h_trial[i] > 1:
                    continue
                h_trial = _fix_sum(h_trial)
                s = _score(h_trial)
                if s < current_score - 1e-12:
                    h = h_trial
                    current_score = s
                    improved = True
                    break
            if improved:
                break
        
        # Try swaps
        if not improved:
            for i in range(N - 1):
                h_trial = h.copy()
                h_trial[i], h_trial[i+1] = h_trial[i+1], h_trial[i]
                h_trial = _fix_sum(h_trial)
                s = _score(h_trial)
                if s < current_score - 1e-12:
                    h = h_trial
                    current_score = s
                    improved = True
                    break
        if not improved:
            break
            
    return current_score, h

def _sinusoidal_h(A, phi, A2=0.0, phi2=0.0):
    """Generate step heights from sinusoidal model with optional 2nd harmonic."""
    i = np.arange(N)
    h = 0.5 + A * np.sin(2.0 * np.pi * i / N + phi)
    if abs(A2) > 1e-12:
        h += A2 * np.sin(4.0 * np.pi * i / N + phi2)
    return h

def _optimize_sinusoidal(deadline):
    """
    Optimize sinusoidal parameters (A, phi, A2, phi2) via:
    1. Coarse grid search over (A, phi)
    2. Local Nelder-Mead refinement on 4 params
    """
    best_score = float('inf')
    best_h = None

    # Phase 1: Coarse grid search (1st harmonic only for speed)
    A_grid = np.linspace(0.05, 0.45, 25)
    phi_grid = np.linspace(0.0, 2.0 * np.pi, 50)

    best_A = 0.25
    best_phi = 0.0
    best_grid_score = float('inf')

    for A in A_grid:
        if time.time() > deadline - 5.0:
            break
        for phi in phi_grid:
            h = _sinusoidal_h(A, phi)
            h = np.clip(h, 0.0, 1.0)
            h = _fix_sum(h)
            s = _score(h)
            if s < best_grid_score:
                best_grid_score = s
                best_A = A
                best_phi = phi

    # Phase 2: Local refinement with Nelder-Mead on 4 params
    def obj_sin(params):
        A, phi, A2, phi2 = params
        A = np.clip(A, 0.0, 0.5)
        A2 = np.clip(A2, -0.3, 0.3)
        h = _sinusoidal_h(A, phi, A2, phi2)
        h = np.clip(h, 0.0, 1.0)
        h = _fix_sum(h)
        return _score(h)

    x0 = np.array([best_A, best_phi, 0.0, 0.0])
    try:
        res = minimize(obj_sin, x0, method='Nelder-Mead',
                       options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14})
        if res.fun < best_grid_score:
            best_grid_score = res.fun
            best_A = float(res.x[0])
            best_phi = float(res.x[1])
            best_A2 = float(res.x[2])
            best_phi2 = float(res.x[3])
    except Exception:
        best_A2 = 0.0
        best_phi2 = 0.0

    # Phase 3: Try multiple starting points for local refinement
    for A_init in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        if time.time() > deadline - 5.0:
            break
        for phi_init in [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]:
            for A2_init in [0.0, 0.05, -0.05]:
                x0 = np.array([A_init, phi_init, A2_init, 0.0])
                try:
                    res = minimize(obj_sin, x0, method='Nelder-Mead',
                                   options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
                    if res.fun < best_grid_score:
                        best_grid_score = res.fun
                        best_A = float(res.x[0])
                        best_phi = float(res.x[1])
                        best_A2 = float(res.x[2])
                        best_phi2 = float(res.x[3])
                except Exception:
                    pass

    # Build best sinusoidal h
    h_sin = _sinusoidal_h(best_A, best_phi, best_A2, best_phi2)
    h_sin = np.clip(h_sin, 0.0, 1.0)
    h_sin = _fix_sum(h_sin)
    best_score = _score(h_sin)
    best_h = h_sin.copy()

    return best_score, best_h

def construct():
    start_time = time.time()
    deadline = start_time + (150.0 - DEADLINE_MARGIN)
    
    best_score = float('inf')
    best_h = None
    
    # 1. Run the parametric sinusoidal optimization (fast, good global structure)
    s_sin, h_sin = _optimize_sinusoidal(deadline)
    if s_sin < best_score:
        best_score = s_sin
        best_h = h_sin.copy()
        
    # 2. Use the sinusoidal result as a starting point for full-vector L-BFGS-B polish
    if best_h is not None and time.time() < deadline - 10.0:
        s_opt, h_opt = _optimize_direct_lbfgsb(best_h, deadline)
        if s_opt < best_score:
            best_score = s_opt
            best_h = h_opt.copy()
            
    # 3. Final local search polish
    if best_h is not None and time.time() < deadline - 2.0:
        s_polish, h_polish = _local_search_polish(best_h, max_iters=100, deadline=deadline - 1.0)
        if s_polish < best_score:
            best_score = s_polish
            best_h = h_polish.copy()
            
    # Ensure valid output
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = np.clip(best_h, 0.0, 1.0)
    best_h = _fix_sum(best_h)
    
    return best_h

if __name__ == "__main__":
    h = construct()
    print(f"Score: {_score(h)}")
    print(f"Sum: {np.sum(h)}")
