"""Refined Cohn-Goncalves Laguerre construction: 12-dim log-space DE + L-BFGS with gradients."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

_START = time.time()
_DEADLINE = 600.0
_MARGIN = 30.0

def _time_left():
    return _DEADLINE - (time.time() - _START) - _MARGIN

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive root locations.
    Score is the upper bound on C4 (lower is better for the evaluator's raw value,
    but the search score is seed/value, so higher search score = better).
    """
    
    # Precompute the transformation matrix from standard basis to Laguerre basis
    m = 12
    
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            try:
                A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
            except Exception:
                A[k, j] = 0.0
                
    M = A.T
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        M_inv = np.linalg.pinv(M)
    
    # Precompute weights for norm
    weights = np.arange(m + 1) + 1.0

    def compute_norm_sq_and_grad(log_roots):
        """
        Compute squared norm and its gradient w.r.t. log_roots.
        roots = exp(log_roots)
        P(x) = prod(x - r_i)
        dP/dlog_r_i = P * (-1 / r_i) * (x - r_i) / (x - r_i) ... wait
        Actually: P(x) = prod_{j != i} (x - r_j) * (x - r_i)
        dP/dlog_r_i = dP/dr_i * dr_i/dlog_r_i = P * (-1/(x - r_i)) * r_i
        But this is evaluated at each x, which is tricky for coefficient derivatives.
        
        Better: p_j = elementary symmetric polynomial of degree (m-j) in r_i.
        dp_j/dlog_r_i = dp_j/dr_i * r_i
        
        For monic polynomial P(x) = x^m + e_1 x^{m-1} + e_2 x^{m-2} + ... + e_m
        where e_k = sum of products of k distinct roots (with sign (-1)^k in standard form).
        
        Actually np.poly returns [1, -e_1, e_2, -e_3, ...] where e_k are elementary symmetric sums.
        Let's use the fact that if P(x) = prod(x - r_i), then:
        P'(x) = sum_i prod_{j != i} (x - r_j)
        And dP/dlog_r_i = r_i * dP/dr_i = r_i * prod_{j != i} (x - r_j)
        
        So dP/dlog_r_i = r_i * P(x) / (x - r_i)
        
        This means the coefficient vector of dP/dlog_r_i can be computed by dividing P by (x - r_i) and multiplying by r_i.
        
        For the gradient of norm_sq = sum_k (M_inv @ p)_k^2 * weights_k:
        Let c = M_inv @ p
        norm_sq = c^T W c where W = diag(weights)
        d(norm_sq)/dp = 2 M_inv^T W c
        d(norm_sq)/dlog_r_i = d(norm_sq)/dp * dp/dlog_r_i = 2 (M_inv^T W c)^T * (coefficient vector of r_i * P/(x-r_i))
        
        To compute P/(x - r_i): polynomial division.
        """
        roots = np.exp(log_roots)
        
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 0.0, np.zeros(12)
        
        # Build polynomial coefficients in standard basis (ascending)
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]  # ascending
        
        if len(p) < m + 1:
            p = np.pad(p, (0, m + 1 - len(p)))
        elif len(p) > m + 1:
            p = p[:m+1]
            
        c = M_inv @ p
        norm_sq = np.sum(c**2 * weights)
        
        if norm_sq <= 1e-10:
            return 0.0, np.zeros(12)
        
        # Gradient computation
        # d(norm_sq)/dp = 2 * M_inv.T @ (weights * c)
        grad_p = 2.0 * M_inv.T @ (weights * c)
        
        # For each root i, compute dp/dlog_r_i
        # P(x) / (x - r_i) is a polynomial of degree m-1
        # We can do synthetic division or use the fact that:
        # If P(x) = sum_{j=0}^m p_j x^j, then P(x)/(x - r) = sum_{j=0}^{m-1} q_j x^j
        # where q_j can be found by: p_m = q_{m-1}, p_{m-1} = q_{m-2} + r * q_{m-1}, etc.
        # Actually, (x - r) * Q(x) = P(x)
        # x*Q - r*Q = P
        # If Q = sum_{j=0}^{m-1} q_j x^j, then:
        # x*Q = sum_{j=1}^{m} q_{j-1} x^j
        # r*Q = sum_{j=0}^{m-1} r*q_j x^j
        # So coefficient of x^m: q_{m-1} = p_m = 1 (monic)
        # Coefficient of x^j for 1 <= j <= m-1: q_{j-1} - r*q_j = p_j
        # So q_{j-1} = p_j + r*q_j
        # Coefficient of x^0: -r*q_0 = p_0 => q_0 = -p_0/r
        
        grad = np.zeros(12)
        for i in range(12):
            r = roots[i]
            if r < 1e-15:
                continue
            
            # Compute Q(x) = P(x) / (x - r)
            # p is ascending: p[0], p[1], ..., p[m]
            # Q has degree m-1, ascending: q[0], ..., q[m-1]
            q = np.zeros(m)
            q[m-1] = p[m]  # = 1
            for j in range(m-1, 0, -1):
                q[j-1] = p[j] + r * q[j]
            # Check: -r*q[0] should equal p[0] (up to numerical error)
            
            # dp/dlog_r_i = r * Q(x)
            dp_dlog_r_i = r * q
            
            # Gradient contribution: grad_p dot dp_dlog_r_i
            # grad_p is length m+1, dp_dlog_r_i is length m
            # Pad dp_dlog_r_i to length m+1 (coefficient of x^m is 0)
            dp_padded = np.zeros(m + 1)
            dp_padded[:m] = dp_dlog_r_i
            
            grad[i] = np.dot(grad_p, dp_padded)
        
        return norm_sq, grad

    def cost_func(log_roots):
        norm_sq, _ = compute_norm_sq_and_grad(log_roots)
        if norm_sq <= 1e-10:
            return 1e10
        return 1.0 / norm_sq

    def cost_grad(log_roots):
        norm_sq, grad_norm = compute_norm_sq_and_grad(log_roots)
        if norm_sq <= 1e-10:
            return 1e10, np.zeros(12)
        # d(1/norm_sq)/dlog_r_i = -1/norm_sq^2 * d(norm_sq)/dlog_r_i
        grad_cost = -grad_norm / (norm_sq ** 2)
        return 1.0 / norm_sq, grad_cost

    # Optimization Strategy:
    # 1. Global search using Differential Evolution.
    # 2. Multi-start local refinement with L-BFGS-B using analytical gradients.
    # 3. Final high-precision polish.
    
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    bounds = [(-4.0, 10.0)] * 12
    
    best_roots = None
    best_val = 1e10
    best_log_roots = None
    
    # Run Differential Evolution
    if _time_left() > 60:
        try:
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=42, 
                maxiter=200, 
                popsize=15, 
                tol=1e-8, 
                polish=True,
                init='sobol',
                updating='deferred',
                workers=1
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass
    
    # Fallback / Initial guess
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    seed_val = cost_func(seed_log_roots)
    
    if best_roots is None or seed_val < best_val:
        best_val = seed_val
        best_roots = seed_roots
        best_log_roots = seed_log_roots

    # Phase 2: Multi-start local refinement with L-BFGS-B using gradients
    # Generate diverse starting points
    rng = np.random.RandomState(42)
    start_points = [best_log_roots.copy()]
    
    # Add perturbed versions of the best solution
    for _ in range(5):
        perturbation = rng.randn(12) * 0.5
        start_point = best_log_roots + perturbation
        start_point = np.clip(start_point, -4.0, 10.0)
        start_points.append(start_point)
    
    # Add some random points in the feasible region
    for _ in range(3):
        random_point = rng.uniform(-4.0, 10.0, size=12)
        start_points.append(random_point)
    
    # Run L-BFGS-B from each starting point
    for sp in start_points:
        if _time_left() < 10:
            break
        try:
            res = minimize(
                cost_func,
                jac=cost_grad,
                x0=sp,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if _time_left() > 10 and best_log_roots is not None:
        try:
            res = minimize(
                cost_func,
                jac=cost_grad,
                x0=best_log_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 200}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass
            
    # Ensure roots are positive
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
