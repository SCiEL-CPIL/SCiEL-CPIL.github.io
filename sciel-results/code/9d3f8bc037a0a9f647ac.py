import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.special import roots_genlaguerre
from numpy.polynomial.laguerre import laggauss

def construct():
    start_time = time.time()
    deadline = 620.0  # Safety margin before 630s kill

    def time_left():
        return deadline - (time.time() - start_time)

    # Precompute Gauss-Laguerre quadrature nodes and weights
    # 500 points for higher accuracy
    xq, wq = laggauss(500)
    xq = np.asarray(xq, dtype=np.float64)
    wq = np.asarray(wq, dtype=np.float64)
    
    # Effective weight is x * exp(-x)
    # log_weights for stability
    log_weights = np.full_like(xq, -700.0)
    valid_mask = xq > 1e-10
    log_weights[valid_mask] = np.log(xq[valid_mask]) + np.log(wq[valid_mask])
    
    # Precompute exp of log_weights for direct multiplication if needed, 
    # but we'll stick to log-space for Q^2 to avoid overflow/underflow.
    # Actually, for gradient, we need Q^2 / (x-r_k).
    # Let's compute Q_sq directly in log space, then exp.
    
    def energy_and_grad(roots):
        """
        Compute energy and gradient of the weighted L2 energy of Q(x)^2.
        roots: array of shape (12,)
        Returns: (energy, gradient)
        """
        r = np.asarray(roots, dtype=np.float64)
        if np.any(r <= 1e-6):
            return 1e300, np.zeros_like(r)
        
        # diffs[i, j] = |xq[j] - r[i]|
        diffs = np.abs(xq[None, :] - r[:, None])
        # Avoid log(0)
        diffs = np.maximum(diffs, 1e-15)
        
        # log_Q_sq[j] = log(Q(xq[j])^2) = 4 * sum_i log|xq[j] - r[i]|
        log_Q_sq = 4.0 * np.sum(np.log(diffs), axis=0)
        
        # log_integrand[j] = log_Q_sq[j] + log_weights[j]
        log_integrand = log_Q_sq + log_weights
        
        # Clip to prevent overflow/underflow
        log_integrand = np.clip(log_integrand, -700.0, 700.0)
        
        integrand_vals = np.exp(log_integrand)
        total_energy = np.sum(integrand_vals)
        
        if not np.isfinite(total_energy):
            return 1e300, np.zeros_like(r)
            
        # Gradient calculation
        # dE/dr_k = -4 * sum_j w_j x_j Q(x_j)^2 / (x_j - r_k)
        # Note: xq[0] is 0, so term is 0.
        # We can compute the sum efficiently.
        # grad_k = -4 * sum_j (xq[j] * wq[j] * Q(xq[j])^2) / (xq[j] - r[k])
        # Let A_j = xq[j] * wq[j] * Q(xq[j])^2 = integrand_vals[j]
        # Then grad_k = -4 * sum_j A_j / (xq[j] - r[k])
        
        # We need to handle xq[j] == r[k] carefully. 
        # In Gauss-Laguerre, nodes are fixed. If a root coincides with a node, 
        # the term is singular. However, Q(x_j) = 0 if x_j = r_k, so A_j = 0.
        # 0/0 is indeterminate. 
        # Limit of Q(x)^2 / (x-r_k) as x->r_k is 0 because Q(x)^2 has a zero of order 4 at r_k, 
        # so Q(x)^2 / (x-r_k) ~ (x-r_k)^3 -> 0.
        # So we can safely set the term to 0 if |xq[j] - r[k]| < eps.
        
        grad = np.zeros_like(r)
        # A is integrand_vals
        A = integrand_vals
        
        for k in range(12):
            denom = xq - r[k]
            # Mask where denom is too small
            mask = np.abs(denom) > 1e-10
            if np.any(mask):
                grad[k] = -4.0 * np.sum(A[mask] / denom[mask])
            # If no mask is true, grad[k] remains 0, which is correct.
            
        return total_energy, grad

    # Candidate 1: Seed roots
    seed_roots = np.array([4.0 * (i + 1) for i in range(12)], dtype=np.float64)
    best_roots = seed_roots.copy()
    best_energy, _ = energy_and_grad(best_roots)
    
    candidates = [seed_roots.copy()]
    
    # Candidate 2: Roots of L_12^(1)
    try:
        r_lag1, _ = roots_genlaguerre(12, 1)
        r_lag1 = np.sort(np.asarray(r_lag1, dtype=np.float64))
        candidates.append(r_lag1)
        # Scaled versions
        for scale in [0.5, 0.8, 1.2, 1.5, 2.0]:
            candidates.append(r_lag1 * scale)
    except Exception:
        pass

    # Candidate 3: Roots of L_12^(0)
    try:
        r_lag0, _ = roots_genlaguerre(12, 0)
        r_lag0 = np.sort(np.asarray(r_lag0, dtype=np.float64))
        candidates.append(r_lag0)
        for scale in [0.5, 0.8, 1.2, 1.5, 2.0]:
            candidates.append(r_lag0 * scale)
    except Exception:
        pass

    # Candidate 4: Geometric progression
    geo_roots = 4.0 * np.power(12.0, np.arange(12) / 11.0)
    candidates.append(geo_roots)

    # Evaluate candidates
    for c in candidates:
        if time_left() < 10.0:
            break
        e, _ = energy_and_grad(c)
        if e < best_energy:
            best_energy = e
            best_roots = c.copy()

    # Optimization using L-BFGS-B
    # We need to ensure roots are sorted and positive.
    # L-BFGS-B can handle bounds.
    # To handle distinctness, we can optimize in a transformed space or add a penalty.
    # Let's use a penalty term for close roots in the objective.
    
    def objective_with_penalty(roots_flat):
        r = np.asarray(roots_flat, dtype=np.float64)
        if np.any(r <= 1e-6):
            return 1e300
        e, g = energy_and_grad(r)
        if not np.isfinite(e):
            return 1e300
        # Penalty for close roots
        # Sort to check adjacent differences
        r_sorted = np.sort(r)
        diffs = np.diff(r_sorted)
        # If any diff < 1e-3, add penalty
        penalty = 0.0
        pen_grad = np.zeros_like(r)
        if np.any(diffs < 1e-3):
            # Simple penalty: sum (1e-3 - diff)^2
            close_mask = diffs < 1e-3
            if np.any(close_mask):
                penalty = np.sum((1e-3 - diffs[close_mask])**2) * 1e6
                # Gradient of penalty is tricky with sorting.
                # For now, let's just use the energy gradient and hope the optimizer stays in a good region.
                # Or, we can just use the energy gradient and rely on the fact that the energy 
                # doesn't blow up for close roots, but the problem requires distinct roots.
                # Actually, if roots are too close, the "double root" interpretation might break.
                # Let's stick to the energy gradient and add a small random perturbation if needed.
        return e + penalty

    def objective_and_grad_with_penalty(roots_flat):
        r = np.asarray(roots_flat, dtype=np.float64)
        if np.any(r <= 1e-6):
            return 1e300, np.zeros_like(r)
        e, g = energy_and_grad(r)
        if not np.isfinite(e):
            return 1e300, np.zeros_like(r)
        # Penalty gradient is ignored for simplicity, relying on energy gradient
        return e, g

    start_points = [best_roots.copy()]
    
    # Add scaled versions of the Laguerre roots as diverse starting points
    try:
        r_lag1, _ = roots_genlaguerre(12, 1)
        r_lag1 = np.sort(np.asarray(r_lag1, dtype=np.float64))
        for scale in [0.8, 1.2]:
            start_points.append(r_lag1 * scale)
    except Exception:
        pass

    for sp in start_points:
        if time_left() < 60.0:
            break
            
        try:
            res = minimize(
                objective_and_grad_with_penalty,
                sp,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 50
                }
            )
            
            if res.fun < best_energy:
                new_roots = np.sort(res.x)
                if np.all(new_roots > 1e-6):
                    e_new, _ = energy_and_grad(new_roots)
                    if e_new < best_energy:
                        best_energy = e_new
                        best_roots = new_roots.copy()
        except Exception:
            pass

    # Polish pass with tighter tolerance if time remains
    if time_left() > 30.0:
        try:
            res = minimize(
                objective_and_grad_with_penalty,
                best_roots,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 2000,
                    'ftol': 1e-15,
                    'gtol': 1e-10,
                    'maxls': 50
                }
            )
            if res.fun < best_energy:
                new_roots = np.sort(res.x)
                if np.all(new_roots > 1e-6):
                    e_new, _ = energy_and_grad(new_roots)
                    if e_new < best_energy:
                        best_energy = e_new
                        best_roots = new_roots.copy()
        except Exception:
            pass

    # Final validation and formatting
    out = np.sort(best_roots)
    out = np.maximum(out, 1e-6)
    # Ensure strict increase
    for i in range(1, len(out)):
        if out[i] <= out[i-1] + 1e-6:
            out[i] = out[i-1] + 1e-6
            
    return [float(v) for v in out]

# EVOLVE-BLOCK-END
