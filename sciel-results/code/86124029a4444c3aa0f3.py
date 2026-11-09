import numpy as np
import time
import sys

def construct():
    start_time = time.time()
    deadline = 620.0  # Safety margin before 630s kill

    def time_left():
        return deadline - (time.time() - start_time)

    # Precompute Gauss-Laguerre quadrature nodes and weights
    # We need sufficient accuracy. 200 points is usually enough for smooth integrands,
    # but our integrand has singularities at the roots (though squared, so it's 0 there, 
    # wait, Q(x) is 0 at roots, so Q(x)^2 is 0. The integrand is smooth everywhere except 
    # potentially large dynamic range. Log-space handles this well.)
    try:
        from numpy.polynomial.laguerre import laggauss
        # Use 300 points for higher precision
        xq, wq = laggauss(300)
        xq = np.asarray(xq, dtype=np.float64)
        wq = np.asarray(wq, dtype=np.float64)
    except Exception:
        # Fallback to simple trapezoidal if laggauss fails (unlikely)
        xq = np.linspace(0, 200, 5000)
        wq = np.exp(-xq) * (200.0 / (5000 - 1))

    # The weight for the energy integral is x * exp(-x).
    # Gauss-Laguerre weights wq are for exp(-x).
    # So the effective weight for our integral is xq * wq.
    # We compute log(xq * wq) to add to the log of the polynomial part.
    # Note: xq[0] is 0, so log(0) is -inf. We handle this by masking or using a small epsilon.
    # However, Q(0) is non-zero (product of -r_i^2), so the integrand at 0 is 0 * Q(0)^2 * 1 = 0.
    # So we can safely ignore the first node or set its contribution to 0.
    log_weights = np.full_like(xq, -700.0) # -inf approx
    valid_mask = xq > 1e-10
    log_weights[valid_mask] = np.log(xq[valid_mask]) + np.log(wq[valid_mask])

    def energy(roots):
        """
        Compute the weighted L2 energy of Q(x)^2 where Q(x) = prod (x-r_i)^2.
        Energy = sum_j w_j * x_j * (prod_i (x_j - r_i)^2)^2
        log(Energy) is not linear, but we can compute the sum in log-space if we want max,
        but we need sum of exps.
        log_term_j = log(x_j) + log(w_j) + 4 * sum_i log(|x_j - r_i|)
        Energy = sum_j exp(log_term_j)
        """
        r = np.asarray(roots, dtype=np.float64)
        if np.any(r <= 0.0):
            return 1e300
        
        # Compute log |x - r_i| for all i
        # Shape: (12, 300)
        diffs = np.abs(xq[None, :] - r[:, None])
        # Avoid log(0)
        diffs = np.maximum(diffs, 1e-15)
        
        log_Q = 2.0 * np.sum(np.log(diffs), axis=0) # log(Q(x))
        # We need Q(x)^2, so log(Q(x)^2) = 2 * log(Q(x)) = 4 * sum log|x-r_i|
        log_Q_sq = 2.0 * log_Q
        
        log_integrand = log_Q_sq + log_weights
        
        # Clip to prevent overflow/underflow in exp
        # If log_integrand is very large, exp will be inf.
        # If very small, exp will be 0.
        log_integrand = np.clip(log_integrand, -700.0, 700.0)
        
        integrand_vals = np.exp(log_integrand)
        total_energy = np.sum(integrand_vals)
        
        if not np.isfinite(total_energy):
            return 1e300
            
        return float(total_energy)

    # Candidate 1: Seed roots
    seed_roots = np.array([4.0 * (i + 1) for i in range(12)], dtype=np.float64)
    best_roots = seed_roots.copy()
    best_energy = energy(best_roots)
    
    candidates = [seed_roots.copy()]
    
    # Candidate 2: Roots of L_12^(1)
    try:
        from scipy.special import roots_genlaguerre
        r_lag1, _ = roots_genlaguerre(12, 1)
        r_lag1 = np.sort(np.asarray(r_lag1, dtype=np.float64))
        candidates.append(r_lag1)
    except Exception:
        pass

    # Candidate 3: Roots of L_12^(0)
    try:
        from scipy.special import roots_genlaguerre
        r_lag0, _ = roots_genlaguerre(12, 0)
        r_lag0 = np.sort(np.asarray(r_lag0, dtype=np.float64))
        # Scale to match the spread of the seed roughly?
        # Seed spread is 4 to 48. L12(0) roots are approx 1.5 to 35.
        # Let's try unscaled and scaled.
        candidates.append(r_lag0)
        candidates.append(r_lag0 * (48.0 / r_lag0.max()))
    except Exception:
        pass

    # Candidate 4: Geometric progression
    geo_roots = 4.0 * np.power(12.0, np.arange(12) / 11.0) # 4 to 48
    candidates.append(geo_roots)

    # Evaluate candidates
    for c in candidates:
        if time_left() < 10.0:
            break
        e = energy(c)
        if e < best_energy:
            best_energy = e
            best_roots = c.copy()

    # Optimization
    try:
        from scipy.optimize import minimize
        
        def objective(roots_flat):
            r = np.asarray(roots_flat, dtype=np.float64)
            if np.any(r <= 1e-6):
                return 1e300
            # Add penalty for closely-spaced roots to improve conditioning
            sorted_r = np.sort(r)
            gaps = np.diff(sorted_r)
            min_gap = np.min(gaps)
            if min_gap < 0.1:
                return 1e300 + (0.1 - min_gap) * 1e10
            return energy(r)

        # Generate diverse starting points
        start_points = [best_roots.copy()]
        
        # Add scaled versions of the Laguerre roots as diverse starting points
        try:
            from scipy.special import roots_genlaguerre
            r_lag1, _ = roots_genlaguerre(12, 1)
            r_lag1 = np.sort(np.asarray(r_lag1, dtype=np.float64))
            for scale in [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
                start_points.append(r_lag1 * scale)
            
            r_lag0, _ = roots_genlaguerre(12, 0)
            r_lag0 = np.sort(np.asarray(r_lag0, dtype=np.float64))
            for scale in [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
                start_points.append(r_lag0 * scale)
        except Exception:
            pass
        
        # Add geometric progressions with different bases
        for base in [1.2, 1.5, 2.0, 3.0]:
            geo = 4.0 * np.power(base, np.arange(12) / 11.0)
            if geo.max() < 100:
                start_points.append(geo)

        # Run optimization from multiple starting points
        for sp in start_points:
            if time_left() < 45.0:
                break
            
            try:
                # First try L-BFGS-B for better convergence
                res = minimize(
                    objective,
                    sp,
                    method='L-BFGS-B',
                    bounds=[(1e-4, None)] * 12,
                    options={
                        'maxiter': 5000,
                        'ftol': 1e-12,
                        'gtol': 1e-10,
                        'maxls': 50
                    }
                )
                
                if res.fun < best_energy:
                    new_roots = np.sort(res.x)
                    if np.all(new_roots > 1e-6):
                        e_new = energy(new_roots)
                        if e_new < best_energy:
                            best_energy = e_new
                            best_roots = new_roots.copy()
            except Exception:
                pass
            
            # Then try Nelder-Mead as a local polish
            if time_left() < 30.0:
                break
            try:
                res = minimize(
                    objective,
                    best_roots,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 3000,
                        'xatol': 1e-9,
                        'fatol': 1e-11,
                        'adaptive': True
                    }
                )
                
                if res.fun < best_energy:
                    new_roots = np.sort(res.x)
                    if np.all(new_roots > 1e-6):
                        e_new = energy(new_roots)
                        if e_new < best_energy:
                            best_energy = e_new
                            best_roots = new_roots.copy()
            except Exception:
                pass

        # Final high-precision polish pass if time remains
        if time_left() > 20.0:
            try:
                res = minimize(
                    objective,
                    best_roots,
                    method='L-BFGS-B',
                    bounds=[(1e-4, None)] * 12,
                    options={
                        'maxiter': 10000,
                        'ftol': 1e-15,
                        'gtol': 1e-12,
                        'maxls': 100
                    }
                )
                if res.fun < best_energy:
                    new_roots = np.sort(res.x)
                    if np.all(new_roots > 1e-6):
                        e_new = energy(new_roots)
                        if e_new < best_energy:
                            best_energy = e_new
                            best_roots = new_roots.copy()
            except Exception:
                pass

    except Exception:
        pass

    # Final validation and formatting
    out = np.sort(best_roots)
    out = np.maximum(out, 1e-6)
    # Ensure strict increase
    for i in range(1, len(out)):
        if out[i] <= out[i-1]:
            out[i] = out[i-1] + 1e-6
            
    return [float(v) for v in out]

# EVOLVE-BLOCK-END
