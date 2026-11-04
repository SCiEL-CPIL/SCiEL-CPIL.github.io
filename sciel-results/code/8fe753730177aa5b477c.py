import numpy as np
import scipy.optimize
import time

N = 100

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin for the 120s limit

    def get_correlations(h):
        """
        Compute linear correlations of h and 1-h for all non-zero shifts.
        """
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        c_nonzero = np.concatenate([c[:N-1], c[N:]])
        return c_nonzero

    def project_to_feasible(h):
        """
        Project h onto the feasible set: 0 <= h_i <= 1, sum(h) = N/2.
        Uses bisection on a Lagrange multiplier for exact projection.
        """
        h = np.clip(h, 0.0, 1.0).copy()
        target = N / 2.0
        current = np.sum(h)
        
        if abs(current - target) < 1e-12:
            return h
        
        # Bisection on lambda: find lambda such that sum(clip(h - lambda, 0, 1)) = target
        # When lambda < 0, we are effectively adding to h (shifting up)
        # When lambda > 0, we are subtracting from h (shifting down)
        # We want sum(clip(h - lambda, 0, 1)) = target
        # If current > target, we need to reduce sum, so lambda should be positive (subtract)
        # If current < target, we need to increase sum, so lambda should be negative (add)
        
        # Range for lambda:
        # Min possible sum is 0 (if lambda is large positive)
        # Max possible sum is N (if lambda is large negative)
        # We can bound lambda roughly.
        lo = -1.0 - max(h)  # Ensures clip(h-lo, 0, 1) is close to 1 for all h
        hi = 1.0 + max(h)   # Ensures clip(h-hi, 0, 1) is close to 0 for all h
        
        def sum_at_lambda(lam):
            return np.sum(np.clip(h - lam, 0.0, 1.0))
        
        # Check boundaries to ensure we bracket the target
        # If sum_at_lambda(lo) < target, we might need to go lower (more negative)?
        # Wait. clip(h - lam, 0, 1).
        # If lam is very negative, h - lam is very large positive -> clip to 1. Sum = N.
        # If lam is very positive, h - lam is very large negative -> clip to 0. Sum = 0.
        
        # So:
        # sum_at_lambda(-large) ~ N
        # sum_at_lambda(+large) ~ 0
        
        # We want sum = target (N/2).
        
        # If current sum > target, we need to decrease sum.
        # Decreasing sum means increasing lambda.
        # If current sum < target, we need to increase sum.
        # Increasing sum means decreasing lambda.
        
        # Let's just bisect in the range [-2, 2] which is usually sufficient for h in [0,1]
        # Actually, let's use a wider range to be safe.
        lo = -2.0
        hi = 2.0
        
        # Verify bracket
        s_lo = sum_at_lambda(lo)
        s_hi = sum_at_lambda(hi)
        
        # If target is not between s_lo and s_hi, expand?
        # For N=100, target=50. s_lo ~ 100, s_hi ~ 0. 50 is between.
        
        for _ in range(60):
            mid = (lo + hi) / 2.0
            s_mid = sum_at_lambda(mid)
            if abs(s_mid - target) < 1e-12:
                break
            if s_mid > target:
                # Sum is too high, need to reduce sum -> increase lambda
                lo = mid
            else:
                # Sum is too low, need to increase sum -> decrease lambda
                hi = mid
        
        lam = (lo + hi) / 2.0
        h = np.clip(h - lam, 0.0, 1.0)
        
        # Final adjustment for any residual error due to floating point
        residual = target - np.sum(h)
        if abs(residual) > 1e-14:
            # Distribute residual among non-boundary elements
            interior = (h > 1e-10) & (h < 1.0 - 1e-10)
            if np.sum(interior) > 0:
                h[interior] += residual / np.sum(interior)
                h = np.clip(h, 0.0, 1.0)
        
        return h

    def lse_objective(h, beta):
        """
        LogSumExp of correlations for smooth approximation of max.
        """
        c_all = get_correlations(h)
        max_c = np.max(c_all)
        if max_c > 500:
            max_c = 500.0
        # Stable LSE
        lse = max_c + np.log(np.sum(np.exp(beta * (c_all - max_c)))) / beta
        return lse

    def optimize_instance(h0, beta, max_iter, ftol=1e-10, gtol=1e-10):
        """
        Optimize using L-BFGS-B.
        """
        bounds = [(0, 1) for _ in range(N)]
        
        def obj(h):
            h_proj = project_to_feasible(h)
            return lse_objective(h_proj, beta)

        try:
            res = scipy.optimize.minimize(
                obj, 
                h0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
            )
            h_opt = res.x
        except Exception:
            h_opt = h0
        
        h_opt = project_to_feasible(h_opt)
        return h_opt

    def score_h(h):
        """Calculate the true max overlap score."""
        c_all = get_correlations(h)
        return np.max(c_all)

    best_h = np.full(N, 0.5)
    best_score = score_h(best_h)  # N/4 = 25.0

    rng = np.random.default_rng(42)
    
    # Multi-restart strategy
    restarts = []
    
    # 1. Uniform start
    restarts.append((np.full(N, 0.5), 5.0, 800))
    
    # 2-6. Random perturbations
    for i in range(5):
        h_rand = rng.uniform(0.1, 0.9, size=N)
        h_rand = project_to_feasible(h_rand)
        beta = 5.0 + i * 5.0
        restarts.append((h_rand, beta, 800))
        
    # 7-12. Bimodal initializations
    for i in range(6):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        h_rand = np.where(h_rand > 0.5, 0.8, 0.2)
        h_rand = project_to_feasible(h_rand)
        beta = 20.0 + i * 5.0
        restarts.append((h_rand, beta, 600))

    # Run all restarts
    for idx, (h0, beta, max_iter) in enumerate(restarts):
        if time.time() - start_time > time_limit:
            break
            
        h_opt = optimize_instance(h0, beta, max_iter)
        current_score = score_h(h_opt)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h_opt.copy()
            
    # High-precision polish pass on the best solution
    if time.time() - start_time < time_limit - 20.0:
        # Use very high beta and tight tolerances
        polish_beta = 100.0
        h_polished = optimize_instance(best_h, polish_beta, 2000, ftol=1e-12, gtol=1e-12)
        polished_score = score_h(h_polished)
        
        if polished_score < best_score:
            best_score = polished_score
            best_h = h_polished.copy()
            
    # Final constraint verification
    best_h = project_to_feasible(best_h)
    
    return best_h
