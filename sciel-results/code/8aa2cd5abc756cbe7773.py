import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 590  # Safety margin for 630s kill

    def smooth_min_dist_sq(pts, lam):
        """
        Compute a smooth approximation of the minimum squared distance.
        Uses the log-sum-exp trick: 
        min(x_i) approx -1/lam * log(sum(exp(-lam * x_i)))
        Here x_i are squared distances.
        """
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (n, d):
            return 0.0
        dists_sq = pdist(pts) ** 2
        if len(dists_sq) == 0:
            return 0.0
        
        # To avoid overflow/underflow in exp, subtract max
        # We want -1/lam * log(sum(exp(-lam * d^2)))
        # This is equivalent to -1/lam * ( -lam * max(d^2) + log(sum(exp(-lam * (d^2 - max(d^2))))) )
        # = max(d^2) - 1/lam * log(sum(exp(-lam * (d^2 - max(d^2)))))
        # Wait, min is what we want.
        # Let y_i = d_i^2. We want min(y_i).
        # min(y) = -1/lam * log(sum(exp(-lam * y)))
        # Let m = max(y). Then -lam * y = -lam * (y - m) - lam * m.
        # sum(exp(-lam * y)) = exp(-lam * m) * sum(exp(-lam * (y - m)))
        # log(sum(exp(-lam * y))) = -lam * m + log(sum(exp(-lam * (y - m))))
        # min(y) = -1/lam * (-lam * m + log(sum(exp(-lam * (y - m)))))
        # min(y) = m - 1/lam * log(sum(exp(-lam * (y - m))))
        
        m = np.max(dists_sq)
        # If all distances are 0, return 0
        if m < 1e-12:
            return 0.0
            
        # Calculate the smooth min
        # Note: we want to MAXIMIZE the min distance, so we MINIMIZE -smooth_min
        # The function returns the approximate min distance squared.
        # We will negate it in the objective function.
        
        # For numerical stability, if lam is large, exp(-lam*(y-m)) is 0 for y>m and 1 for y=m.
        # sum is approx count_of_min.
        # log(sum) is approx log(count).
        # min approx m - log(count)/lam.
        
        shifted = -lam * (dists_sq - m)
        # Clip to avoid underflow issues, though exp handles it usually
        # exp(shifted) where shifted <= 0
        exp_sum = np.sum(np.exp(shifted))
        if exp_sum < 1e-300:
            return m
        log_sum = np.log(exp_sum)
        smooth_min = m - (1.0 / lam) * log_sum
        return smooth_min

    def objective_neg_smooth_min(x, lam):
        pts = x.reshape((n, d))
        # Center the points to remove translational invariance
        pts = pts - np.mean(pts, axis=0)
        # Optional: normalize to unit sphere to prevent scaling issues?
        # The ratio is scale invariant, but the optimizer might drift.
        # Let's keep it free but centered.
        
        val = smooth_min_dist_sq(pts, lam)
        return -val

    def gradient_neg_smooth_min(x, lam):
        """
        Numerical gradient for the smooth minimum objective.
        """
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        
        eps = 1e-7
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            
            f_plus = objective_neg_smooth_min(x_plus, lam)
            f_minus = objective_neg_smooth_min(x_minus, lam)
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad

    def score(pts):
        """Compute (d_min / d_max)^2 for the point configuration."""
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (n, d) or not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    best_score = 0.0
    best_points = None

    # Strategy 1: Differential Evolution for global search
    # Bounds: We can assume points are within [-1, 1] for each dimension.
    # Since the ratio is scale invariant, the absolute scale doesn't matter,
    # but the optimizer needs a bounded space.
    bounds = [(-1.0, 1.0)] * (n * d)
    
    # Use a moderate lambda for the DE phase to keep it smooth enough
    lam_de = 10.0
    
    def de_objective(x):
        return objective_neg_smooth_min(x, lam_de)

    try:
        if time.time() < deadline - 180:
            result_de = differential_evolution(
                de_objective,
                bounds,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=False, # We will polish separately
                updating='deferred',
                workers=-1 # Use all cores if possible, but DE in scipy is often single-threaded per iteration?
                # Actually, scipy DE is not parallelized by default in older versions, 
                # but 'workers' might not be supported or might be slow. 
                # Let's stick to default single-threaded to be safe with time.
            )
            if result_de.x is not None:
                cand = result_de.x.reshape((n, d))
                cand = cand - np.mean(cand, axis=0)
                s = score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
    except Exception as e:
        pass

    # Strategy 2: Local Refinement with increasing lambda (Annealing)
    # Start from best_points if found, otherwise random init
    if best_points is None:
        # Fallback initialization: Fibonacci sphere
        rng = np.random.RandomState(123)
        indices = np.arange(0, n, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n)
        theta = np.pi * (1 + 5**0.5) * indices
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        best_points = np.column_stack([x, y, z])
        best_points = best_points - np.mean(best_points, axis=0)
        best_score = score(best_points)

    # Annealing loop: Increase lambda to sharpen the minimum
    # We run L-BFGS-B with numerical gradient
    lambdas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    current_points = best_points.copy()
    
    for lam in lambdas:
        if time.time() > deadline - 30:
            break
            
        try:
            # L-BFGS-B with numerical gradient
            # We flatten the points
            x0 = current_points.flatten()
            
            # Define the objective and gradient for this specific lambda
            def obj(x):
                return objective_neg_smooth_min(x, lam)
            
            def grad(x):
                return gradient_neg_smooth_min(x, lam)
            
            # Run minimize
            res = minimize(
                obj,
                x0,
                jac=grad,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-8}
            )
            
            if res.x is not None:
                cand = res.x.reshape((n, d))
                cand = cand - np.mean(cand, axis=0)
                s = score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
                    current_points = cand.copy()
        except Exception as e:
            # If optimization fails, keep current points
            pass

    # Strategy 3: Multiple restarts with random perturbations if time remains
    if time.time() < deadline - 60:
        rng = np.random.RandomState(999)
        for i in range(5):
            if time.time() > deadline - 30:
                break
            # Perturb the best solution
            perturbed = best_points + rng.randn(n, d) * 0.1
            perturbed = perturbed - np.mean(perturbed, axis=0)
            
            # Quick polish with high lambda
            lam = 50.0
            x0 = perturbed.flatten()
            try:
                def obj(x):
                    return objective_neg_smooth_min(x, lam)
                def grad(x):
                    return gradient_neg_smooth_min(x, lam)
                
                res = minimize(
                    obj,
                    x0,
                    jac=grad,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-8}
                )
                if res.x is not None:
                    cand = res.x.reshape((n, d))
                    cand = cand - np.mean(cand, axis=0)
                    s = score(cand)
                    if s > best_score:
                        best_score = s
                        best_points = cand.copy()
            except Exception:
                pass

    # Final normalization to ensure finite and clean output
    # The score is scale invariant, so we can scale to any reasonable size.
    # Let's normalize to have max distance = 1 for consistency.
    if best_points is not None and np.all(np.isfinite(best_points)):
        dists = pdist(best_points)
        d_max = np.max(dists)
        if d_max > 1e-12:
            best_points = best_points / d_max
        # Center again
        best_points = best_points - np.mean(best_points, axis=0)
        
    else:
        # Fallback to zero if something went wrong (should not happen)
        best_points = np.zeros((n, d))

    return best_points
