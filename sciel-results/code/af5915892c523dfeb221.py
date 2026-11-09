import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 590  # Safety margin for 630s kill

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

    def smooth_min_dist_sq(pts, lam):
        """
        Compute a smooth approximation of the minimum squared distance.
        Uses the log-sum-exp trick.
        """
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (n, d):
            return 0.0
        dists_sq = pdist(pts) ** 2
        if len(dists_sq) == 0:
            return 0.0
        
        m = np.max(dists_sq)
        if m < 1e-12:
            return 0.0
            
        shifted = -lam * (dists_sq - m)
        # Clip to avoid underflow issues
        shifted = np.clip(shifted, -700, 0)
        exp_sum = np.sum(np.exp(shifted))
        if exp_sum < 1e-300:
            return m
        log_sum = np.log(exp_sum)
        smooth_min = m - (1.0 / lam) * log_sum
        return smooth_min

    def objective_neg_smooth_min(x, lam):
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
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

    def coulomb_objective(x):
        """
        Minimize sum of 1/r^2 between all pairs.
        This encourages points to spread out as much as possible.
        We use a soft version to avoid singularities: 1/(r^2 + eps)^2
        """
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        dists_sq = pdist(pts) ** 2
        eps = 1e-6
        # Sum of 1/(d^2 + eps)^2
        val = np.sum(1.0 / (dists_sq + eps)**2)
        return val

    def coulomb_gradient(x):
        """
        Numerical gradient for Coulomb objective.
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
            
            f_plus = coulomb_objective(x_plus)
            f_minus = coulomb_objective(x_minus)
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad

    def fibonacci_sphere_init(rng):
        indices = np.arange(0, n, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n)
        theta = np.pi * (1 + 5**0.5) * indices
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        pts = np.column_stack([x, y, z])
        pts = pts - np.mean(pts, axis=0)
        return pts

    best_score = 0.0
    best_points = None

    # Strategy 1: Coulomb Repulsion Pre-stage
    # This helps spread points out before optimizing the min/max ratio.
    # We run this for a few iterations to get a good starting point.
    
    # Initialize with Fibonacci sphere
    rng = np.random.RandomState(42)
    x0 = fibonacci_sphere_init(rng).flatten()
    
    try:
        # Optimize Coulomb energy
        res_coulomb = minimize(
            coulomb_objective,
            x0,
            jac=coulomb_gradient,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res_coulomb.x is not None:
            cand = res_coulomb.x.reshape((n, d))
            cand = cand - np.mean(cand, axis=0)
            s = score(cand)
            if s > best_score:
                best_score = s
                best_points = cand.copy()
    except Exception:
        pass

    if best_points is None:
        best_points = fibonacci_sphere_init(rng)
        best_score = score(best_points)

    # Strategy 2: Differential Evolution for global search on smooth min
    # Use a moderate lambda for the DE phase to keep it smooth enough
    lam_de = 10.0
    
    def de_objective(x):
        return objective_neg_smooth_min(x, lam_de)

    bounds = [(-1.0, 1.0)] * (n * d)
    
    try:
        if time.time() < deadline - 180:
            result_de = differential_evolution(
                de_objective,
                bounds,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=False,
                updating='deferred'
            )
            if result_de.x is not None:
                cand = result_de.x.reshape((n, d))
                cand = cand - np.mean(cand, axis=0)
                s = score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
    except Exception:
        pass

    # Strategy 3: Local Refinement with increasing lambda (Annealing)
    # Start from best_points if found, otherwise random init
    
    lambdas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    current_points = best_points.copy()
    
    for lam in lambdas:
        if time.time() > deadline - 30:
            break
            
        try:
            x0 = current_points.flatten()
            
            def obj(x):
                return objective_neg_smooth_min(x, lam)
            
            def grad(x):
                return gradient_neg_smooth_min(x, lam)
            
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
        except Exception:
            pass

    # Strategy 4: Multiple restarts with random perturbations if time remains
    if time.time() < deadline - 60:
        rng2 = np.random.RandomState(999)
        for i in range(10):
            if time.time() > deadline - 30:
                break
            # Perturb the best solution
            perturbed = best_points + rng2.randn(n, d) * 0.1
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
