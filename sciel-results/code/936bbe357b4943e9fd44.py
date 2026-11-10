# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, SLSQP
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    start_time = time.time()
    deadline = 600  # seconds
    safety_margin = 30  # seconds
    
    n = 16
    d = 2
    
    def compute_R(points):
        """Compute the R metric directly."""
        dists = pdist(points, 'euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective_smooth(params, p_min=100, p_max=100):
        """
        Smooth surrogate objective using log-sum-exp for soft-min and soft-max.
        """
        if time.time() > start_time + deadline - safety_margin:
            raise TimeoutError()
        
        points = params.reshape(n, d)
        dists = pdist(points, 'euclidean')
        if len(dists) == 0:
            return 0.0
        
        # Soft min of dists
        d_scaled_min = -dists / p_min
        max_d_min = np.max(d_scaled_min)
        soft_min = -p_min * (max_d_min + np.log(np.sum(np.exp(d_scaled_min - max_d_min))))
        
        # Soft max of dists
        d_scaled_max = dists / p_max
        max_d_max = np.max(d_scaled_max)
        soft_max = p_max * (max_d_max + np.log(np.sum(np.exp(d_scaled_max - max_d_max))))
        
        if soft_max <= 0:
            return 0.0
        
        ratio = soft_min / soft_max
        return -(ratio ** 2)

    def objective_exact(params):
        """Exact objective for final evaluation."""
        if time.time() > start_time + deadline - safety_margin:
            raise TimeoutError()
            
        points = params.reshape(n, d)
        R = compute_R(points)
        return -R

    def sqp_maximize_min_dist(points, max_dist_bound, max_iter=200):
        """
        SQP solver to maximize minimum pairwise distance subject to all distances <= max_dist_bound.
        Returns (optimized_points, achieved_min_dist, success).
        """
        if time.time() > start_time + deadline - safety_margin:
            return points, 0.0, False
        
        n_points = points.shape[0]
        
        # Objective: maximize minimum distance
        # We minimize negative of a soft-min approximation for differentiability
        def obj_min_neg(x):
            pts = x.reshape(n_points, d)
            dists = pdist(pts)
            if len(dists) == 0:
                return 0.0
            # Use soft-min with moderate temperature
            p = 10.0
            d_scaled = -dists / p
            max_d = np.max(d_scaled)
            soft_min = -p * (max_d + np.log(np.sum(np.exp(d_scaled - max_d))))
            return -soft_min  # minimize negative = maximize
        
        # Constraints: all pairwise distances <= max_dist_bound
        # For SLSQP, we need inequality constraints: g(x) >= 0
        # So: max_dist_bound^2 - ||p_i - p_j||^2 >= 0
        
        def make_constraint(i, j):
            def constraint(x):
                pts = x.reshape(n_points, d)
                diff = pts[i] - pts[j]
                dist_sq = np.dot(diff, diff)
                return max_dist_bound**2 - dist_sq
            return constraint
        
        # Generate all pairwise constraints
        constraints = []
        for i in range(n_points):
            for j in range(i+1, n_points):
                constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
        
        # Initial guess
        x0 = points.flatten()
        
        try:
            result = minimize(
                obj_min_neg,
                x0,
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': max_iter, 'ftol': 1e-10}
            )
            
            if result.success:
                opt_points = result.x.reshape(n_points, d)
                dists = pdist(opt_points)
                if len(dists) > 0:
                    min_dist = np.min(dists)
                    max_dist = np.max(dists)
                    # Verify constraint satisfaction
                    if max_dist <= max_dist_bound * (1 + 1e-6):
                        return opt_points, min_dist, True
                return opt_points, 0.0, False
            else:
                return points, 0.0, False
        except Exception:
            return points, 0.0, False

    def sqp_bisection_refine(points, time_budget=30.0):
        """
        Use bisection on max_dist_bound to find optimal configuration.
        Start with a loose bound, tighten until infeasible, then refine.
        """
        if time.time() > start_time + deadline - safety_margin:
            return points
        
        current_points = points.copy()
        current_score = compute_R(current_points)
        dists = pdist(current_points)
        d_max = np.max(dists) if len(dists) > 0 else 1.0
        
        # Start with a bound slightly above current max
        bound = d_max * 1.05
        best_points = current_points.copy()
        best_score = current_score
        
        # Bisection: try to reduce the bound
        # We want to find the smallest bound where the problem is feasible
        # and gives a good score
        
        n_bisections = 8
        for i in range(n_bisections):
            if time.time() - start_time > deadline - safety_margin:
                break
            
            # Try to solve with current bound
            opt_pts, min_d, success = sqp_maximize_min_dist(current_points, bound, max_iter=150)
            
            if success and min_d > 0:
                score = compute_R(opt_pts)
                if score > best_score:
                    best_score = score
                    best_points = opt_pts.copy()
                    current_points = opt_pts.copy()
                
                # Try to tighten the bound
                new_bound = bound * 0.95
                # Check if new bound is still feasible (quick check)
                dists_new = pdist(opt_pts)
                if len(dists_new) > 0 and np.max(dists_new) <= new_bound * 1.01:
                    bound = new_bound
                else:
                    # Bound is too tight, relax slightly
                    bound = bound * 0.98
            else:
                # Infeasible, relax bound
                bound = bound * 1.05
        
        return best_points

    def simulated_annealing(initial_points, n_iter=50000, t_start=1.0, t_end=0.001, seed=42):
        """
        Simulated Annealing to explore the global landscape.
        """
        rng = np.random.RandomState(seed)
        current_points = initial_points.copy()
        current_R = compute_R(current_points)
        best_points = current_points.copy()
        best_R = current_R
        
        n_iter = min(n_iter, 200000)
        
        for i in range(n_iter):
            if time.time() > start_time + deadline - safety_margin:
                break
            
            alpha = i / n_iter
            temperature = t_start * (t_end / t_start) ** alpha
            
            n_moves = 1 if rng.rand() < 0.7 else 2
            indices = rng.choice(n, size=n_moves, replace=False)
            
            scale = 0.5 * temperature + 0.01
            new_points = current_points.copy()
            for idx in indices:
                new_points[idx] += rng.normal(0, scale, size=d)
            
            new_R = compute_R(new_points)
            
            if new_R > current_R or rng.rand() < np.exp((new_R - current_R) / max(temperature, 1e-10)):
                current_points = new_points
                current_R = new_R
                if current_R > best_R:
                    best_R = current_R
                    best_points = current_points.copy()
        
        return best_points, best_R

    def local_refine(points):
        """
        Local refinement using smooth surrogate, then SQP, then exact objective.
        """
        if time.time() > start_time + deadline - safety_margin:
            return points
        
        params = points.flatten()
        
        # Step 1: Optimize smooth surrogate with L-BFGS-B (coarse)
        best_params = params.copy()
        best_smooth_val = objective_smooth(best_params, p_min=50, p_max=50)
        
        try:
            result = minimize(
                lambda x: objective_smooth(x, p_min=50, p_max=50),
                best_params, 
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            if result.success:
                smooth_candidate = result.x
                if objective_smooth(smooth_candidate, p_min=50, p_max=50) < best_smooth_val:
                    best_params = smooth_candidate
                    best_smooth_val = objective_smooth(best_params, p_min=50, p_max=50)
        except Exception:
            pass
        
        # Step 2: Refine with finer smooth surrogate
        if time.time() > start_time + deadline - safety_margin:
            return best_params.reshape(n, d)
        
        try:
            result = minimize(
                lambda x: objective_smooth(x, p_min=20, p_max=20),
                best_params, 
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if result.success:
                smooth_candidate = result.x
                if objective_smooth(smooth_candidate, p_min=20, p_max=20) < best_smooth_val:
                    best_params = smooth_candidate
                    best_smooth_val = objective_smooth(best_params, p_min=20, p_max=20)
        except Exception:
            pass
        
        # Step 3: Even finer smooth surrogate
        if time.time() > start_time + deadline - safety_margin:
            return best_params.reshape(n, d)
        
        try:
            result = minimize(
                lambda x: objective_smooth(x, p_min=5, p_max=5),
                best_params, 
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            if result.success:
                smooth_candidate = result.x
                if objective_smooth(smooth_candidate, p_min=5, p_max=5) < best_smooth_val:
                    best_params = smooth_candidate
                    best_smooth_val = objective_smooth(best_params, p_min=5, p_max=5)
        except Exception:
            pass
        
        # Step 4: SQP bisection refinement (key improvement)
        if time.time() > start_time + deadline - safety_margin:
            return best_params.reshape(n, d)
        
        sqp_points = sqp_bisection_refine(best_params.reshape(n, d), time_budget=20.0)
        sqp_score = compute_R(sqp_points)
        smooth_score = compute_R(best_params.reshape(n, d))
        
        if sqp_score > smooth_score:
            best_params = sqp_points.flatten()
        
        # Step 5: Powell on exact objective
        if time.time() > start_time + deadline - safety_margin:
            return best_params.reshape(n, d)
        
        best_val = objective_exact(best_params)
        try:
            result = minimize(
                objective_exact, 
                best_params, 
                method='Powell', 
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if result.fun < best_val:
                best_params = result.x
                best_val = result.fun
        except Exception:
            pass
        
        # Step 6: Nelder-Mead polish on exact objective
        if time.time() > start_time + deadline - safety_margin:
            return best_params.reshape(n, d)
        
        try:
            result = minimize(
                objective_exact, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < best_val:
                best_params = result.x
                best_val = result.fun
        except Exception:
            pass
        
        # Step 7: Final high-precision polish
        if time.time() > start_time + deadline - safety_margin:
            return best_params.reshape(n, d)
        
        try:
            result = minimize(
                objective_exact, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-16, 'fatol': 1e-16, 'adaptive': True}
            )
            if result.fun < best_val:
                best_params = result.x
                best_val = result.fun
        except Exception:
            pass
        
        return best_params.reshape(n, d)

    # Generate diverse initial points for SA
    def make_initial_points(seed):
        """Create various initial configurations."""
        rng = np.random.RandomState(seed)
        config_type = seed % 4
        
        if config_type == 0:
            # 4x4 grid
            x = np.linspace(-1.5, 1.5, 4)
            y = np.linspace(-1.5, 1.5, 4)
            X, Y = np.meshgrid(x, y)
            points = np.column_stack((X.ravel(), Y.ravel()))
        elif config_type == 1:
            # Random uniform in unit square
            points = rng.uniform(-1, 1, size=(n, d))
        elif config_type == 2:
            # Regular hexagonal packing approximation
            points = np.zeros((n, d))
            for i in range(n):
                row = i // 4
                col = i % 4
                points[i, 0] = col * 1.0 + (row % 2) * 0.5
                points[i, 1] = row * 0.866
            points -= points.mean(axis=0)
        else:
            # Random on circle
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            radii = rng.uniform(0.5, 1.5, size=n)
            points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        
        return points

    # Run multiple SA restarts with different initial conditions
    best_points = None
    best_R = 0.0
    
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > start_time + deadline - safety_margin:
            break
        
        init_points = make_initial_points(seed=42 + restart)
        
        # Run SA with this initial condition
        sa_points, sa_R = simulated_annealing(init_points, n_iter=50000, t_start=1.0, t_end=0.0005, seed=42 + restart)
        
        if sa_R > best_R:
            best_R = sa_R
            best_points = sa_points.copy()
        
        # Local refinement (includes SQP)
        refined_points = local_refine(sa_points)
        refined_R = compute_R(refined_points)
        
        if refined_R > best_R:
            best_R = refined_R
            best_points = refined_points.copy()
    
    # Final high-precision polish on the best result - multiple passes
    if best_points is not None:
        for polish_pass in range(3):
            if time.time() > start_time + deadline - safety_margin:
                break
            final_points = local_refine(best_points)
            final_R = compute_R(final_points)
            if final_R > best_R:
                best_points = final_points
                best_R = final_R
            else:
                break
    
    # Fallback if something went wrong
    if best_points is None or best_points.shape != (n, d):
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        X, Y = np.meshgrid(x, y)
        best_points = np.column_stack((X.ravel(), Y.ravel()))
    
    return best_points

# EVOLVE-BLOCK-END
