# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution, basinhopping

def _score(points):
    """Compute (d_min/d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = dists.min()
    d_max = dists.max()
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    
    Uses a multi-stage optimization:
    1. Differential evolution for global search
    2. Basin hopping with smooth surrogate
    3. Sequential refinement with increasing smoothness
    4. Final exact-objective polish
    """
    n = 14
    d = 3
    
    # Deadline guard
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 30.0
    
    best_points = None
    best_score = -1.0
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > (time_limit - safety_margin)
    
    def objective(points):
        """Returns (d_min/d_max)^2, higher is better."""
        if len(points) < 2:
            return 0.0
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = dists.min()
        d_max = dists.max()
        if d_max == 0:
            return 0.0
        if not np.isfinite(d_min) or not np.isfinite(d_max):
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def smooth_objective(points, beta=50.0):
        """
        Smooth surrogate: maximize soft_min(dists) / max(dists)
        """
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_max = dists.max()
        if d_max == 0 or not np.isfinite(d_max):
            return 0.0
        min_d = dists.min()
        shifted = -beta * (dists - min_d)
        max_shifted = np.max(shifted)
        log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        soft_min = min_d - log_sum / beta
        
        if not np.isfinite(soft_min) or soft_min <= 0:
            return 0.0
        
        ratio = soft_min / d_max
        if not np.isfinite(ratio):
            return 0.0
        return ratio * ratio

    def neg_smooth(x, beta=50.0):
        """Negative smooth objective for minimization."""
        points = x.reshape(n, d)
        return -smooth_objective(points, beta)

    def neg_objective(x):
        """Negative exact objective for minimization."""
        points = x.reshape(n, d)
        return -objective(points)

    # Initialization strategies
    def fibonacci_sphere(n_points):
        points = []
        for i in range(n_points):
            phi = np.arccos(1 - 2 * (i + 0.5) / n_points)
            theta = np.pi * (1 + 5**0.5) * i
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points.append([x, y, z])
        return np.array(points)
    
    def symmetric_start(n_points):
        phi = (1 + 5**0.5) / 2
        icosahedron = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        norms = np.linalg.norm(icosahedron, axis=1, keepdims=True)
        icosahedron /= norms
        
        p13 = np.array([1.0, 1.0, 1.0])
        p14 = np.array([-1.0, -1.0, -1.0])
        p13 /= np.linalg.norm(p13)
        p14 /= np.linalg.norm(p14)
        
        return np.vstack([icosahedron, [p13, p14]])
    
    def random_sphere(n_points, seed):
        rng = np.random.default_rng(seed)
        points = rng.standard_normal((n_points, 3))
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / norms
    
    def static_solution():
        pts = np.array([
            [ 0.000000,  0.000000,  1.000000],
            [ 0.000000, -1.000000,  0.000000],
            [ 0.000000,  1.000000,  0.000000],
            [ 1.000000,  0.000000,  0.000000],
            [-1.000000,  0.000000,  0.000000],
            [ 0.000000,  0.000000, -1.000000],
            [ 0.577350,  0.577350,  0.577350],
            [ 0.577350, -0.577350, -0.577350],
            [-0.577350,  0.577350, -0.577350],
            [-0.577350, -0.577350,  0.577350],
            [ 0.816497,  0.408248, -0.408248],
            [-0.816497,  0.408248,  0.408248],
            [ 0.408248, -0.816497,  0.408248],
            [-0.408248, -0.816497, -0.408248],
        ])
        return pts

    def parametric_points(params):
        """
        Generate 14 points from a low-dimensional parameterization.
        params = [eps1, eps2, z_amp, phase, rot_x, rot_y]
        - eps1: amplitude of cos(2*theta) radial perturbation
        - eps2: amplitude of cos(3*theta) radial perturbation  
        - z_amp: out-of-plane displacement amplitude
        - phase: phase offset for radial perturbations
        - rot_x, rot_y: rotation angles to break symmetry
        """
        eps1 = params[0]
        eps2 = params[1]
        z_amp = params[2]
        phase = params[3]
        rot_x = params[4]
        rot_y = params[5]
        
        theta = 2 * np.pi * np.arange(n) / n + phase
        r = 1.0 + eps1 * np.cos(2 * theta) + eps2 * np.cos(3 * theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = z_amp * np.sin(2 * theta)
        
        pts = np.column_stack([x, y, z])
        
        # Apply rotation
        cx, sx = np.cos(rot_x), np.sin(rot_x)
        cy, sy = np.cos(rot_y), np.sin(rot_y)
        
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        
        pts = pts @ Ry.T @ Rx.T
        return pts

    def neg_parametric(params):
        """Negative objective for parametric optimization."""
        pts = parametric_points(params)
        return -objective(pts)
    
    # Phase 0: Parametric low-dimensional optimization
    if not check_deadline():
        try:
            param_bounds = [(-0.3, 0.3), (-0.3, 0.3), (-0.5, 0.5), (0, 2*np.pi), (-np.pi/4, np.pi/4), (-np.pi/4, np.pi/4)]
            
            de_param = differential_evolution(
                neg_parametric,
                param_bounds,
                maxiter=200,
                popsize=20,
                seed=42,
                tol=1e-8,
                polish=True
            )
            
            if not check_deadline():
                best_params = de_param.x
                param_points = parametric_points(best_params)
                param_score = objective(param_points)
                
                if param_score > best_score:
                    best_score = param_score
                    best_points = param_points.copy()
                    
                # Also try local refinement from this starting point
                x0 = param_points.flatten()
                for beta in [100.0, 200.0, 500.0]:
                    if check_deadline():
                        break
                    res = minimize(
                        lambda x, b=beta: neg_smooth(x, b),
                        x0,
                        method='Powell',
                        options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    x0 = res.x
                
                if not check_deadline():
                    res_final = minimize(
                        neg_objective,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14}
                    )
                    opt_points = res_final.x.reshape(n, d)
                    opt_score = objective(opt_points)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = opt_points.copy()
                        
        except Exception:
            pass
    
    # Phase 1: Differential evolution for global optimization
    if not check_deadline():
        try:
            bounds = [(-1.0, 1.0)] * (n * d)
            de_result = differential_evolution(
                neg_smooth,
                bounds,
                args=(20.0,),
                maxiter=100,
                popsize=15,
                seed=42,
                tol=1e-6,
                polish=False
            )
            
            if de_result.success or not check_deadline():
                de_points = de_result.x.reshape(n, d)
                de_score = objective(de_points)
                
                if de_score > best_score:
                    best_score = de_score
                    best_points = de_points.copy()
                    
        except Exception:
            pass
    
    # Phase 2: Multi-start basin hopping + refinement
    init_strategies = []
    init_strategies.append(("fib_sphere", fibonacci_sphere(n)))
    init_strategies.append(("symmetric", symmetric_start(n)))
    init_strategies.append(("static", static_solution()))
    
    # Add parametric initializations with different parameters
    param_inits = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.05, 0.1, 0.3, 0.1, 0.05],
        [0.05, 0.1, 0.15, 0.5, -0.1, 0.1],
        [0.15, 0.05, 0.05, 0.7, 0.15, -0.1],
    ]
    for i, p in enumerate(param_inits):
        init_strategies.append((f"parametric_{i}", parametric_points(p)))
    
    for seed in range(12):
        init_strategies.append((f"rand_sphere_{seed}", random_sphere(n, seed)))
    
    # Add best DE result if found
    if best_points is not None:
        init_strategies.append(("de_best", best_points.copy()))
    
    for name, init_points in init_strategies:
        if check_deadline():
            break
        
        x0 = init_points.flatten()
        
        try:
            # Basin hopping with smooth surrogate
            result = basinhopping(
                lambda x: neg_smooth(x, beta=30.0),
                x0,
                minimizer_kwargs={
                    'method': 'Powell',
                    'options': {
                        'maxiter': 500,
                        'xatol': 1e-8,
                        'fatol': 1e-10
                    }
                },
                niter=20,
                stepsize=0.8,
                T=100.0,
                seed=42
            )
            
            if check_deadline():
                break
            
            # Sequential refinement with adaptive beta annealing
            x_current = result.x
            
            for beta in [50.0, 100.0, 200.0, 300.0, 500.0, 800.0, 1200.0]:
                if check_deadline():
                    break
                
                res = minimize(
                    lambda x, b=beta: neg_smooth(x, b),
                    x_current,
                    method='Powell',
                    options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13}
                )
                x_current = res.x
            
            if check_deadline():
                break
            
            # High-precision SLSQP polish with tight tolerances
            res_slsqp = minimize(
                lambda x: neg_smooth(x, beta=1200.0),
                x_current,
                method='SLSQP',
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            if check_deadline():
                break
            
            # Final exact-objective Nelder-Mead polish with very tight tolerance
            res_final = minimize(
                neg_objective,
                res_slsqp.x,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-13, 'fatol': 1e-15}
            )
            
            if check_deadline():
                break
            
            optimized_points = res_final.x.reshape(n, d)
            score = objective(optimized_points)
            
            if score > best_score:
                best_score = score
                best_points = optimized_points.copy()
                
        except Exception:
            continue
    
    # If no valid result, use Fibonacci sphere as fallback
    if best_points is None:
        best_points = fibonacci_sphere(n)
    
    # Verify the result is valid
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = fibonacci_sphere(n)
    
    # Ensure shape is correct
    if best_points.shape != (n, d):
        best_points = fibonacci_sphere(n)
    
    # Final validation
    if not np.isfinite(best_score) or best_score < 0:
        best_points = fibonacci_sphere(n)
        best_score = objective(best_points)
    
    # Final high-precision polish on the best result found
    # Use remaining time budget for iterative refinement
    if not check_deadline():
        try:
            x_best = best_points.flatten()
            remaining_time = time_limit - safety_margin - elapsed()
            
            # Adaptive multi-stage polish with time-aware budget
            polish_configs = [
                # (beta, method, maxiter, xatol/ftol, gtol)
                (800.0, 'SLSQP', 5000, 1e-15, 1e-13),
                (1200.0, 'SLSQP', 5000, 1e-15, 1e-14),
                (2000.0, 'SLSQP', 8000, 1e-16, 1e-14),
                (3000.0, 'SLSQP', 10000, 1e-16, 1e-15),
            ]
            
            # Allocate time: spend ~60% on SLSQP stages, 40% on Nelder-Mead
            slsqp_budget = remaining_time * 0.6
            nm_budget = remaining_time * 0.4
            
            for beta, method, maxiter, ftol, gtol in polish_configs:
                if check_deadline():
                    break
                
                stage_start = time.time()
                try:
                    res_polish = minimize(
                        lambda x, b=beta: neg_smooth(x, b),
                        x_best,
                        method=method,
                        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
                    )
                    stage_time = time.time() - stage_start
                    
                    # Check if we're running out of time
                    if elapsed() > time_limit - safety_margin - 5.0:
                        break
                    
                    x_best = res_polish.x
                    polished_score = objective(x_best.reshape(n, d))
                    
                    if polished_score > best_score:
                        best_score = polished_score
                        best_points = x_best.reshape(n, d).copy()
                        
                except Exception:
                    continue
            
            # Final Nelder-Mead exact objective polish with remaining time
            if not check_deadline() and nm_budget > 10.0:
                try:
                    # Use more iterations if we have more time
                    nm_maxiter = int(nm_budget * 500)  # ~500 iterations per second
                    nm_maxiter = max(5000, min(nm_maxiter, 50000))
                    
                    res_final2 = minimize(
                        neg_objective,
                        x_best,
                        method='Nelder-Mead',
                        options={'maxiter': nm_maxiter, 'xatol': 1e-14, 'fatol': 1e-16}
                    )
                    
                    if not check_deadline():
                        polished_points = res_final2.x.reshape(n, d)
                        polished_score = objective(polished_points)
                        
                        if polished_score > best_score:
                            best_score = polished_score
                            best_points = polished_points.copy()
                            
                except Exception:
                    pass
                    
            # Additional refinement: focus on critical pairs
            # Try to improve by slightly perturbing points that form min/max distance pairs
            if not check_deadline() and remaining_time > 30.0:
                try:
                    pts = best_points.copy()
                    dists = pdist(pts)
                    d_min_idx = np.argmin(dists)
                    d_max_idx = np.argmax(dists)
                    
                    # Convert pair indices to point indices
                    def dist_idx_to_pair(idx, n_pts):
                        # Invert the pdist index mapping
                        i = int((np.sqrt(8 * idx + 1) - 1) // 2)
                        j = i + 1 + int(idx - i * (2 * n_pts - i - 1) // 2)
                        return i, j
                    
                    min_i, min_j = dist_idx_to_pair(d_min_idx, n)
                    max_i, max_j = dist_idx_to_pair(d_max_idx, n)
                    
                    # Try small perturbations to increase min distance or decrease max distance
                    rng = np.random.default_rng(12345)
                    perturb_trials = 50
                    
                    for trial in range(perturb_trials):
                        if check_deadline():
                            break
                        
                        pts_trial = pts.copy()
                        
                        # Strategy 1: Move min-distance pair apart
                        if trial < perturb_trials // 2:
                            direction = pts_trial[min_i] - pts_trial[min_j]
                            norm_dir = np.linalg.norm(direction)
                            if norm_dir > 1e-10:
                                direction /= norm_dir
                                # Move points apart
                                pts_trial[min_i] += 0.01 * direction
                                pts_trial[min_j] -= 0.01 * direction
                        else:
                            # Strategy 2: Move max-distance pair closer
                            direction = pts_trial[max_i] - pts_trial[max_j]
                            norm_dir = np.linalg.norm(direction)
                            if norm_dir > 1e-10:
                                direction /= norm_dir
                                # Move points closer
                                pts_trial[max_i] -= 0.01 * direction
                                pts_trial[max_j] += 0.01 * direction
                        
                        trial_score = objective(pts_trial)
                        
                        if trial_score > best_score:
                            best_score = trial_score
                            best_points = pts_trial.copy()
                            pts = pts_trial.copy()  # Continue from improved solution
                            # Recalculate critical pairs
                            dists = pdist(pts)
                            d_min_idx = np.argmin(dists)
                            d_max_idx = np.argmax(dists)
                            min_i, min_j = dist_idx_to_pair(d_min_idx, n)
                            max_i, max_j = dist_idx_to_pair(d_max_idx, n)
                            
                except Exception:
                    pass
                    
        except Exception:
            pass
    
    return best_points

# EVOLVE-BLOCK-END
