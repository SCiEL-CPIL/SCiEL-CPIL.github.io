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
    1. Structured initialization from known good configurations
    2. Differential evolution for global search
    3. Basin hopping with smooth surrogate
    4. Sequential refinement with adaptive beta annealing
    5. Final exact-objective polish with high precision
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
    
    def icosahedron_plus_two():
        """12 icosahedron vertices + 2 points at (±1,±1,±1)/√3 directions"""
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
    
    def tetrahedral_symmetry():
        """Try a configuration with tetrahedral symmetry considerations"""
        # Start with icosahedron and perturb the two extra points
        pts = icosahedron_plus_two()
        # Apply small random perturbations to break degeneracy
        rng = np.random.default_rng(123)
        pts = pts + rng.standard_normal((14, 3)) * 0.01
        return pts
    
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
        params = [eps1, eps2, z_amp, phase, rot_x, rot_y, radial_scale]
        - eps1: amplitude of cos(2*theta) radial perturbation
        - eps2: amplitude of cos(3*theta) radial perturbation  
        - z_amp: out-of-plane displacement amplitude
        - phase: phase offset for radial perturbations
        - rot_x, rot_y: rotation angles to break symmetry
        - radial_scale: overall radial scaling
        """
        eps1 = params[0]
        eps2 = params[1]
        z_amp = params[2]
        phase = params[3]
        rot_x = params[4]
        rot_y = params[5]
        radial_scale = params[6]
        
        theta = 2 * np.pi * np.arange(n) / n + phase
        r = radial_scale * (1.0 + eps1 * np.cos(2 * theta) + eps2 * np.cos(3 * theta))
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
    
    # Phase 0: Structured initialization and quick local refinement
    init_strategies = []
    init_strategies.append(("fib_sphere", fibonacci_sphere(n)))
    init_strategies.append(("icosahedron_plus_two", icosahedron_plus_two()))
    init_strategies.append(("tetrahedral", tetrahedral_symmetry()))
    init_strategies.append(("static", static_solution()))
    
    for seed in range(8):
        init_strategies.append((f"rand_sphere_{seed}", random_sphere(n, seed)))
    
    # Quick local refinement from each initial strategy
    for name, init_points in init_strategies:
        if check_deadline():
            break
        
        x0 = init_points.flatten()
        
        try:
            # Quick Powell refinement with moderate beta
            res = minimize(
                lambda x: neg_smooth(x, beta=30.0),
                x0,
                method='Powell',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if check_deadline():
                break
            
            opt_points = res.x.reshape(n, d)
            score = objective(opt_points)
            
            if score > best_score:
                best_score = score
                best_points = opt_points.copy()
                
        except Exception:
            continue
    
    # Phase 1: Parametric low-dimensional optimization
    if not check_deadline():
        try:
            param_bounds = [
                (-0.3, 0.3),  # eps1
                (-0.3, 0.3),  # eps2
                (-0.5, 0.5),  # z_amp
                (0, 2*np.pi), # phase
                (-np.pi/4, np.pi/4), # rot_x
                (-np.pi/4, np.pi/4), # rot_y
                (0.5, 1.5)    # radial_scale
            ]
            
            de_param = differential_evolution(
                neg_parametric,
                param_bounds,
                maxiter=150,
                popsize=15,
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
                    
                # Local refinement from parametric result
                x0 = param_points.flatten()
                for beta in [50.0, 100.0]:
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
    
    # Phase 2: Differential evolution for global optimization
    if not check_deadline():
        try:
            bounds = [(-1.0, 1.0)] * (n * d)
            de_result = differential_evolution(
                neg_smooth,
                bounds,
                args=(20.0,),
                maxiter=80,
                popsize=12,
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
    
    # Phase 3: Multi-start basin hopping + adaptive refinement
    # Build initialization list from all previous best results
    init_strategies = []
    if best_points is not None:
        init_strategies.append(("best_so_far", best_points.copy()))
    
    # Add some random initializations for diversity
    for seed in range(6):
        init_strategies.append((f"rand_{seed}", random_sphere(n, seed + 100)))
    
    for name, init_points in init_strategies:
        if check_deadline():
            break
        
        x0 = init_points.flatten()
        
        try:
            # Basin hopping with smooth surrogate
            result = basinhopping(
                lambda x: neg_smooth(x, beta=25.0),
                x0,
                minimizer_kwargs={
                    'method': 'Powell',
                    'options': {
                        'maxiter': 400,
                        'xatol': 1e-9,
                        'fatol': 1e-11
                    }
                },
                niter=15,
                stepsize=0.7,
                T=80.0,
                seed=42
            )
            
            if check_deadline():
                break
            
            # Adaptive beta annealing: start low, increase gradually
            x_current = result.x
            betas = [40.0, 60.0, 80.0, 120.0, 180.0, 250.0, 350.0, 500.0]
            
            for beta in betas:
                if check_deadline():
                    break
                
                res = minimize(
                    lambda x, b=beta: neg_smooth(x, b),
                    x_current,
                    method='Powell',
                    options={'maxiter': 1500, 'xatol': 1e-11, 'fatol': 1e-13}
                )
                x_current = res.x
            
            if check_deadline():
                break
            
            # SLSQP polish with high beta
            res_slsqp = minimize(
                lambda x: neg_smooth(x, beta=500.0),
                x_current,
                method='SLSQP',
                options={'maxiter': 3000, 'ftol': 1e-15}
            )
            
            if check_deadline():
                break
            
            # Final exact-objective Nelder-Mead polish with high precision
            res_final = minimize(
                neg_objective,
                res_slsqp.x,
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-13, 'fatol': 1e-15}
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
    if not np.all(np.isfinite(best_points)):
        best_points = fibonacci_sphere(n)
    
    # Ensure shape is correct
    if best_points.shape != (n, d):
        best_points = fibonacci_sphere(n)
    
    # Final validation
    if not np.isfinite(best_score) or best_score < 0:
        best_points = fibonacci_sphere(n)
    
    return best_points

# EVOLVE-BLOCK-END
