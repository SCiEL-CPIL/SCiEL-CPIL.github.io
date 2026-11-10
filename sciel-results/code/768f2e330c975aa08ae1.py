# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def moving_sofa_2d() -> np.ndarray:
    # Grid definition
    x_grid = np.arange(-3.0, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_points = len(X_flat)
    
    start_time = time.time()
    time_limit = 620.0
    deadline = start_time + time_limit
    
    def check_inside(x, y, a_deg):
        theta = -np.deg2rad(a_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        
        wx = X_flat - x
        wy = Y_flat - y
        
        u = c * wx - s * wy
        v = s * wx + c * wy
        
        in_r1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        in_r2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        
        return in_r1 | in_r2
    
    def get_placements(path):
        """Generate all placements from a path including endpoints and interpolations."""
        full_path = np.vstack([
            [[0.0, 0.0, 0.0]],
            path,
            [[-2.0, 0.0, 90.0]]
        ])
        
        placements = []
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i + 1]
            
            da = p2[2] - p1[2]
            m = max(1, int(np.ceil(abs(da))))
            
            ts = np.linspace(0, 1, m + 1)
            for t in ts:
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                a = p1[2] + t * (p2[2] - p1[2])
                placements.append((x, y, a))
        
        # Remove duplicate consecutive placements
        unique_placements = []
        prev = None
        for p in placements:
            if prev is None or abs(p[0]-prev[0]) > 1e-10 or abs(p[1]-prev[1]) > 1e-10 or abs(p[2]-prev[2]) > 1e-10:
                unique_placements.append(p)
            prev = p
        
        return unique_placements
    
    def evaluate_path(path):
        """Compute the hard score: number of sofa points inside all placements."""
        placements = get_placements(path)
        
        mask = np.ones(n_points, dtype=bool)
        
        for x, y, a in placements:
            inside = check_inside(x, y, a)
            mask &= inside
            
            if not mask.any():
                return 0.0
        
        return np.sum(mask) * 0.0001
    
    def evaluate_path_soft(path, beta=20.0):
        """Compute a smooth surrogate score using soft min of margins."""
        placements = get_placements(path)
        n_placements = len(placements)
        
        if n_placements == 0:
            return 0.0
        
        # For each point, compute the minimum margin across all placements
        # margin > 0 means inside, margin < 0 means outside
        # soft_min = -beta * log(sum(exp(-margin_j / beta)))
        
        # We'll accumulate log-sum-exp for numerical stability
        log_sum_exp = np.full(n_points, -np.inf, dtype=np.float64)
        
        for x, y, a in placements:
            theta = -np.deg2rad(a)
            c = np.cos(theta)
            s = np.sin(theta)
            
            wx = X_flat - x
            wy = Y_flat - y
            
            u = c * wx - s * wy
            v = s * wx + c * wy
            
            # Distance to boundary of each rectangle (positive = inside)
            # For r1: [0,1] x [-3,1]
            d1_u = np.minimum(u, 1.0 - u)
            d1_v = np.minimum(v + 3.0, 1.0 - v)
            dist_r1 = np.minimum(d1_u, d1_v)
            
            # For r2: [-3,1] x [0,1]
            d2_u = np.minimum(u + 3.0, 1.0 - u)
            d2_v = np.minimum(v, 1.0 - v)
            dist_r2 = np.minimum(d2_u, d2_v)
            
            # Distance to the L-shaped corridor boundary
            in_r1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
            in_r2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
            
            # Compute margin
            margin = np.zeros(n_points, dtype=np.float64)
            
            both = in_r1 & in_r2
            margin[both] = np.maximum(dist_r1[both], dist_r2[both])
            
            only_r1 = in_r1 & ~in_r2
            margin[only_r1] = dist_r1[only_r1]
            
            only_r2 = in_r2 & ~in_r1
            margin[only_r2] = dist_r2[only_r2]
            
            neither = ~in_r1 & ~in_r2
            d_to_r1 = np.maximum(
                np.maximum(0.0 - u[neither], u[neither] - 1.0),
                np.maximum(-3.0 - v[neither], v[neither] - 1.0)
            )
            d_to_r2 = np.maximum(
                np.maximum(-3.0 - u[neither], u[neither] - 1.0),
                np.maximum(0.0 - v[neither], v[neither] - 1.0)
            )
            margin[neither] = -np.minimum(d_to_r1, d_to_r2)
            
            # Update log-sum-exp
            current = -margin / beta
            
            max_val = np.maximum(log_sum_exp, current)
            log_sum_exp = max_val + np.log(np.exp(log_sum_exp - max_val) + np.exp(current - max_val))
        
        # soft_min = -beta * log_sum_exp
        soft_min = -beta * log_sum_exp
        
        # Use a smooth step function with sharper transition
        safe = 1.0 / (1.0 + np.exp(-soft_min * 10.0))
        
        return np.sum(safe) * 0.0001
    
    # Initial seed path
    seed_path = np.array(
        [
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ],
        dtype=np.float64,
    )
    
    best_path = seed_path.copy()
    best_score = evaluate_path(best_path)
    
    def make_path_from_params(params):
        """Convert parameter vector to path.
        Parameters: for each of 8 waypoints, (x, y, a). Angles are variable."""
        path = []
        for i in range(8):
            path.append([params[3*i], params[3*i+1], params[3*i+2]])
        return np.array(path, dtype=np.float64)
    
    # Initial parameters: x, y, angle for each of 8 waypoints
    current_params = np.array([
        -0.05, 0.05, 10.0,
        -0.15, 0.15, 20.0,
        -0.25, 0.25, 30.0,
        -0.3, 0.3, 40.0,
        -0.45, 0.25, 50.0,
        -0.55, 0.2, 60.0,
        -0.65, 0.22, 75.0,
        -0.72, 0.28, 86.0
    ])
    
    def nelder_mead(f, x0, max_iter=200, tol=1e-6, time_limit_check=None, step_size=0.1):
        """Nelder-Mead optimization with time limit."""
        n = len(x0)
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initial simplex
        simplex = [x0.copy()]
        fvals = [f(x0)]
        
        for i in range(n):
            xi = x0.copy()
            xi[i] += step_size
            simplex.append(xi)
            fvals.append(f(xi))
        
        best_params = x0.copy()
        best_val = min(fvals)
        
        for iteration in range(max_iter):
            if time_limit_check is not None and time.time() > time_limit_check:
                break
            
            # Sort by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if abs(fvals[0] - fvals[-1]) < tol:
                break
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = f(xr)
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = f(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = f(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = f(simplex[i])
            
            # Track best
            if fvals[0] < best_val:
                best_val = fvals[0]
                best_params = simplex[0].copy()
        
        return best_params, best_val
    
    # Phase 1: Optimize with soft objective (smoother landscape) with adaptive beta
    def objective_soft(params):
        path = make_path_from_params(params)
        return -evaluate_path_soft(path, beta=15.0)
    
    if time.time() < deadline - 60:
        # Run soft optimization with decreasing beta for progressive sharpening
        for beta in [30.0, 15.0, 8.0]:
            if time.time() > deadline - 40:
                break
            
            def obj_beta(params):
                path = make_path_from_params(params)
                return -evaluate_path_soft(path, beta=beta)
            
            best_params_soft, best_val_soft = nelder_mead(
                obj_beta, current_params,
                max_iter=80, tol=1e-5,
                time_limit_check=deadline - 40,
                step_size=0.08
            )
            
            path_soft = make_path_from_params(best_params_soft)
            score_soft = evaluate_path(path_soft)
            
            if score_soft > best_score:
                best_score = score_soft
                best_path = path_soft
            current_params = best_params_soft
    
    # Phase 2: Refine with hard objective
    def objective_hard(params):
        path = make_path_from_params(params)
        return -evaluate_path(path)
    
    if time.time() < deadline - 30:
        best_params_hard, best_val_hard = nelder_mead(
            objective_hard, current_params,
            max_iter=150, tol=1e-7,
            time_limit_check=deadline - 30,
            step_size=0.03
        )
        
        path_hard = make_path_from_params(best_params_hard)
        score_hard = evaluate_path(path_hard)
        
        if score_hard > best_score:
            best_score = score_hard
            best_path = path_hard
            current_params = best_params_hard
    
    # Phase 3: Random restarts with hard objective
    np.random.seed(42)
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        # Perturb around current best with varying scales
        perturbation_scale = 0.1 + 0.1 * (restart % 3)
        perturbed = current_params + np.random.normal(0, perturbation_scale, size=len(current_params))
        
        bp, bv = nelder_mead(
            objective_hard, perturbed,
            max_iter=60, tol=1e-6,
            time_limit_check=deadline - 15,
            step_size=0.05
        )
        
        path_r = make_path_from_params(bp)
        score_r = evaluate_path(path_r)
        
        if score_r > best_score:
            best_score = score_r
            best_path = path_r
            current_params = bp
    
    # Phase 4: Fine-tune the best solution with coordinate descent
    if time.time() < deadline - 10:
        # Coordinate descent on each parameter
        for coord_pass in range(3):
            if time.time() > deadline - 5:
                break
            
            for i in range(len(current_params)):
                if time.time() > deadline - 5:
                    break
                
                # Try small steps in each direction
                for step in [0.01, -0.01, 0.005, -0.005]:
                    candidate = current_params.copy()
                    candidate[i] += step
                    path_c = make_path_from_params(candidate)
                    score_c = evaluate_path(path_c)
                    
                    if score_c > best_score:
                        best_score = score_c
                        best_path = path_c
                        current_params = candidate
    
    return best_path

# EVOLVE-BLOCK-END
