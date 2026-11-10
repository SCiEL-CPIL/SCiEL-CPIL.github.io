import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580  # Leave 50s margin for 630s limit

    def score(pts):
        pts = pts - np.mean(pts, axis=0)
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(x):
        if time.time() > deadline:
            return np.inf
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        dists = pdist(pts)
        if len(dists) == 0:
            return np.inf
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return np.inf
        return -(d_min / d_max) ** 2

    def objective_gradient(x):
        """Compute objective and its gradient using automatic differentiation via finite differences."""
        if time.time() > deadline:
            return np.inf, np.zeros_like(x)
        
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        dists = pdist(pts)
        if len(dists) == 0:
            return np.inf, np.zeros_like(x)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return np.inf, np.zeros_like(x)
        
        val = -(d_min / d_max) ** 2
        
        # Compute gradient using finite differences
        grad = np.zeros_like(x)
        eps = 1e-7
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            obj_plus = objective(x_plus)
            if np.isinf(obj_plus):
                x_minus = x.copy()
                x_minus[i] -= eps
                obj_minus = objective(x_minus)
                if not np.isinf(obj_minus):
                    grad[i] = (obj_plus - obj_minus) / (2 * eps) if not np.isinf(obj_plus) else 0
                else:
                    grad[i] = 0
            else:
                x_minus = x.copy()
                x_minus[i] -= eps
                obj_minus = objective(x_minus)
                if np.isinf(obj_minus):
                    grad[i] = (obj_plus - val) / eps
                else:
                    grad[i] = (obj_plus - obj_minus) / (2 * eps)
        
        return val, grad

    def smooth_objective(x, temperature=0.05):
        """Smooth approximation using log-sum-exp for soft-min/soft-max"""
        if time.time() > deadline:
            return np.inf
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        dists = pdist(pts)
        if len(dists) == 0:
            return np.inf
        if np.any(dists < 0):
            return np.inf
        
        t = temperature
        # Clip distances to avoid numerical issues
        dists_clipped = np.clip(dists, 1e-10, 1e10)
        
        # Soft min: -t * log(sum(exp(-dists/t)))
        exp_neg = np.exp(-dists_clipped / t)
        # Scale to avoid overflow
        max_exp_neg = np.max(exp_neg)
        if max_exp_neg > 1e300:
            exp_neg = exp_neg / max_exp_neg
        log_sum_exp_neg = np.log(np.sum(exp_neg)) + np.log(max_exp_neg)
        soft_min = -t * log_sum_exp_neg
        
        # Soft max: t * log(sum(exp(dists/t)))
        exp_pos = np.exp(dists_clipped / t)
        max_exp_pos = np.max(exp_pos)
        if max_exp_pos > 1e300:
            exp_pos = exp_pos / max_exp_pos
        log_sum_exp_pos = np.log(np.sum(exp_pos)) + np.log(max_exp_pos)
        soft_max = t * log_sum_exp_pos
        
        if soft_max < 1e-12:
            return np.inf
        
        return -(soft_min / soft_max) ** 2

    def coulomb_objective(x, k=1.0):
        """Coulomb repulsion energy to spread points"""
        if time.time() > deadline:
            return np.inf
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        dists = pdist(pts)
        if np.any(dists < 1e-10):
            return 1e10
        # Minimize sum of 1/r^2 (strong repulsion at close range)
        return np.sum(1.0 / (dists ** 2))

    # Icosahedron vertices (12 points)
    phi = (1 + np.sqrt(5)) / 2
    ico12 = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico12 = ico12 / np.linalg.norm(ico12, axis=1, keepdims=True)

    best_points = None
    best_score = -np.inf

    # Generate initial configurations
    initial_configs = []
    rng = np.random.RandomState(42)

    # Config 1: Icosahedron + 2 points at center region
    cfg1 = np.vstack([ico12, np.array([[0, 0, 0.5], [0, 0, -0.5]])])
    initial_configs.append(cfg1)

    # Config 2: Icosahedron + 2 points near opposite poles
    cfg2 = np.vstack([ico12, np.array([[0, 0, 1.2], [0, 0, -1.2]])])
    initial_configs.append(cfg2)

    # Config 3: Random perturbations of icosahedron
    for _ in range(5):
        extra = rng.randn(2, 3) * 0.5
        cfg = np.vstack([ico12, extra])
        initial_configs.append(cfg)

    # Config 4: Slightly perturbed icosahedron + 2 points
    for _ in range(5):
        pert = ico12 + rng.randn(12, 3) * 0.1
        extra = rng.randn(2, 3) * 0.5
        cfg = np.vstack([pert, extra])
        initial_configs.append(cfg)

    # Config 5: Fibonacci sphere for 14 points
    def fibonacci_sphere(n_pts):
        indices = np.arange(0, n_pts, dtype=float) + 0.5
        phi_f = np.arccos(1 - 2 * indices / n_pts)
        theta = np.pi * (1 + 5**0.5) * indices
        x = np.cos(theta) * np.sin(phi_f)
        y = np.sin(theta) * np.sin(phi_f)
        z = np.cos(phi_f)
        return np.column_stack([x, y, z])

    fib14 = fibonacci_sphere(14)
    initial_configs.append(fib14)

    # Config 6: Fibonacci + perturbations
    for _ in range(5):
        pert = fib14 + rng.randn(14, 3) * 0.1
        initial_configs.append(pert)

    # Config 7: Cube + octahedron based
    cube8 = np.array([
        [1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],
        [-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]
    ], dtype=float)
    cube8 = cube8 / np.linalg.norm(cube8, axis=1, keepdims=True)
    octa6 = np.array([
        [1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]
    ], dtype=float)
    octa6_scaled = octa6 * 0.7
    cfg7 = np.vstack([cube8, octa6_scaled])
    initial_configs.append(cfg7)

    # Config 8: Random initializations
    for _ in range(10):
        random_init = rng.randn(n, d)
        initial_configs.append(random_init)

    # Stage 1: Coulomb repulsion to spread points
    for i, init in enumerate(initial_configs):
        if time.time() > deadline - 120:
            break
        
        init_flat = init.flatten()
        
        try:
            result_coulomb = minimize(
                coulomb_objective,
                init_flat,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if result_coulomb.fun < np.inf:
                # Stage 2: Optimize the ratio from Coulomb-spread configuration
                if time.time() > deadline - 90:
                    break
                
                result_nm = minimize(
                    objective,
                    result_coulomb.x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                        'adaptive': True
                    }
                )
                
                if result_nm.fun < np.inf:
                    pts = result_nm.x.reshape((n, d))
                    s = score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts
                    
                    # Stage 3: BFGS refinement
                    if time.time() < deadline - 30 and s > 0:
                        try:
                            result_bfgs = minimize(
                                objective,
                                result_nm.x,
                                method='BFGS',
                                options={
                                    'maxiter': 1000,
                                    'gtol': 1e-10
                                }
                            )
                            if result_bfgs.fun < np.inf:
                                pts_bfgs = result_bfgs.x.reshape((n, d))
                                s_bfgs = score(pts_bfgs)
                                if s_bfgs > best_score:
                                    best_score = s_bfgs
                                    best_points = pts_bfgs
                        except Exception:
                            pass
        except Exception:
            pass

    # If we still have time, do more random restarts with direct ratio optimization
    if time.time() < deadline - 60:
        for _ in range(20):
            if time.time() > deadline - 30:
                break
            random_init = rng.randn(n, d)
            init_flat = random_init.flatten()
            try:
                result = minimize(
                    objective,
                    init_flat,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1500,
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                        'adaptive': True
                    }
                )
                if result.fun < np.inf:
                    pts = result.x.reshape((n, d))
                    s = score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts
                    
                    # BFGS refinement
                    if time.time() < deadline - 20 and s > 0:
                        try:
                            result_bfgs = minimize(
                                objective,
                                result.x,
                                method='BFGS',
                                options={
                                    'maxiter': 500,
                                    'gtol': 1e-10
                                }
                            )
                            if result_bfgs.fun < np.inf:
                                pts_bfgs = result_bfgs.x.reshape((n, d))
                                s_bfgs = score(pts_bfgs)
                                if s_bfgs > best_score:
                                    best_score = s_bfgs
                                    best_points = pts_bfgs
                        except Exception:
                            pass
            except Exception:
                pass

    # Final polish: try L-BFGS-B from best solution
    if best_points is not None and time.time() < deadline - 10:
        try:
            best_flat = best_points.flatten()
            result_lbfgs = minimize(
                objective,
                best_flat,
                method='L-BFGS-B',
                options={
                    'maxiter': 2000,
                    'gtol': 1e-12
                }
            )
            if result_lbfgs.fun < np.inf:
                pts_lbfgs = result_lbfgs.x.reshape((n, d))
                s_lbfgs = score(pts_lbfgs)
                if s_lbfgs > best_score:
                    best_score = s_lbfgs
                    best_points = pts_lbfgs
        except Exception:
            pass

    # Final high-precision polish with tight tolerances
    if best_points is not None and time.time() < deadline - 5:
        try:
            best_flat = best_points.flatten()
            result_final = minimize(
                objective,
                best_flat,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-14,
                    'fatol': 1e-16,
                    'adaptive': True
                }
            )
            if result_final.fun < np.inf:
                pts_final = result_final.x.reshape((n, d))
                s_final = score(pts_final)
                if s_final > best_score:
                    best_score = s_final
                    best_points = pts_final
        except Exception:
            pass

    # Ensure we have a valid result
    if best_points is None:
        best_points = initial_configs[0]
    
    # Final normalization: center and scale
    best_points = best_points - np.mean(best_points, axis=0)
    norm_max = np.max(np.linalg.norm(best_points, axis=1))
    if norm_max > 0:
        best_points = best_points / norm_max * 100

    # Ensure finite
    if not np.all(np.isfinite(best_points)):
        best_points = np.zeros((n, d))

    return best_points
