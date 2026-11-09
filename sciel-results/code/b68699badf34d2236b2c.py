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

    def smooth_objective(x, temperature=0.1):
        """Smooth approximation using soft-min/soft-max"""
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
        # Use log-sum-exp style smoothing
        t = temperature
        soft_min = -np.log(np.sum(np.exp(-dists / t))) * t
        soft_max = np.log(np.sum(np.exp(dists / t))) * t
        if soft_max < 1e-12:
            return np.inf
        return -(soft_min / soft_max) ** 2

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

    # Generate several initial configurations
    initial_configs = []

    # Config 1: Icosahedron + 2 points at center (will be pushed out)
    cfg1 = np.vstack([ico12, np.array([[0, 0, 0.5], [0, 0, -0.5]])])
    initial_configs.append(cfg1)

    # Config 2: Icosahedron + 2 points near opposite poles
    cfg2 = np.vstack([ico12, np.array([[0, 0, 1.2], [0, 0, -1.2]])])
    initial_configs.append(cfg2)

    # Config 3: Icosahedron + 2 points at random positions
    rng = np.random.RandomState(42)
    for _ in range(5):
        extra = rng.randn(2, 3) * 0.5
        cfg = np.vstack([ico12, extra])
        initial_configs.append(cfg)

    # Config 4: Slightly perturbed icosahedron + 2 points
    for _ in range(3):
        pert = ico12 + rng.randn(12, 3) * 0.1
        extra = rng.randn(2, 3) * 0.5
        cfg = np.vstack([pert, extra])
        initial_configs.append(cfg)

    # Config 5: 14 points roughly evenly distributed on sphere
    # Use a fibonacci sphere for 14 points
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
    for _ in range(3):
        pert = fib14 + rng.randn(14, 3) * 0.1
        initial_configs.append(pert)

    # Config 7: Cube + octahedron based configuration
    cube8 = np.array([
        [1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],
        [-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]
    ], dtype=float)
    cube8 = cube8 / np.linalg.norm(cube8, axis=1, keepdims=True)
    octa6 = np.array([
        [1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]
    ], dtype=float)
    # Scale octahedron to fit between cube vertices
    octa6_scaled = octa6 * 0.7
    cfg7 = np.vstack([cube8, octa6_scaled])
    initial_configs.append(cfg7)

    # Config 8: Truncated octahedron approximation
    trunc_pts = []
    # 8 hexagonal faces, 6 square faces
    # Approximate with 14 points
    for _ in range(3):
        random_init = rng.randn(n, d)
        initial_configs.append(random_init)

    # Optimize each initial configuration
    for i, init in enumerate(initial_configs):
        if time.time() > deadline - 45:
            break
        
        init_flat = init.flatten()
        
        # First pass: Nelder-Mead
        try:
            result_nm = minimize(
                objective,
                init_flat,
                method='Nelder-Mead',
                options={
                    'maxiter': 1500,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if result_nm.fun < np.inf:
                pts = result_nm.x.reshape((n, d))
                s = score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts
                
                # Second pass: BFGS refinement from Nelder-Mead result
                if time.time() < deadline - 30:
                    try:
                        result_bfgs = minimize(
                            objective,
                            result_nm.x,
                            method='BFGS',
                            options={
                                'maxiter': 500,
                                'gtol': 1e-8
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

    # If we still have time, do more random restarts
    if best_points is None:
        best_points = initial_configs[0]
    
    if time.time() < deadline - 60:
        for _ in range(15):
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
                        'maxiter': 800,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
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
                                    'maxiter': 300,
                                    'gtol': 1e-8
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
                    'maxiter': 1000,
                    'gtol': 1e-10
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

    # Final normalization: center and scale
    best_points = best_points - np.mean(best_points, axis=0)
    norm_max = np.max(np.linalg.norm(best_points, axis=1))
    if norm_max > 0:
        best_points = best_points / norm_max * 100

    # Ensure finite
    if not np.all(np.isfinite(best_points)):
        best_points = np.zeros((n, d))

    return best_points
