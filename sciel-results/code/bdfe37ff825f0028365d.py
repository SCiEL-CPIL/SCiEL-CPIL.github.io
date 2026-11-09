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

    # Optimize each initial configuration using Nelder-Mead
    for i, init in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        init_flat = init.flatten()
        
        try:
            result = minimize(
                objective,
                init_flat,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
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
        except Exception:
            pass

    # If we still have time, do a few more random restarts
    if best_points is None:
        best_points = initial_configs[0]
    
    if time.time() < deadline - 60:
        for _ in range(10):
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
                        'maxiter': 1000,
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
