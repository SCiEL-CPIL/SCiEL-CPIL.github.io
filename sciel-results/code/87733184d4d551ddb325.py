import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    deadline = time.time() + 500  # Safety margin for 600s limit

    def objective(points):
        # points is (14, 3)
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        # Penalty for not being on unit sphere to keep them well-spread
        # We want to maximize (d_min / d_max)^2
        # So minimize - (d_min / d_max)^2
        # Add a small penalty to keep points on sphere? 
        # Actually, if we start on sphere and use repulsion, they stay close.
        # Nelder-Mead might drift. 
        # Let's just use the raw metric. The optimal solution is on the sphere.
        return -((d_min / d_max) ** 2)

    def repulsion_step(points, step_size):
        # points: (n, 3)
        # Compute forces
        # dists is (n, n)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # Avoid division by zero
        np.fill_diagonal(dists_sq, 1.0)
        # Force F_i = sum_j (p_i - p_j) / dist^2
        # (p_i - p_j) is diff[i, j]
        # We want force on i from j.
        # F_i = sum_j diff[i, j] / dists_sq[i, j]
        # dists_sq is (n, n).
        # We need to broadcast dists_sq to (n, n, 1)
        forces = np.sum(diff / dists_sq[:, :, np.newaxis], axis=1)
        # Update
        new_points = points + step_size * forces
        # Normalize to unit sphere
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        new_points = new_points / norms
        return new_points

    best_points = None
    best_score = -np.inf

    # Try multiple initializations
    # 1. Fibonacci Sphere
    def fib_sphere(n):
        points = []
        for i in range(n):
            z = 1 - 2 * i / (n - 1)
            r = np.sqrt(1 - z**2)
            theta = np.pi * (1 + 5**0.5) * i
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)

    initializations = []
    initializations.append(fib_sphere(n))
    
    # 2. Random starts
    np.random.seed(42)
    for _ in range(5):
        pts = np.random.randn(n, d)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        initializations.append(pts)

    # 3. Symmetric guess: 2 poles + 12 ring
    pts_sym = np.zeros((n, 3))
    pts_sym[0] = [0, 0, 1]
    pts_sym[1] = [0, 0, -1]
    for k in range(12):
        theta = 2 * np.pi * k / 12
        pts_sym[2 + k] = [np.cos(theta), np.sin(theta), 0.5] # Start at h=0.5
    initializations.append(pts_sym)

    for init_pts in initializations:
        if time.time() > deadline - 60:
            break

        # Repulsion phase
        pts = init_pts.copy()
        step = 0.1
        for i in range(500):
            if time.time() > deadline - 30:
                break
            pts = repulsion_step(pts, step)
            step *= 0.99 # Annealing

        # Polish with Nelder-Mead
        try:
            res = minimize(objective, pts.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            pts_opt = res.x.reshape(n, d)
            # Normalize final result to unit sphere to be safe?
            # The metric is scale invariant, but returning points on unit sphere is standard.
            norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            pts_opt = pts_opt / norms
            
            score = -objective(pts_opt)
            if score > best_score:
                best_score = score
                best_points = pts_opt
        except Exception:
            continue

    if best_points is None:
        # Fallback to random
        np.random.seed(42)
        best_points = np.random.randn(n, d)
        best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)

    return best_points
