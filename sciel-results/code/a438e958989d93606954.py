"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    deadline = time.time() + 120.0
    best_score = 0.0
    best_points = None
    
    # Start with hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            init_points = points[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.98
    
    # Local optimization using repulsion forces
    def score(pts):
        # Compute pairwise distances
        d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        min_pair = np.min(np.where(np.eye(N, dtype=bool), np.inf, d)) / 2.0
        min_boundary = np.min(1.0 - np.linalg.norm(pts, axis=1))
        return min(min_pair, min_boundary)
    
    def energy(pts):
        # Negative of score (we want to maximize score, so minimize negative)
        return -score(pts)
    
    def energy_grad(pts):
        # Gradient of energy (negative of gradient of score)
        # This is complex, so use numerical gradient or just use BFGS without gradient
        return None
    
    # Try multiple random restarts with small perturbations
    for restart in range(20):
        if time.time() > deadline - 30.0:
            break
        
        # Perturb initial points
        if restart == 0:
            pts = init_points.copy()
        else:
            pts = init_points + np.random.randn(N, 2) * 0.01
        
        # Optimize using BFGS
        result = minimize(
            lambda x: energy(x.reshape(N, 2)),
            pts.flatten(),
            method='BFGS',
            options={'maxiter': 500, 'gtol': 1e-8}
        )
        
        opt_pts = result.x.reshape(N, 2)
        
        # Check validity
        if not np.all(np.isfinite(opt_pts)):
            continue
        if np.any(np.linalg.norm(opt_pts, axis=1) >= 1.0):
            continue
        
        s = score(opt_pts)
        if s > best_score:
            best_score = s
            best_points = opt_pts.copy()
    
    if best_points is not None:
        return best_points
    return init_points
# EVOLVE-BLOCK-END
