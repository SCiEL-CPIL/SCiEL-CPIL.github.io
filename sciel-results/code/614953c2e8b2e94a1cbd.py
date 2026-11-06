"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _score(points):
    """Compute the raw packing score."""
    if len(points) != N:
        return -1.0
    # Boundary violations
    radii = np.sqrt(np.sum(points**2, axis=1))
    boundary = 1.0 - np.max(radii)
    if boundary < 0:
        return -1.0
    # Pairwise distances
    n = len(points)
    d = np.zeros((n, n))
    for i in range(n):
        d[i, i+1:] = np.sqrt(np.sum((points[i] - points[i+1:])**2, axis=1))
        d[i+1:, i] = d[i, i+1:]
    # Minimum pairwise distance
    min_dist = np.min(d[np.triu_indices(n, k=1)])
    return min(min_dist / 2.0, boundary)


def _hexagonal_init():
    """Generate a hexagonal lattice packing as initial guess."""
    # Try to fit hexagonal packing
    best_points = None
    best_score = -1
    for r in np.linspace(0.1, 0.25, 150):
        # Hexagonal grid spacing
        dx = 2 * r
        dy = 2 * r * np.sqrt(3) / 2
        # Generate points
        points = []
        for row in range(-15, 16):
            for col in range(-15, 16):
                x = col * dx + (row % 2) * dx / 2
                y = row * dy
                if np.sqrt(x*x + y*y) <= 1 - r:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            # Sort by distance from center, take closest N
            dists = np.sqrt(np.sum(points**2, axis=1))
            idx = np.argsort(dists)
            points = points[idx[:N]]
            score = _score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
    if best_points is None:
        # Fallback: random
        best_points = np.random.randn(N, 2) * 0.5
    return best_points


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initial hexagonal packing
    points = _hexagonal_init()
    best_points = points.copy()
    best_score = _score(points)
    
    # Optimize using L-BFGS-B with multiple restarts
    def objective(x):
        pts = x.reshape(-1, 2)
        # Negative score (we minimize)
        s = _score(pts)
        return -s
    
    for restart in range(5):
        if time.time() > deadline - 20:
            break
        
        if restart == 0:
            x0 = points.ravel().copy()
        else:
            # Perturb best solution
            noise = 0.01 * np.random.randn(N, 2)
            x0 = (best_points + noise).ravel()
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            x_opt = result.x
            pts_opt = x_opt.reshape(-1, 2)
            score_opt = _score(pts_opt)
            
            if score_opt > best_score:
                best_score = score_opt
                best_points = pts_opt.copy()
        except Exception:
            continue
    
    # Polish: gradient-based refinement
    if time.time() < deadline - 10:
        def energy(pts):
            """Compute negative score with penalty for violations."""
            radii = np.sqrt(np.sum(pts**2, axis=1))
            boundary_viol = np.sum(np.maximum(0, radii - 1.0)**2)
            n = len(pts)
            min_dist = 1e10
            for i in range(n):
                for j in range(i+1, n):
                    d = np.sqrt(np.sum((pts[i] - pts[j])**2))
                    if d < min_dist:
                        min_dist = d
            s = min(min_dist / 2.0, 1.0 - np.max(radii))
            return -s + 100.0 * boundary_viol
        
        for polish_iter in range(1000):
            if time.time() > deadline - 5:
                break
            x = best_points.ravel()
            # Numerical gradient
            eps = 1e-7
            grad = np.zeros_like(x)
            for i in range(len(x)):
                xp = x.copy()
                xm = x.copy()
                xp[i] += eps
                xm[i] -= eps
                grad[i] = (energy(xp.reshape(-1, 2)) - energy(xm.reshape(-1, 2))) / (2 * eps)
            
            # Simple gradient descent
            step = 0.001
            x_new = x - step * grad
            pts_new = x_new.reshape(-1, 2)
            score_new = _score(pts_new)
            
            if score_new > best_score:
                best_score = score_new
                best_points = pts_new.copy()
            else:
                step *= 0.5
                if step < 1e-10:
                    break
    
    return best_points
# EVOLVE-BLOCK-END
