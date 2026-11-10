# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    
    Uses a constrained optimization approach where points are constrained to the unit sphere
    and the objective minimizes the variance of pairwise distances, encouraging all distances
    to be as equal as possible.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    d = 3
    
    # Precompute indices for upper triangle of distance matrix
    iu = np.triu_indices(n, k=1)
    n_pairs = len(iu[0])
    
    def pairwise_distances(pts):
        """Compute all pairwise distances from a flat array."""
        pts = pts.reshape(n, d)
        diff = pts[iu[0]] - pts[iu[1]]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        return dists

    def objective(x):
        """
        Minimize the variance of pairwise distances.
        x is a flat array of shape (n*d,).
        """
        dists = pairwise_distances(x)
        mean_d = np.mean(dists)
        if mean_d < 1e-10:
            return 1e10
        return np.sum((dists - mean_d)**2)

    def sphere_constraints():
        """
        Constraints: each point must lie on the unit sphere.
        """
        cons = []
        for i in range(n):
            def con_func(x, idx=i):
                pts = x.reshape(n, d)
                return np.sum(pts[idx]**2) - 1.0
            cons.append({'type': 'eq', 'fun': con_func})
        return cons

    best_pts = None
    best_score = -np.inf
    
    # Run multiple trials with different random initializations
    np.random.seed(42)
    
    # Strategy 1: Random points on sphere
    for trial in range(30):
        x0 = np.random.randn(n, d)
        norms = np.linalg.norm(x0, axis=1, keepdims=True)
        x0 = x0 / norms
        x0_flat = x0.flatten()
        
        result = minimize(
            objective, 
            x0_flat, 
            method='SLSQP',
            constraints=sphere_constraints(),
            options={'maxiter': 2000, 'ftol': 1e-12}
        )
        
        if result.success or result.fun < 1e-6:
            pts = result.x.reshape(n, d)
            if np.all(np.isfinite(pts)):
                dists = pairwise_distances(result.x)
                if np.min(dists) > 1e-10 and np.max(dists) > 1e-10:
                    score = (np.min(dists) / np.max(dists)) ** 2
                    if score > best_score:
                        best_score = score
                        best_pts = pts
    
    # Strategy 2: Start from known good configurations (e.g., icosahedron-based)
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(ico_verts, axis=1, keepdims=True)
    ico_verts = ico_verts / norms
    
    for pole_config in range(5):
        angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
        theta = angles[pole_config]
        extra1 = np.array([np.sin(theta), np.cos(theta), 0])
        extra2 = np.array([-np.sin(theta), -np.cos(theta), 0])
        
        x0 = np.vstack([ico_verts, extra1, extra2])
        norms = np.linalg.norm(x0, axis=1, keepdims=True)
        x0 = x0 / norms
        x0_flat = x0.flatten()
        
        result = minimize(
            objective, 
            x0_flat, 
            method='SLSQP',
            constraints=sphere_constraints(),
            options={'maxiter': 3000, 'ftol': 1e-12}
        )
        
        if result.success or result.fun < 1e-6:
            pts = result.x.reshape(n, d)
            if np.all(np.isfinite(pts)):
                dists = pairwise_distances(result.x)
                if np.min(dists) > 1e-10 and np.max(dists) > 1e-10:
                    score = (np.min(dists) / np.max(dists)) ** 2
                    if score > best_score:
                        best_score = score
                        best_pts = pts
    
    # Strategy 3: Use differential evolution to find good initial points, then polish with SLSQP
    def de_objective(flat_params):
        thetas = flat_params[0::2]
        phis = flat_params[1::2]
        
        x = np.sin(thetas) * np.cos(phis)
        y = np.sin(thetas) * np.sin(phis)
        z = np.cos(thetas)
        pts = np.column_stack([x, y, z])
        
        dists = pairwise_distances(pts.flatten())
        mean_d = np.mean(dists)
        if mean_d < 1e-10:
            return 1e10
        return np.sum((dists - mean_d)**2)
    
    bounds_de = [(0, np.pi)] * n + [(0, 2*np.pi)] * n
    
    de_result = differential_evolution(
        de_objective,
        bounds_de,
        seed=42,
        maxiter=100,
        tol=1e-8,
        popsize=15,
        polish=False
    )
    
    if de_result.success:
        thetas = de_result.x[0::2]
        phis = de_result.x[1::2]
        x = np.sin(thetas) * np.cos(phis)
        y = np.sin(thetas) * np.sin(phis)
        z = np.cos(thetas)
        pts_de = np.column_stack([x, y, z])
        x0_flat = pts_de.flatten()
        
        result = minimize(
            objective, 
            x0_flat, 
            method='SLSQP',
            constraints=sphere_constraints(),
            options={'maxiter': 3000, 'ftol': 1e-12}
        )
        
        if result.success or result.fun < 1e-6:
            pts = result.x.reshape(n, d)
            if np.all(np.isfinite(pts)):
                dists = pairwise_distances(result.x)
                if np.min(dists) > 1e-10 and np.max(dists) > 1e-10:
                    score = (np.min(dists) / np.max(dists)) ** 2
                    if score > best_score:
                        best_score = score
                        best_pts = pts
    
    # Fallback if optimization failed
    if best_pts is None:
        np.random.seed(123)
        best_pts = np.random.randn(n, d)
        norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
        best_pts = best_pts / norms
    
    return best_pts


# EVOLVE-BLOCK-END
