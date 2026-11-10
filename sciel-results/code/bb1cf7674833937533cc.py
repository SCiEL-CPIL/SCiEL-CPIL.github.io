# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    
    Uses a penalty-based optimization where all pairwise distances are encouraged
    to be equal. The objective minimizes the sum of squared deviations from the
    mean distance, with points constrained to the unit sphere.
    
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
    
    # Strategy 1: Random points on sphere with multiple restarts
    for trial in range(50):
        x0 = np.random.randn(n, d)
        norms = np.linalg.norm(x0, axis=1, keepdims=True)
        x0 = x0 / norms
        x0_flat = x0.flatten()
        
        result = minimize(
            objective, 
            x0_flat, 
            method='SLSQP',
            constraints=sphere_constraints(),
            options={'maxiter': 5000, 'ftol': 1e-14}
        )
        
        pts = result.x.reshape(n, d)
        if np.all(np.isfinite(pts)):
            dists = pairwise_distances(result.x)
            if np.min(dists) > 1e-10 and np.max(dists) > 1e-10:
                score = (np.min(dists) / np.max(dists)) ** 2
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()
    
    # Strategy 2: Start from icosahedron-based configurations
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(ico_verts, axis=1, keepdims=True)
    ico_verts = ico_verts / norms
    
    # Try multiple configurations of the two extra points
    for config_idx in range(20):
        # Generate random positions for the two extra points
        np.random.seed(config_idx * 100 + 42)
        extra1 = np.random.randn(3)
        extra1 = extra1 / np.linalg.norm(extra1)
        extra2 = np.random.randn(3)
        extra2 = extra2 / np.linalg.norm(extra2)
        
        x0 = np.vstack([ico_verts, extra1, extra2])
        norms = np.linalg.norm(x0, axis=1, keepdims=True)
        x0 = x0 / norms
        x0_flat = x0.flatten()
        
        result = minimize(
            objective, 
            x0_flat, 
            method='SLSQP',
            constraints=sphere_constraints(),
            options={'maxiter': 5000, 'ftol': 1e-14}
        )
        
        pts = result.x.reshape(n, d)
        if np.all(np.isfinite(pts)):
            dists = pairwise_distances(result.x)
            if np.min(dists) > 1e-10 and np.max(dists) > 1e-10:
                score = (np.min(dists) / np.max(dists)) ** 2
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()
    
    # Strategy 3: Local refinement with small perturbations
    if best_pts is not None:
        for refine_trial in range(20):
            np.random.seed(refine_trial * 7 + 999)
            perturbation = np.random.randn(n, d) * 0.01
            x0 = best_pts + perturbation
            norms = np.linalg.norm(x0, axis=1, keepdims=True)
            x0 = x0 / norms
            x0_flat = x0.flatten()
            
            result = minimize(
                objective, 
                x0_flat, 
                method='SLSQP',
                constraints=sphere_constraints(),
                options={'maxiter': 5000, 'ftol': 1e-14}
            )
            
            pts = result.x.reshape(n, d)
            if np.all(np.isfinite(pts)):
                dists = pairwise_distances(result.x)
                if np.min(dists) > 1e-10 and np.max(dists) > 1e-10:
                    score = (np.min(dists) / np.max(dists)) ** 2
                    if score > best_score:
                        best_score = score
                        best_pts = pts.copy()
    
    # Fallback if optimization failed
    if best_pts is None:
        np.random.seed(123)
        best_pts = np.random.randn(n, d)
        norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
        best_pts = best_pts / norms
    
    return best_pts


# EVOLVE-BLOCK-END
