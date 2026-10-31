import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    def objective(pts):
        """
        Calculate the squared ratio of max to min pairwise distance.
        Using squared distances: (max_d / min_d)^2 = (max_d_sq / min_d_sq)
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # Mask diagonal
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        # Check for coincident points
        min_d_sq = np.min(d_sq)
        if min_d_sq < 1e-12:
            return 1e10
            
        max_d_sq = np.max(d_sq)
        
        return max_d_sq / min_d_sq

    def make_hexagonal_cluster(n):
        """
        Generate a hexagonal lattice cluster of n points centered at origin.
        """
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.5, np.sqrt(3) / 2])
        
        # Generate a large enough grid of lattice points
        pts = []
        for i in range(-5, 6):
            for j in range(-5, 6):
                p = i * v1 + j * v2
                pts.append(p)
        pts = np.array(pts)
        
        # Center the cluster
        pts -= pts.mean(axis=0)
        
        # Sort by distance to center and pick the closest n
        dists = np.linalg.norm(pts, axis=1)
        order = np.argsort(dists)
        return pts[order[:n]]

    # Initial best candidate from hexagonal cluster
    init_pts = make_hexagonal_cluster(N)
    best_val = objective(init_pts)
    best_pts = init_pts.copy()
    
    # Determine bounds for Differential Evolution
    # The optimal configuration for 16 points in 2D minimizing diameter ratio
    # is likely bounded. A hexagonal cluster of 19 points has a radius of ~3 units (if spacing 1).
    # Let's set bounds to [-5, 5] x [-5, 5] to be safe.
    bounds = [(-5.0, 5.0)] * (N * D)
    
    # Run Differential Evolution
    # We use a tight maxiter to ensure we finish in time, but DE is parallelizable in thought, 
    # though scipy's is sequential. 
    # popsize=20, maxiter=300 -> ~6000 evaluations. 
    # Each eval is ~16*16/2 = 128 subtractions/additions/muls. Very fast.
    try:
        de_result = differential_evolution(
            objective, 
            bounds, 
            maxiter=500, 
            popsize=20, 
            tol=1e-8,
            seed=42,
            polish=True  # Let DE do its own local polish at the end if possible, or we do it below
        )
        
        if de_result.fun < best_val:
            best_val = de_result.fun
            best_pts = de_result.x.reshape(N, D).copy()
    except Exception:
        pass

    # Final local polish using Powell on the best candidate found
    if time.time() < deadline - 10:
        try:
            # Reshape for minimize
            current = best_pts.copy()
            result = minimize(
                objective,
                current,
                method='Powell',
                options={'maxiter': 5000, 'maxfev': 50000}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_pts = result.x.copy()
        except Exception:
            pass

    # Ensure we return valid distinct points
    # If best_val is still very high, it might mean points coincided, but objective handles that.
    # The DE bounds prevent extreme values, so points should be distinct.
    
    return best_pts
