import numpy as np
from scipy.optimize import minimize
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    def objective(pts):
        """
        Calculate the squared ratio of max to min pairwise distance.
        """
        # pts: (N, 2)
        # Compute all pairwise squared distances to avoid sqrt where possible, 
        # but we need actual distances for the ratio.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # Mask diagonal
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        # Check for coincident points
        if np.any(d_sq < 1e-12):
            return 1e10
            
        d = np.sqrt(d_sq)
        d_min = d.min()
        d_max = d.max()
        
        if d_min < 1e-12:
            return 1e10
            
        return (d_max / d_min) ** 2

    def make_square_grid(n):
        side = int(np.ceil(n ** 0.5))
        grid = []
        for i in range(side):
            for j in range(side):
                grid.append([i, j])
        return np.array(grid[:n], dtype=float)

    def make_triangular_cluster(n):
        # Generate a large triangular lattice and pick n closest to origin
        # Triangular lattice basis vectors: v1 = (1, 0), v2 = (0.5, sqrt(3)/2)
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.5, np.sqrt(3) / 2])
        
        pts = []
        for i in range(-10, 11):
            for j in range(-10, 11):
                p = i * v1 + j * v2
                pts.append(p)
        pts = np.array(pts)
        
        # Center
        pts -= pts.mean(axis=0)
        
        # Sort by distance to center
        dists = np.linalg.norm(pts, axis=1)
        order = np.argsort(dists)
        return pts[order[:n]]

    def make_circular_optimized(n):
        # 1 center, 6 on ring 1, 9 on ring 2
        # We can optimize the radii and angles slightly
        pts = []
        pts.append([0, 0])
        r1 = 1.0
        for i in range(6):
            angle = 2 * np.pi * i / 6
            pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
        
        # Ring 2: 9 points
        # In a perfect hex lattice, the second ring is at distance 2*sqrt(3) ? No.
        # Distance to next nearest in hex lattice is sqrt(3) if min is 1?
        # Nearest neighbors: 6 at dist 1.
        # Next nearest: 6 at dist sqrt(3) (corners of hexagon) and 6 at dist 2?
        # Let's just place them evenly on a circle of radius r2 and optimize r2.
        r2 = 1.732 # approx sqrt(3)
        for i in range(9):
            angle = 2 * np.pi * i / 9 + np.pi / 9
            pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
            
        return np.array(pts, dtype=float)

    # Initial candidates
    candidates = [
        make_square_grid(N),
        make_triangular_cluster(N),
        make_circular_optimized(N)
    ]
    
    best_val = np.inf
    best_pts = candidates[0].copy()
    
    # Local optimization for each candidate
    for i, init_pts in enumerate(candidates):
        if time.time() > deadline:
            break
            
        # Try a few perturbations
        for trial in range(3):
            if time.time() > deadline:
                break
            if trial == 0:
                current = init_pts.copy()
            else:
                current = init_pts + np.random.randn(N, D) * 0.5
            
            try:
                # Use Powell's method, it handles non-smoothness reasonably well
                # maxfev is limited to ensure we don't hang
                result = minimize(
                    objective,
                    current,
                    method='Powell',
                    options={'maxiter': 2000, 'maxfev': 20000}
                )
                val = result.fun
                if val < best_val:
                    best_val = val
                    best_pts = result.x.copy()
            except Exception:
                pass

    # If we still have time, try Differential Evolution with tight bounds around the best found
    if time.time() < deadline - 30:
        # Center the best points
        center = best_pts.mean(axis=0)
        best_pts_centered = best_pts - center
        
        # Determine bounds based on the extent of the best points
        extents = best_pts_centered.max(axis=0) - best_pts_centered.min(axis=0)
        margin = 2.0
        lower = (best_pts_centered.min(axis=0) - margin).flatten()
        upper = (best_pts_centered.max(axis=0) + margin).flatten()
        
        bounds = list(zip(lower, upper))
        
        try:
            de_result = minimize(
                lambda x: objective(x.reshape(N, D)),
                best_pts_centered.flatten(),
                method='Powell', # DE is too slow for 150s if not careful, stick to local
                options={'maxiter': 5000, 'maxfev': 50000}
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_pts = de_result.x.reshape(N, D) + center
        except Exception:
            pass
            
    # Final check: ensure points are distinct
    # The objective function returns 1e10 if points coincide, so we should be safe.
    # However, numerical precision might cause issues.
    
    return best_pts
