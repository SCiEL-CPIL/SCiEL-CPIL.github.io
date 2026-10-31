import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
DEADLINE = 110.0  # Safety margin for 120s limit

def construct():
    start_time = time.time()
    
    def voronoi_radii(pts):
        """
        Compute the inradius of the Voronoi cell for each point.
        pts: (N, 2) array of coordinates.
        Returns: (N,) array of radii.
        """
        x = pts[:, 0]
        y = pts[:, 1]
        
        # Distance to boundaries
        r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        
        try:
            tri = Delaunay(pts)
            # Build adjacency list
            neighbors = [[] for _ in range(N)]
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(3):
                        if i != j:
                            neighbors[simplex[i]].append(simplex[j])
        except Exception:
            # Fallback if Delaunay fails (e.g., points too close)
            # Use all pairs (slow but robust)
            radii = np.full(N, np.inf)
            for i in range(N):
                min_r = r_bound[i]
                for j in range(N):
                    if i != j:
                        dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        min_r = min(min_r, dist / 2.0)
                radii[i] = min_r
            return radii

        radii = np.copy(r_bound)
        for i in range(N):
            min_r = r_bound[i]
            for j in neighbors[i]:
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                min_r = min(min_r, dist / 2.0)
            radii[i] = min_r
        
        return radii

    def objective(z):
        pts = z.reshape(-1, 2)
        radii = voronoi_radii(pts)
        return -np.sum(radii)

    def hexagonal_init(n_circles, perturb=0.0):
        x = np.zeros(n_circles)
        y = np.zeros(n_circles)
        
        cols = 6
        rows = 5
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= n_circles:
                    break
                x_offset = 0.5 if row % 2 == 1 else 0.0
                x[idx] = (col + 0.5 + x_offset) / (cols + 0.5)
                y[idx] = (row + 0.5) / (rows + 0.5)
                idx += 1
        
        while idx < n_circles:
            x[idx] = 0.5
            y[idx] = 0.5
            idx += 1
            
        if perturb > 0:
            rng = np.random.default_rng(999)
            x += rng.uniform(-perturb, perturb, n_circles)
            y += rng.uniform(-perturb, perturb, n_circles)
            
        x = np.clip(x, 0.01, 0.99)
        y = np.clip(y, 0.01, 0.99)
        
        return np.column_stack([x, y])

    def grid_init(n_circles, perturb=0.0):
        side = int(np.ceil(np.sqrt(n_circles)))
        x = np.zeros(n_circles)
        y = np.zeros(n_circles)
        
        for i in range(n_circles):
            row = i // side
            col = i % side
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            
        if perturb > 0:
            rng = np.random.default_rng(888)
            x += rng.uniform(-perturb, perturb, n_circles)
            y += rng.uniform(-perturb, perturb, n_circles)
            
        x = np.clip(x, 0.01, 0.99)
        y = np.clip(y, 0.01, 0.99)
        
        return np.column_stack([x, y])

    def random_init(rng):
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        return np.column_stack([x, y])

    best_score = 0.0
    best_pts = None
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Multi-start local optimization
    # We optimize positions only. Radii are derived.
    
    bounds = [(0, 1)] * (2 * N)
    
    # Hexagonal starts
    for k in range(10):
        if time.time() - start_time > DEADLINE - 20:
            break
        pts0 = hexagonal_init(N, perturb=0.02 * k)
        z0 = pts0.flatten()
        
        try:
            res = minimize(
                objective, 
                z0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            pts_opt = res.x.reshape(-1, 2)
            # Clip to ensure strict boundary
            pts_opt[:, 0] = np.clip(pts_opt[:, 0], 1e-5, 1-1e-5)
            pts_opt[:, 1] = np.clip(pts_opt[:, 1], 1e-5, 1-1e-5)
            
            radii = voronoi_radii(pts_opt)
            score = np.sum(radii)
            
            if score > best_score:
                best_score = score
                best_pts = pts_opt.copy()
        except Exception:
            continue

    # Grid starts
    for k in range(5):
        if time.time() - start_time > DEADLINE - 20:
            break
        pts0 = grid_init(N, perturb=0.01 * k)
        z0 = pts0.flatten()
        
        try:
            res = minimize(
                objective, 
                z0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            pts_opt = res.x.reshape(-1, 2)
            pts_opt[:, 0] = np.clip(pts_opt[:, 0], 1e-5, 1-1e-5)
            pts_opt[:, 1] = np.clip(pts_opt[:, 1], 1e-5, 1-1e-5)
            
            radii = voronoi_radii(pts_opt)
            score = np.sum(radii)
            
            if score > best_score:
                best_score = score
                best_pts = pts_opt.copy()
        except Exception:
            continue

    # Random starts
    for k in range(15):
        if time.time() - start_time > DEADLINE - 20:
            break
        pts0 = random_init(rng)
        z0 = pts0.flatten()
        
        try:
            res = minimize(
                objective, 
                z0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            pts_opt = res.x.reshape(-1, 2)
            pts_opt[:, 0] = np.clip(pts_opt[:, 0], 1e-5, 1-1e-5)
            pts_opt[:, 1] = np.clip(pts_opt[:, 1], 1e-5, 1-1e-5)
            
            radii = voronoi_radii(pts_opt)
            score = np.sum(radii)
            
            if score > best_score:
                best_score = score
                best_pts = pts_opt.copy()
        except Exception:
            continue

    # Phase 2: Refinement with Nelder-Mead (derivative free, robust to non-smoothness)
    if best_pts is not None:
        if time.time() - start_time < DEADLINE - 10:
            z0 = best_pts.flatten()
            try:
                res = minimize(
                    objective, 
                    z0, 
                    method='Nelder-Mead', 
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                pts_opt = res.x.reshape(-1, 2)
                pts_opt[:, 0] = np.clip(pts_opt[:, 0], 1e-5, 1-1e-5)
                pts_opt[:, 1] = np.clip(pts_opt[:, 1], 1e-5, 1-1e-5)
                
                radii = voronoi_radii(pts_opt)
                score = np.sum(radii)
                
                if score > best_score:
                    best_score = score
                    best_pts = pts_opt.copy()
            except Exception:
                pass

    # Final Fallback
    if best_pts is None:
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
        best_pts = np.column_stack([x, y])

    # Construct final result
    radii = voronoi_radii(best_pts)
    
    # Add small epsilon to radii? No, they are exact max.
    # Ensure they are positive
    radii = np.maximum(radii, 1e-6)
    
    result = np.column_stack([best_pts[:, 0], best_pts[:, 1], radii])
    
    return result

if __name__ == "__main__":
    res = construct()
    print(np.sum(res[:, 2]))
