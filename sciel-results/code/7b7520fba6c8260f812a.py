"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    def compute_exact_score(pts):
        """Compute the exact packing radius for given centers."""
        if len(pts) != N:
            return -1.0
        if not np.all(np.isfinite(pts)):
            return -1.0
        
        # Radial distances
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0):
            return -1.0
        
        # Pairwise distances
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = pts[i_idx] - pts[j_idx]
        pair_dists = np.linalg.norm(diffs, axis=1)
        
        min_pair_dist = np.min(pair_dists)
        min_radial_dist = 1.0 - np.max(dists)
        
        r = min(min_pair_dist / 2.0, min_radial_dist)
        return r

    def generate_hexagonal_initialization():
        """Generate a hexagonal lattice clipped to the unit circle with denser spacing search."""
        best_pts = None
        best_score = -1.0
        
        # Try a denser range of spacings
        for scale in np.linspace(0.85, 1.15, 13):
            spacing = 2.0 / (np.sqrt(N) * scale)
            
            # Generate grid points
            nx = int(2.0 / spacing) + 5
            ny = int(2.0 / (spacing * np.sqrt(3.0) / 2.0)) + 5
            
            # Center the grid
            x_coords = (np.arange(-nx//2, nx//2 + 1) + 0.5) * spacing
            y_coords = (np.arange(-ny//2, ny//2 + 1)) * spacing * np.sqrt(3.0) / 2.0
            
            pts_list = []
            for i, y in enumerate(y_coords):
                offset = 0.0 if i % 2 == 0 else spacing / 2.0
                for x in x_coords:
                    px = x + offset
                    py = y
                    if np.hypot(px, py) <= 1.0 - spacing / 2.0:
                        pts_list.append([px, py])
            
            if len(pts_list) < N:
                continue
                
            pts = np.array(pts_list)
            norms = np.linalg.norm(pts, axis=1)
            idx = np.argsort(norms)
            pts = pts[idx[:N]]
            
            score = compute_exact_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts
        
        if best_pts is None:
            best_pts = np.random.rand(N, 2) * 2 - 1
            norms = np.linalg.norm(best_pts, axis=1)
            best_pts = best_pts * (0.9 / norms)[:, None]
            best_score = compute_exact_score(best_pts)
            
        return best_pts, best_score

    def soft_min(x, y, beta):
        """
        Smooth approximation of min(x, y) using log-sum-exp for numerical stability.
        """
        if np.isscalar(x) and np.isscalar(y):
            m = min(x, y)
            # Use log-sum-exp trick for stability
            ex = np.exp(-beta * (x - m))
            ey = np.exp(-beta * (y - m))
            val = -1.0 / beta * (np.log(ex + ey) - np.log(2.0))
            return m - val
        return min(x, y)

    def create_objective(pts_init, beta):
        """
        Create the objective function for L-BFGS-B.
        """
        def objective(params):
            if time.time() > deadline:
                return 1e10
            
            pts = params.reshape(-1, 2)
            
            if not np.all(np.isfinite(pts)):
                return 1e10
            
            dists = np.linalg.norm(pts, axis=1)
            if np.any(dists > 1.0):
                return 1e10 + np.sum(np.maximum(dists - 1.0, 0.0))
            
            i_idx, j_idx = np.triu_indices(N, k=1)
            diffs = pts[i_idx] - pts[j_idx]
            pair_dists = np.linalg.norm(diffs, axis=1)
            
            min_pair_dist = np.min(pair_dists)
            min_radial_dist = 1.0 - np.max(dists)
            
            x = min_pair_dist / 2.0
            y = min_radial_dist
            
            val = soft_min(x, y, beta)
            
            return -val
        
        return objective

    # 1. Generate initial points
    pts, best_score = generate_hexagonal_initialization()
    
    # 2. Multi-stage optimization with increasing beta
    betas = [5.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
    
    current_pts = pts
    current_score = best_score
    
    for beta in betas:
        if time.time() > deadline - 15:
            break
            
        objective = create_objective(current_pts, beta)
        
        try:
            res = minimize(objective, current_pts.flatten(), method='L-BFGS-B',
                           bounds=[(-1.0, 1.0)] * (2 * N),
                           options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20})
            
            if res.success or res.fun < 0:
                new_pts = res.x.reshape(-1, 2)
                new_score = compute_exact_score(new_pts)
                
                if new_score > current_score:
                    current_pts = new_pts
                    current_score = new_score
        except Exception:
            pass

    # 3. Final high-precision polish
    if time.time() < deadline - 5:
        # Try multiple high-beta values for final polish
        for beta in [3000.0, 5000.0, 8000.0]:
            if time.time() > deadline - 3:
                break
            objective = create_objective(current_pts, beta)
            try:
                res = minimize(objective, current_pts.flatten(), method='L-BFGS-B',
                               bounds=[(-1.0, 1.0)] * (2 * N),
                               options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 30})
                
                if res.success or res.fun < 0:
                    new_pts = res.x.reshape(-1, 2)
                    new_score = compute_exact_score(new_pts)
                    
                    if new_score > current_score:
                        current_pts = new_pts
                        current_score = new_score
            except Exception:
                pass

    # Final validation
    current_pts = current_pts.reshape(-1, 2)
    norms = np.linalg.norm(current_pts, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        current_pts[mask] *= (1.0 / norms[mask])[:, np.newaxis]
        
    return current_pts
# EVOLVE-BLOCK-END
