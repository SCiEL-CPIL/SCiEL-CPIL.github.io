import numpy as np
import time
from scipy.optimize import minimize

N = 13

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin
    
    # Initial guess: regular 13-gon (unit circle)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r0 = np.ones(N)
    
    # Precompute triangle index pairs
    tri_idx = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                tri_idx.append((i, j, k))
    tri_idx = np.array(tri_idx)  # shape (286, 3)
    
    def compute_score(r):
        """Compute min_triangle_area / convex_hull_area given radii."""
        pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        
        # Convex hull area via shoelace (points are in order)
        x = pts[:, 0]
        y = pts[:, 1]
        hull_area = 0.5 * abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))
        
        if hull_area < 1e-12:
            return -1e10  # penalty
        
        # All triangle areas
        i, j, k = tri_idx[:, 0], tri_idx[:, 1], tri_idx[:, 2]
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        v1 = pj - pi
        v2 = pk - pi
        areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        
        min_area = np.min(areas)
        
        return min_area / hull_area
    
    def objective(r):
        """We minimize negative score."""
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            return -1e10
        return -compute_score(r)
    
    # Initial evaluation
    best_score = compute_score(r0)
    best_r = r0.copy()
    
    # Optimize
    try:
        result = minimize(objective, r0, method='L-BFGS-B', 
                         bounds=[(0.1, 10.0)] * N,
                         options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if result.fun < 0:  # valid result
            score = -result.fun
            if score > best_score:
                best_score = score
                best_r = result.x.copy()
    except Exception:
        pass
    
    # Try multiple random restarts if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        for restart in range(20):
            if time.time() - start_time > deadline - 5:
                break
            # Random perturbation of radii
            r_init = r0 * (1 + 0.2 * np.random.randn(N))
            r_init = np.clip(r_init, 0.1, 10.0)
            
            try:
                result = minimize(objective, r_init, method='L-BFGS-B',
                                 bounds=[(0.1, 10.0)] * N,
                                 options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                if result.fun < 0:
                    score = -result.fun
                    if score > best_score:
                        best_score = score
                        best_r = result.x.copy()
            except Exception:
                pass
    
    # Final points
    final_r = best_r
    pts = np.column_stack([final_r * np.cos(angles), final_r * np.sin(angles)])
    
    return pts

# EVOLVE-BLOCK-START
# (kept for compatibility, but construct is defined above)
# EVOLVE-BLOCK-END
