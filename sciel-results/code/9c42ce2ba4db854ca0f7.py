import numpy as np
import scipy.optimize
import time
import itertools

N = 14
D = 3

def construct():
    start_time = time.time()
    time_limit = 120.0
    safety_margin = 5.0
    
    best_score = float('inf')
    best_points = None
    
    # Heuristic initialization: Icosahedron vertices + 2 points
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    # Normalize
    norms = np.linalg.norm(ico_vertices, axis=1, keepdims=True)
    ico_vertices = ico_vertices / norms
    
    # Add 2 points, e.g., centers of opposite faces or just random
    # For N=14, adding 2 to ico is a good start.
    # Let's add points along the z-axis or similar.
    # Actually, let's just use random restarts primarily, but include this as one seed.
    
    seeds = []
    seeds.append(ico_vertices.copy())
    # Add a couple of random seeds
    for _ in range(2):
        pts = np.random.randn(N, D)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        seeds.append(pts)
        
    def compute_score(points):
        # points: (N, D)
        # Compute pairwise distances
        # dist2[i,j] = ||p_i - p_j||^2
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        # Exclude diagonal
        np.fill_diagonal(dist2, np.inf)
        d_min = np.min(dist2)
        d_max = np.max(dist2)
        if d_min <= 1e-10:
            return float('inf')
        ratio = d_max / d_min
        return ratio * ratio

    def objective(flat_points):
        points = flat_points.reshape(N, D)
        # Normalize to unit sphere to keep scale fixed
        # This helps the optimizer focus on angular distribution
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        points = points / norms
        return compute_score(points)

    for seed in seeds:
        if time.time() - start_time > time_limit - safety_margin:
            break
            
        x0 = seed.flatten()
        
        # Try Powell method
        try:
            res = scipy.optimize.minimize(
                objective, 
                x0, 
                method='Powell', 
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            score = res.fun
            if score < best_score:
                best_score = score
                best_points = res.x.reshape(N, D)
                # Normalize best points for storage
                norms = np.linalg.norm(best_points, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                best_points = best_points / norms
        except Exception:
            pass

        # If we have time, try a local refinement with L-BFGS-B on the sphere?
        # Powell is usually sufficient for N=14.
        
        # Add more random restarts if time permits
        if time.time() - start_time < time_limit - 20.0:
             for _ in range(5):
                 if time.time() - start_time > time_limit - safety_margin:
                     break
                 pts = np.random.randn(N, D)
                 pts /= np.linalg.norm(pts, axis=1, keepdims=True)
                 x0 = pts.flatten()
                 try:
                     res = scipy.optimize.minimize(
                         objective, 
                         x0, 
                         method='Powell', 
                         options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
                     )
                     score = res.fun
                     if score < best_score:
                         best_score = score
                         best_points = res.x.reshape(N, D)
                         norms = np.linalg.norm(best_points, axis=1, keepdims=True)
                         norms[norms == 0] = 1.0
                         best_points = best_points / norms
                 except Exception:
                     pass

    if best_points is None:
        # Fallback: return the icosahedron + 2 random
        pts = seeds[0].copy()
        # Ensure 14 points
        if len(pts) < N:
            extra = np.random.randn(N - len(pts), D)
            extra /= np.linalg.norm(extra, axis=1, keepdims=True)
            pts = np.vstack([pts, extra])
        else:
            pts = pts[:N]
        return pts

    return best_points

# EVOLVE-BLOCK-START
# The code above is the implementation. 
# I will place the imports and function definition here as required by the structure.
# The prompt says "Keep the function names and signatures of the seed program."
# The seed program has `construct()`.
# I will wrap the logic in the EVOLVE-BLOCK.
# EVOLVE-BLOCK-END
