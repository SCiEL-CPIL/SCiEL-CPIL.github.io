import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    def soft_max(y, beta):
        # Stable log-sum-exp
        m = np.max(y)
        if m > 100: # overflow protection, though y are logs of dists
            return m + np.log(np.sum(np.exp(beta * (y - m)))) / beta
        return (1.0 / beta) * np.log(np.sum(np.exp(beta * y)))

    def soft_min(y, beta):
        m = np.min(y)
        if m < -100:
            return m - np.log(np.sum(np.exp(-beta * (y - m)))) / beta
        return -(1.0 / beta) * np.log(np.sum(np.exp(-beta * y)))

    def objective(x, beta):
        if time.time() > deadline:
            return np.inf
            
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        
        try:
            dists = pdist(pts)
        except Exception:
            return np.inf
            
        if np.any(~np.isfinite(dists)) or np.any(dists <= 1e-9):
            return np.inf
            
        log_dists = np.log(dists)
        # Minimize the range of log distances
        # This is equivalent to minimizing log(d_max) - log(d_min) = log(d_max/d_min)
        # Which is equivalent to maximizing d_min/d_max
        val = soft_max(log_dists, beta) - soft_min(log_dists, beta)
        return val

    def icosahedron():
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        norms = np.linalg.norm(verts, axis=1)
        return verts / norms[:, np.newaxis]

    def generate_seeds():
        seeds = []
        ico = icosahedron()
        
        # Seed 1: Icosahedron + 2 points on Z axis (5-fold symmetry)
        # Align ico[0] to Z
        v0 = ico[0]
        v0 = v0 / np.linalg.norm(v0)
        z = np.array([0, 0, 1])
        axis = np.cross(v0, z)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-10:
            axis = axis / axis_norm
            angle = np.arccos(np.dot(v0, z))
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            ico_rot = ico @ R.T
        else:
            ico_rot = ico.copy()
            
        for r in [0.0, 0.5, 0.8, 1.0, 1.2, 1.5]:
            p1 = np.array([0, 0, r])
            p2 = np.array([0, 0, -r])
            seeds.append(np.vstack([ico_rot, p1, p2]))

        # Seed 2: Icosahedron + 2 points on 3-fold axis
        # Face centroid of 0, 1, 10
        c = (ico[0] + ico[1] + ico[10]) / 3.0
        c = c / np.linalg.norm(c)
        axis = np.cross(c, z)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-10:
            axis = axis / axis_norm
            angle = np.arccos(np.dot(c, z))
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            ico_rot_3 = ico @ R.T
        else:
            ico_rot_3 = ico.copy()
            
        for r in [0.0, 0.5, 0.8, 1.0, 1.2, 1.5]:
            p1 = np.array([0, 0, r])
            p2 = np.array([0, 0, -r])
            seeds.append(np.vstack([ico_rot_3, p1, p2]))
            
        # Seed 3: Random points on sphere (Fibonacci-like or random)
        np.random.seed(42)
        for _ in range(5):
            pts = np.random.randn(n, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            # Add some noise to break symmetry
            pts += np.random.randn(n, 3) * 0.1
            seeds.append(pts)
            
        # Seed 4: Two tetrahedra? Or just random 14 points
        for _ in range(5):
            pts = np.random.randn(n, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            seeds.append(pts)
            
        return seeds

    seeds = generate_seeds()
    best_x = None
    best_val = np.inf
    
    # Beta annealing schedule
    betas = [10, 20, 50, 100, 200, 500, 1000]
    
    # Evaluate seeds with lowest beta first to get a good starting point
    # Then refine with higher betas
    
    # We will process each seed, and for each seed, we will anneal beta.
    # To save time, we might only do full annealing on the best few seeds.
    
    # Step 1: Evaluate all seeds with beta=10
    seed_results = []
    for i, seed in enumerate(seeds):
        x0 = seed.flatten()
        val = objective(x0, 10)
        if np.isfinite(val):
            seed_results.append((val, i, x0))
            
    seed_results.sort(key=lambda x: x[0])
    
    # Step 2: Refine the top seeds with annealing
    top_k = min(10, len(seed_results))
    for val, idx, x0 in seed_results[:top_k]:
        if time.time() > deadline - 60:
            break
            
        current_x = x0
        current_val = val
        
        for beta in betas:
            if time.time() > deadline - 10:
                break
                
            try:
                res = minimize(
                    objective, 
                    current_x, 
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if np.isfinite(res.fun) and res.fun < current_val:
                    current_val = res.fun
                    current_x = res.x
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_x = current_x
                else:
                    # If it didn't improve, keep current_x for next beta?
                    # Or break? Usually keep the best found so far for this seed.
                    pass
            except Exception:
                pass

    # Step 3: If we have time, do a few random restarts with annealing
    if time.time() < deadline - 60:
        np.random.seed(123)
        for _ in range(5):
            if time.time() > deadline - 30:
                break
            x0 = np.random.randn(n * d) * 0.5
            # Normalize to unit sphere roughly?
            pts = x0.reshape(n, d)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / (norms + 1e-9)
            x0 = pts.flatten()
            
            current_x = x0
            current_val = objective(x0, 10)
            
            for beta in betas:
                if time.time() > deadline - 10:
                    break
                try:
                    res = minimize(
                        objective, 
                        current_x, 
                        args=(beta,),
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    if np.isfinite(res.fun) and res.fun < current_val:
                        current_val = res.fun
                        current_x = res.x
                        
                        if current_val < best_val:
                            best_val = current_val
                            best_x = current_x
                except Exception:
                    pass

    if best_x is None:
        # Fallback to best seed
        if seed_results:
            best_x = seed_results[0][2]
        else:
            # Absolute fallback
            best_x = seeds[0].flatten()
            
    final_points = best_x.reshape((n, d))
    final_points = final_points - np.mean(final_points, axis=0)
    
    return final_points
