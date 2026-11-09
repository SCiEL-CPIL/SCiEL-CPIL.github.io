import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import time

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    
    # Time management
    start_time = time.time()
    deadline = start_time + 590  # Leave 40s margin for safety
    
    # Objective function to minimize (we minimize the negative of the metric)
    def objective(x):
        if time.time() > deadline:
            return np.inf
            
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        
        try:
            dists = pdist(pts)
        except Exception:
            return np.inf
            
        if np.any(~np.isfinite(dists)):
            return np.inf
            
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max == 0:
            return np.inf
            
        ratio = (d_min / d_max) ** 2
        return -ratio

    def icosahedron():
        """Return 12 vertices of regular icosahedron on unit sphere."""
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        norms = np.linalg.norm(verts, axis=1)
        return verts / norms[:, np.newaxis]

    def generate_symmetric_seeds():
        """Generate high-quality symmetric seeds based on icosahedron + 2 extra points."""
        ico = icosahedron()
        seeds = []
        
        # Case 1: Extra points on the z-axis (5-fold axis)
        v0 = ico[0]
        v0_norm = np.linalg.norm(v0)
        v0 = v0 / v0_norm
        
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
            ico_rotated = ico @ R.T
        else:
            ico_rotated = ico.copy()
            
        for r_extra in [0.0, 0.3, 0.5, 0.7, 0.9, 1.1]:
            p1 = np.array([0, 0, r_extra])
            p2 = np.array([0, 0, -r_extra])
            seed = np.vstack([ico_rotated, p1.reshape(1,3), p2.reshape(1,3)])
            seeds.append(seed)
            
        # Case 2: Extra points on a 3-fold axis
        c = (ico[0] + ico[1] + ico[10]) / 3.0
        c_norm = np.linalg.norm(c)
        c = c / c_norm
        
        z = np.array([0, 0, 1])
        axis = np.cross(c, z)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-10:
            axis = axis / axis_norm
            angle = np.arccos(np.dot(c, z))
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            ico_rotated_3 = ico @ R.T
        else:
            ico_rotated_3 = ico.copy()
            
        for r_extra in [0.0, 0.3, 0.5, 0.7, 0.9, 1.1]:
            p1 = np.array([0, 0, r_extra])
            p2 = np.array([0, 0, -r_extra])
            seed = np.vstack([ico_rotated_3, p1.reshape(1,3), p2.reshape(1,3)])
            seeds.append(seed)
            
        # Case 3: Extra points on a 2-fold axis
        m = (ico[0] + ico[1]) / 2.0
        m_norm = np.linalg.norm(m)
        m = m / m_norm
        
        z = np.array([0, 0, 1])
        axis = np.cross(m, z)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-10:
            axis = axis / axis_norm
            angle = np.arccos(np.dot(m, z))
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            ico_rotated_2 = ico @ R.T
        else:
            ico_rotated_2 = ico.copy()
            
        for r_extra in [0.0, 0.3, 0.5, 0.7, 0.9, 1.1]:
            p1 = np.array([0, 0, r_extra])
            p2 = np.array([0, 0, -r_extra])
            seed = np.vstack([ico_rotated_2, p1.reshape(1,3), p2.reshape(1,3)])
            seeds.append(seed)
            
        # Case 4: Random dipole directions
        np.random.seed(42)
        for _ in range(8):
            p1 = np.random.randn(3)
            p1 /= np.linalg.norm(p1)
            p2 = -p1
            seed = np.vstack([ico, p1.reshape(1,3), p2.reshape(1,3)])
            seeds.append(seed)

        return seeds

    best_x = None
    best_val = np.inf
    
    # Quick local refinement from best symmetric seeds
    seeds = generate_symmetric_seeds()
    seed_scores = []
    for i, seed in enumerate(seeds):
        x = seed.flatten()
        val = objective(x)
        seed_scores.append((val, i))
    seed_scores.sort()
    
    # Refine top 3 seeds with Nelder-Mead (faster than L-BFGS-B for non-smooth)
    for rank, (val, idx) in enumerate(seed_scores[:3]):
        if time.time() > deadline - 120:
            break
        x0 = seeds[idx].flatten()
        try:
            res_nm = minimize(
                objective, x0, method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11}
            )
            if res_nm.fun < best_val and np.isfinite(res_nm.fun):
                best_val = res_nm.fun
                best_x = res_nm.x
        except Exception:
            pass

    # Main strategy: Multiple DE runs with increasing budget
    bounds = [(-2, 2)] * (n * d)
    de_configs = [
        {'seed': 1, 'maxiter': 80, 'popsize': 20, 'tol': 0.005},
        {'seed': 42, 'maxiter': 100, 'popsize': 25, 'tol': 0.003},
        {'seed': 123, 'maxiter': 120, 'popsize': 30, 'tol': 0.002},
        {'seed': 777, 'maxiter': 150, 'popsize': 35, 'tol': 0.001},
    ]
    
    for cfg in de_configs:
        if time.time() > deadline - 60:
            break
        try:
            res_de = differential_evolution(
                objective, bounds,
                seed=cfg['seed'], maxiter=cfg['maxiter'],
                popsize=cfg['popsize'], tol=cfg['tol'],
                polish=False, init='sobol'
            )
            if res_de.fun < best_val and np.isfinite(res_de.fun):
                best_val = res_de.fun
                best_x = res_de.x
        except Exception:
            pass

    # Polish the best result with Nelder-Mead
    if best_x is not None and time.time() < deadline - 10:
        try:
            res_final = minimize(
                objective, best_x, method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if res_final.fun < best_val and np.isfinite(res_final.fun):
                best_val = res_final.fun
                best_x = res_final.x
        except Exception:
            pass

    # Fallback: use best seed if no optimization succeeded
    if best_x is None:
        best_x = seeds[seed_scores[0][1]].flatten()

    # Reshape and center the final result
    final_points = best_x.reshape((n, d))
    final_points = final_points - np.mean(final_points, axis=0)
    
    return final_points
