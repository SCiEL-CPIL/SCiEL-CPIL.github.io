"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n_opt = N - 2

    def objective_reduced(x):
        pts = np.zeros((N, D))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = x.reshape(n_opt, D)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        dists = dists[np.triu_indices(N, k=1)]
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min < 1e-9:
            return 1e9
        return (d_max / d_min)**2

    def make_hex_init():
        hex_pts = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i + 0.5 * j
                y = np.sqrt(3) / 2 * j
                hex_pts.append([x, y])
        hex_pts = np.array(hex_pts)
        dists_to_origin = np.sum(hex_pts**2, axis=1)
        idx = np.argsort(dists_to_origin)[:N]
        pts = hex_pts[idx]
        
        min_d = 1e9
        p1, p2 = 0, 1
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_d:
                    min_d = d
                    p1, p2 = i, j
                    
        pts = pts - pts[p1]
        dx, dy = pts[p2]
        d = np.sqrt(dx**2 + dy**2)
        pts = pts / d
        theta = np.arctan2(dy, dx)
        c, s = np.cos(-theta), np.sin(-theta)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T
        
        mask = np.ones(N, dtype=bool)
        mask[p1] = False
        mask[p2] = False
        reduced_pts = pts[mask]
        return reduced_pts.flatten()

    best_x = None
    best_score = 1e9
    
    # Primary init: Hexagonal
    x0_hex = make_hex_init()
    
    # Inits list
    inits = [x0_hex]
    
    # Add some random inits
    for _ in range(5):
        x0_rand = np.random.uniform(-1, 1, (n_opt, D)).flatten()
        inits.append(x0_rand)
        
    for x_init in inits:
        if time.time() > deadline:
            break
            
        # Add small noise to hex init to break symmetry if stuck
        if x_init is x0_hex:
             x_init = x_init + np.random.normal(0, 0.01, x_init.shape)
             
        try:
            res = scipy.optimize.minimize(objective_reduced, x_init, method='Powell', 
                                          options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-7})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x
        except Exception:
            pass

    if best_x is None:
        # Fallback to grid
        side = 4
        grid = list(np.meshgrid(np.arange(side), np.arange(side)))
        pts = np.array([(i, j) for i in grid[0].flatten() for j in grid[1].flatten()][:N], dtype=float)
        return pts
        
    pts = np.zeros((N, D))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = best_x.reshape(n_opt, D)
    
    return pts
# EVOLVE-BLOCK-END
