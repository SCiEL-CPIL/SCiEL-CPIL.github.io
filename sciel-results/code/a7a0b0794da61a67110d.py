import numpy as np
import time
import math
from scipy.optimize import minimize

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def compute_radius(centers):
        """
        Computes the max radius for the given centers.
        Returns 0 if invalid.
        """
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        # Check bounds
        if np.any(np.abs(centers) > 0.5 + 1e-9):
            return 0.0
        
        # Wall distance
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        # Pair distance
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        # Set diagonal to inf
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = np.min(dist_sq)
        min_pair = np.sqrt(min_dist_sq)
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def hexagonal_init(rows, cols, jitter=0.0, seed=0):
        """
        Generates a hexagonal lattice packing.
        """
        rng = np.random.RandomState(seed)
        # Approximate d
        d = 1.0 / cols * 0.99
        
        # Check if it fits vertically
        row_spacing = d * math.sqrt(3) / 2
        total_height = (rows - 1) * row_spacing + d
        if total_height > 1.0:
            d = 1.0 / ((rows - 1) * math.sqrt(3) / 2 + 1) * 0.99
            
        pts = []
        for row in range(rows):
            y = -0.5 + d/2 + row * (d * math.sqrt(3) / 2)
            if row % 2 == 0:
                x_start = -0.5 + d/2
                for col in range(cols):
                    x = x_start + col * d
                    if jitter > 0:
                        x += rng.uniform(-jitter*d, jitter*d)
                        y += rng.uniform(-jitter*d, jitter*d)
                    pts.append([x, y])
            else:
                x_start = -0.5 + d
                for col in range(cols - 1):
                    x = x_start + col * d
                    if jitter > 0:
                        x += rng.uniform(-jitter*d, jitter*d)
                        y += rng.uniform(-jitter*d, jitter*d)
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) >= N:
            return pts[:N]
        else:
            return None

    def objective(x_flat):
        """
        Objective function for Nelder-Mead.
        We want to maximize radius, so we minimize -radius.
        Nelder-Mead handles non-smooth functions well.
        """
        centers = x_flat.reshape(N, 2)
        r = compute_radius(centers)
        return -r

    # Main Logic
    best_centers = None
    best_radius = 0.0
    
    # Generate diverse initializations
    inits = []
    
    # 1. Hexagonal grids with different row counts and jitters
    for rows in range(12, 18):
        cols = math.ceil(N / rows)
        pts = hexagonal_init(rows, cols, jitter=0.0, seed=rows)
        if pts is not None:
            inits.append(pts)
        for jitter in [0.05, 0.1, 0.2]:
            pts = hexagonal_init(rows, cols, jitter=jitter, seed=rows*100 + int(jitter*100))
            if pts is not None:
                inits.append(pts)
                
    # 2. Square grids
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    inits.append(grid[:N])
    
    # 3. Random
    for seed in range(5):
        rng = np.random.RandomState(seed)
        pts = rng.uniform(-0.49, 0.49, (N, 2))
        inits.append(pts)

    # Evaluate all inits
    for i, init in enumerate(inits):
        if time.time() > deadline - 5:
            break
        r = compute_radius(init)
        if r > best_radius:
            best_radius = r
            best_centers = init.copy()
            
    # Sort inits by radius to prioritize better starts
    init_scores = [(compute_radius(init), i) for i, init in enumerate(inits)]
    init_scores.sort(reverse=True)

    # Run Nelder-Mead on the best few initializations
    tried_indices = set()
    # Nelder-Mead is slower per iteration than BFGS for high dimensions, 
    # but it doesn't need gradients. 
    # With N=200, dimension is 400. NM is very slow in high D.
    # However, the directive says to try NM. 
    # To make it feasible, we might need to limit the number of restarts or iterations.
    # Standard NM options: maxiter, xatol, fatol.
    
    for score, idx in init_scores:
        if len(tried_indices) >= 5:
            break
        if idx in tried_indices:
            continue
        tried_indices.add(idx)
        
        if time.time() > deadline - 30:
            break
            
        x0 = inits[idx].flatten()
        
        try:
            # Nelder-Mead can be unstable. 
            # maxiter: 2000 might be too many for 400D in 120s. 
            # Let's try 500 iterations first.
            res = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'disp': False}
            )
            
            optimized_centers = res.x.reshape(N, 2)
            
            # Ensure validity
            r_opt = compute_radius(optimized_centers)
            
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = optimized_centers.copy()
                
        except Exception:
            pass

    # Final polish: Run Nelder-Mead again on the best result found, perhaps with tighter tolerance
    if best_centers is not None and time.time() < deadline - 10:
        try:
            x0 = best_centers.flatten()
            res = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False}
            )
            final_centers = res.x.reshape(N, 2)
            r_final = compute_radius(final_centers)
            if r_final > best_radius:
                best_radius = r_final
                best_centers = final_centers.copy()
        except Exception:
            pass

    if best_centers is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers

# EVOLVE-BLOCK-END
