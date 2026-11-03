"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    n = N  # 50

    def compute_radius(centers):
        """Compute the max common radius for given centers (vectorized)."""
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        # Distance to boundary
        dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
        min_boundary_dist = np.min(dist_to_boundary)
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        return r

    def objective_flat(x, beta=50.0):
        """Smooth objective: minimize -smooth_min_radius.
        Uses log-sum-exp to approximate the minimum of all constraints.
        """
        centers = x.reshape(n, 2)
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        # Get upper triangle
        iu = np.triu_indices(n, k=1)
        pair_dists = dists[iu]
        
        # Boundary distances
        boundary_dists = np.min(0.5 - np.abs(centers), axis=1)
        
        # Combine: radius is min of (pair_dists/2) and boundary_dists
        # Use log-sum-exp for smooth min
        all_constraints = np.concatenate([pair_dists / 2.0, boundary_dists])
        
        # Smooth min via log-sum-exp: -logsumexp(-beta * x) approximates min(x)
        # For numerical stability
        max_val = np.max(all_constraints)
        shifted = beta * (all_constraints - max_val)
        # Clip for numerical stability
        shifted = np.clip(shifted, -500, 500)
        logsumexp = max_val * beta + np.log(np.sum(np.exp(shifted)))
        smooth_min = -(logsumexp - max_val * beta) / beta  # This is wrong, let me redo
        
        # Correct smooth min: min(x) ≈ -1/beta * log(sum(exp(-beta*x)))
        neg_beta_x = -beta * all_constraints
        neg_beta_x = np.clip(neg_beta_x, -500, 500)
        max_nb = np.max(neg_beta_x)
        logsumexp_val = max_nb + np.log(np.sum(np.exp(neg_beta_x - max_nb)))
        smooth_min_val = -logsumexp_val / beta
        
        return -smooth_min_val  # We minimize, so negate

    def objective_grad(x, beta=50.0):
        """Gradient of the smooth objective."""
        centers = x.reshape(n, 2)
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        iu = np.triu_indices(n, k=1)
        pair_dists = dists[iu]
        
        boundary_dists = np.min(0.5 - np.abs(centers), axis=1)
        all_constraints = np.concatenate([pair_dists / 2.0, boundary_dists])
        
        # Softmax weights for the min
        neg_beta_x = -beta * all_constraints
        neg_beta_x = np.clip(neg_beta_x, -500, 500)
        max_nb = np.max(neg_beta_x)
        exp_vals = np.exp(neg_beta_x - max_nb)
        weights = exp_vals / np.sum(exp_vals)
        
        # Gradient: d(-smooth_min)/dx = sum(w_i * d(constraint_i)/dx)
        # For pair distances: d(pair_dist_ij/2)/dx_i = (x_i - x_j) / (2 * dist_ij)
        # For boundary: depends on which coordinate is closest
        
        grad = np.zeros_like(x)
        
        # Pairwise contributions
        n_pairs = len(iu)
        for idx in range(n_pairs):
            i, j = iu[0][idx], iu[1][idx]
            d = centers[i] - centers[j]
            dist = pair_dists[idx]
            if dist > 1e-12:
                w = weights[idx] * beta
                # d(-smooth_min)/dx_i includes w * d(pair_dist/2)/dx_i
                # d(pair_dist/2)/dx_i = d/(2*dist) where d = x_i - x_j
                grad[i*2:i*2+2] += w * d / (2.0 * dist)
                grad[j*2:j*2+2] += w * (-d) / (2.0 * dist)
        
        # Boundary contributions
        for idx in range(n, 2*n):
            i = idx - n
            w = weights[idx] * beta
            # boundary_dist_i = min(0.5 - |x_i|, 0.5 - |y_i|)
            # d(0.5 - |x|)/dx = -sign(x), d(0.5 - |y|)/dy = -sign(y)
            # But we need the min, so only the active constraint contributes
            bx = 0.5 - abs(centers[i, 0])
            by = 0.5 - abs(centers[i, 1])
            if bx <= by:
                # x is the binding constraint
                if centers[i, 0] > 0:
                    grad[i*2] += w * (-1.0)  # d(0.5-x)/dx = -1
                else:
                    grad[i*2] += w * (1.0)   # d(0.5+x)/dx = 1
            else:
                if centers[i, 1] > 0:
                    grad[i*2+1] += w * (-1.0)
                else:
                    grad[i*2+1] += w * (1.0)
        
        return grad

    def hex_init():
        """Hexagonal lattice initialization for 50 circles."""
        # Try to fit a hexagonal pattern
        # 7 columns x 7 rows = 49, plus 1 extra
        # Or use a pattern that fits better
        pts = []
        # Use a hexagonal arrangement
        # For 50 circles, try various row/column counts
        best_pts = None
        best_r = 0
        
        for rows in range(5, 11):
            for cols in range(5, 11):
                if rows * cols < 50:
                    continue
                if rows * cols > 55:  # don't go too far
                    continue
                # Hexagonal offset
                spacing = 1.0 / (cols + 0.5 * (rows - 1) / cols * 0.866)
                # Actually, let's just use a simple grid with hex offset
                # Row spacing: s, column spacing: s, offset: s/2 for odd rows
                # Total width: (cols-1)*s + s/2 = (cols - 0.5)*s
                # Total height: (rows-1)*s*sqrt(3)/2
                # We need both <= 1
                # s <= 1/(cols-0.5) and s*sqrt(3)/2 <= 1/(rows-1)
                s_x = 1.0 / (cols - 0.5)
                s_y = 2.0 / ((rows - 1) * np.sqrt(3)) if rows > 1 else 1.0
                s = min(s_x, s_y)
                
                pts = []
                for r in range(rows):
                    for c in range(cols):
                        x = (c + (0.5 if r % 2 == 1 else 0)) * s - (cols - 0.5) * s / 2
                        y = r * s * np.sqrt(3) / 2 - (rows - 1) * s * np.sqrt(3) / 4
                        pts.append([x, y])
                
                if len(pts) >= 50:
                    pts = pts[:50]
                    pts_arr = np.array(pts)
                    # Check bounds
                    if np.all(np.abs(pts_arr) <= 0.5):
                        r_test = compute_radius(pts_arr)
                        if r_test > best_r:
                            best_r = r_test
                            best_pts = pts_arr
        
        if best_pts is not None:
            return best_pts
        
        # Fallback: 7x7 grid + center
        ticks7 = (np.arange(7) + 0.5) / 7 - 0.5
        grid = np.array([(x, y) for y in ticks7 for x in ticks7])
        grid = np.vstack([grid, np.array([[0.0, 0.0]])])
        return grid[:50]

    def grid_init(rows, cols):
        """Simple grid initialization."""
        total = rows * cols
        if total < 50:
            return None
        xs = (np.arange(cols) + 0.5) / cols - 0.5
        ys = (np.arange(rows) + 0.5) / rows - 0.5
        grid = np.array([(x, y) for y in ys for x in xs])
        if len(grid) >= 50:
            grid = grid[:50]
        return grid

    # Generate multiple initializations
    inits = []
    
    # Hexagonal
    hex_pts = hex_init()
    if hex_pts is not None:
        inits.append(hex_pts)
    
    # Various grids
    for rows, cols in [(7, 7), (5, 10), (10, 5), (6, 9), (9, 6), (5, 9), (9, 5)]:
        g = grid_init(rows, cols)
        if g is not None:
            inits.append(g)
    
    # Add jittered versions
    np.random.seed(42)
    for base in inits[:3]:
        jittered = base + 0.01 * np.random.randn(50, 2)
        jittered = np.clip(jittered, -0.49, 0.49)
        inits.append(jittered)
    
    # Find best initial radius
    best_init = None
    best_init_r = 0
    for init in inits:
        r = compute_radius(init)
        if r > best_init_r:
            best_init_r = r
            best_init = init
    
    if best_init is None:
        # Absolute fallback
        ticks7 = (np.arange(7) + 0.5) / 7 - 0.5
        best_init = np.array([(x, y) for y in ticks7 for x in ticks7])
        best_init = np.vstack([best_init, np.array([[0.0, 0.0]])])[:50]
    
    best_centers = best_init.copy()
    best_r = best_init_r
    
    # L-BFGS-B optimization from multiple starts
    bounds = [(-0.5, 0.5)] * (2 * n)
    
    for init_idx, init in enumerate(inits[:5]):  # Try top 5 initializations
        if time.time() > deadline - 20:
            break
        
        x0 = init.flatten()
        
        # Use L-BFGS-B with gradient
        try:
            result = minimize(
                objective_flat,
                x0,
                jac=objective_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if result.success or result.fun < objective_flat(x0):
                centers = result.x.reshape(n, 2)
                r = compute_radius(centers)
                if r > best_r:
                    best_r = r
                    best_centers = centers.copy()
        except Exception:
            pass
    
    # Refine the best solution with higher beta (sharper min)
    if time.time() < deadline - 10:
        # Use the best centers as starting point with higher beta
        x0 = best_centers.flatten()
        
        def objective_sharp(x, beta=100.0):
            centers = x.reshape(n, 2)
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            iu = np.triu_indices(n, k=1)
            pair_dists = dists[iu]
            boundary_dists = np.min(0.5 - np.abs(centers), axis=1)
            all_constraints = np.concatenate([pair_dists / 2.0, boundary_dists])
            neg_beta_x = -beta * all_constraints
            neg_beta_x = np.clip(neg_beta_x, -500, 500)
            max_nb = np.max(neg_beta_x)
            logsumexp_val = max_nb + np.log(np.sum(np.exp(neg_beta_x - max_nb)))
            smooth_min_val = -logsumexp_val / beta
            return -smooth_min_val
        
        try:
            result = minimize(
                objective_sharp,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
            )
            centers = result.x.reshape(n, 2)
            r = compute_radius(centers)
            if r > best_r:
                best_r = r
                best_centers = centers.copy()
        except Exception:
            pass
    
    # Final safety check
    best_centers = np.clip(best_centers, -0.499999, 0.499999)
    best_centers = np.nan_to_num(best_centers, nan=0.0, posinf=0.499999, neginf=-0.499999)
    
    return best_centers
# EVOLVE-BLOCK-END
