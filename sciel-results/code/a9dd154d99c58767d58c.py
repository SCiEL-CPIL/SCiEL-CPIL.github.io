"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    def compute_radius(centers):
        """Compute the maximum radius for given centers."""
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        n = len(centers)
        # Efficient pairwise distance calculation
        # Using broadcasting for memory efficiency with N=200
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def hexagonal_init():
        """Generate a good initial hexagonal packing."""
        # Approximate optimal layout for 200 circles
        # Try to fit ~14x14 or similar
        # We'll generate a hex grid and trim/pad
        pts = []
        spacing = 1.0 / 14.0
        h = spacing * math.sqrt(3) / 2
        
        row = 0
        y = -0.5 + spacing/2
        while y <= 0.5:
            x = -0.5 + spacing/2
            if row % 2 == 1:
                x += spacing / 2
            while x <= 0.5:
                pts.append([x, y])
            y += h
            row += 1
        
        pts = np.array(pts)
        if len(pts) > N:
            # Randomly select N points to preserve diversity, or just take first N
            # Taking first N is deterministic and fine
            pts = pts[:N]
        elif len(pts) < N:
            # Need more points, adjust spacing
            # This fallback is rare for N=200 in unit square
            # For now, just return what we have (should not happen)
            pass
            
        return pts

    def objective_and_grad(flat_centers):
        """
        Objective function based on Lennard-Jones potential + wall repulsion.
        We minimize this energy to maximize the minimum distance.
        """
        centers = flat_centers.reshape(-1, 2)
        
        # Wall repulsion: Soft penalty for approaching boundaries
        # We want centers to stay at least r away from walls.
        # Let's use a soft potential for walls: 1/d^12 where d is distance to wall.
        # But we don't know r yet. We can use a fixed scale or adaptive.
        # Better: Use a potential that encourages uniformity.
        
        # Let's use a simpler repulsive potential: sum of 1/d^2 for pairs, 
        # and 1/d_wall^2 for walls.
        # This drives points apart.
        
        # Pairwise repulsion
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, 1e10) # Avoid division by zero
        
        # 1/d^2 potential
        # Clip distances to avoid singularity
        dists_sq_clipped = np.maximum(dists_sq, 1e-6)
        pair_energy = np.sum(1.0 / dists_sq_clipped)
        
        # Wall repulsion
        # Distance to walls
        dist_walls = np.minimum(0.5 - np.abs(centers), 1e-6)
        wall_energy = np.sum(1.0 / (dist_walls**2))
        
        # Total energy
        energy = pair_energy + 10.0 * wall_energy # Weight walls more heavily
        
        # Gradient calculation
        # d/dx_i (1/d_ij^2) = -2 * (x_i - x_j) / d_ij^4
        # This is computationally expensive to do analytically for all pairs in a loop.
        # Let's use numerical gradient? No, L-BFGS-B is faster with analytical.
        # But for N=200, 400 variables. Analytical is better.
        
        grad = np.zeros_like(flat_centers)
        
        # Pairwise gradient
        # For each i, sum over j != i of -2 * (c_i - c_j) / d_ij^4
        # dists_sq is (N, N)
        # d_ij^4 = dists_sq^2
        inv_d4 = 1.0 / (dists_sq_clipped**2)
        np.fill_diagonal(inv_d4, 0.0)
        
        # grad_i = sum_j -2 * (c_i - c_j) * inv_d4[i, j]
        # c_i - c_j is the diff matrix
        # So grad = -2 * sum_j (diff[i, j] * inv_d4[i, j])
        grad = -2.0 * np.sum(diff * inv_d4[:, :, np.newaxis], axis=1)
        
        # Wall gradient
        # d/dx (1/(0.5-|x|)^2)
        # Let u = 0.5 - |x|. d = u^2. d/dx (1/d) = -1/d^2 * du/dx
        # du/dx = -sign(x)
        # So d/dx = -1/u^4 * (-sign(x)) = sign(x) / u^4
        # Similarly for y.
        
        # dist_walls is (N, 2)
        # u_x = 0.5 - |x|, u_y = 0.5 - |y|
        x = centers[:, 0]
        y = centers[:, 1]
        
        u_x = 0.5 - np.abs(x)
        u_y = 0.5 - np.abs(y)
        
        # Clip to avoid zero
        u_x = np.maximum(u_x, 1e-6)
        u_y = np.maximum(u_y, 1e-6)
        
        # Gradient x component from wall: sign(x) / u_x^4 * weight
        # Gradient y component from wall: sign(y) / u_y^4 * weight
        grad_x_wall = np.sign(x) / (u_x**4)
        grad_y_wall = np.sign(y) / (u_y**4)
        
        # Add to total gradient (scaled by wall weight 10.0)
        grad[:, 0] += 10.0 * grad_x_wall
        grad[:, 1] += 10.0 * grad_y_wall
        
        return energy, grad.flatten()

    # 1. Generate initial configuration
    centers = hexagonal_init()
    if len(centers) < N:
        # Fallback to square grid
        m = int(math.ceil(math.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        centers = grid[:N]
        
    flat_centers = centers.flatten()
    
    # 2. Optimize using L-BFGS-B
    # We minimize the repulsive energy. As energy decreases, min distance increases.
    # Bounds: centers must be in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        # Use L-BFGS-B
        res = minimize(
            objective_and_grad,
            flat_centers,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8, 'disp': False}
        )
        if res.success or res.fun < 1e10:
            optimized_centers = res.x.reshape(-1, 2)
            # Ensure bounds are strictly respected (clip)
            optimized_centers = np.clip(optimized_centers, -0.5, 0.5)
            centers = optimized_centers
    except Exception:
        pass
    
    # 3. Local refinement using exact radius calculation and simple hill climbing
    # The energy minimization gives a good approximate solution.
    # Now we refine by directly maximizing the minimum distance.
    
    current_r = compute_radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    # Simple local search: move points to increase min distance
    # We'll do a few iterations of moving the most constrained points.
    
    for iteration in range(50):
        if time.time() > deadline - 5:
            break
            
        # Compute current radius and bottleneck
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        min_wall_idx = np.argmin(wall_dists)
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        min_pair_idx = np.argmin(dists)
        i, j = np.unravel_index(min_pair_idx, (N, N))
        
        r = min(min_wall, min_pair / 2.0)
        
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
        
        # Try to move the bottleneck point(s)
        # If wall is bottleneck, move that point away from wall
        if min_wall < min_pair / 2.0:
            idx = min_wall_idx // 2
            pt = centers[idx]
            # Determine direction away from wall
            if pt[0] < -0.4: dir = [1, 0]
            elif pt[0] > 0.4: dir = [-1, 0]
            elif pt[1] < -0.4: dir = [0, 1]
            else: dir = [0, -1]
            
            # Try small steps
            for step in [0.001, 0.005, 0.01]:
                new_pt = pt + np.array(dir) * step
                if -0.5 <= new_pt[0] <= 0.5 and -0.5 <= new_pt[1] <= 0.5:
                    centers[idx] = new_pt
                    new_r = compute_radius(centers)
                    if new_r > r:
                        r = new_r
                        break
                    else:
                        centers[idx] = pt
        else:
            # Move i away from j
            pt_i = centers[i]
            pt_j = centers[j]
            dir = (pt_i - pt_j)
            norm = np.linalg.norm(dir)
            if norm > 1e-6:
                dir = dir / norm
            else:
                dir = [1, 0]
            
            for step in [0.001, 0.005, 0.01]:
                new_pt_i = pt_i + np.array(dir) * step
                if -0.5 <= new_pt_i[0] <= 0.5 and -0.5 <= new_pt_i[1] <= 0.5:
                    centers[i] = new_pt_i
                    new_r = compute_radius(centers)
                    if new_r > r:
                        r = new_r
                        break
                    else:
                        centers[i] = pt_i
                        
    # Final check
    final_r = compute_radius(best_centers)
    if final_r > best_r:
        best_r = final_r
    
    return best_centers
# EVOLVE-BLOCK-END
