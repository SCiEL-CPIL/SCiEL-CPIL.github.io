"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def remaining_time():
        return deadline - time.time()

    def compute_true_radius(pts):
        """Compute the actual radius: min of all pair distances / 2 and wall distances."""
        # Check if all points are inside the square
        if np.any(pts < -0.5) or np.any(pts > 0.5):
            return 0.0
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists) / 2.0
        
        wall_dists = 0.5 - np.abs(pts)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist, min_wall_dist)

    def initial_hex_grid():
        """Generate a hexagonal grid of points."""
        # For N=200, a hexagonal packing with ~15 rows
        # Let's try to fit as many as possible
        # Hex packing: rows are offset by half a spacing
        # Let spacing = d. Row height = d * sqrt(3)/2
        # We want to maximize d such that we can fit 200 points.
        
        # Let's try a grid of roughly 15x14 or similar.
        # 15 rows. If we have 15 rows, average ~13-14 per row.
        
        # Let's start with a guess for d.
        # Area = 1. N * pi * r^2 <= 1. r <= 1/sqrt(200*pi) ~ 0.0399.
        # Best known r ~ 0.0366. d ~ 0.073.
        
        # Let's generate a hex grid with d = 0.075 (slightly larger to ensure fit, then optimize)
        d = 0.075
        rows = 15
        # y positions
        y = np.linspace(-0.5 + d/2, 0.5 - d/2, rows)
        
        pts = []
        for i, y_i in enumerate(y):
            # x positions
            # Offset every other row
            offset = 0 if i % 2 == 0 else d/2
            # How many points fit in x?
            # x ranges from -0.5 + d/2 to 0.5 - d/2
            # Width available = 1 - d
            # Number of points = floor((1 - d) / d) + 1
            n_x = int((1.0 - d) / d) + 1
            # Center them
            x_start = -0.5 + d/2 - offset
            x = x_start + np.arange(n_x) * d
            # Clip to bounds
            x = x[(x >= -0.5 + d/2 - 1e-9) & (x <= 0.5 - d/2 + 1e-9)]
            
            for x_i in x:
                pts.append([x_i, y_i])
        
        pts = np.array(pts)
        
        # If we have too many, sample 200. If too few, add random points?
        # With d=0.075, n_x = int(0.925/0.075)+1 = 12+1=13.
        # 15 rows * 13 = 195 points. Close to 200.
        # Let's try d=0.072.
        # n_x = int(0.928/0.072)+1 = 12+1=13. Still 13.
        # 15*13 = 195.
        # Try 16 rows?
        # Height needed = 15 * d * sqrt(3)/2.
        # 15 * 0.075 * 0.866 = 0.975. Fits in 1.
        # 16 rows: 15 * 0.075 * 0.866 = 0.975. Wait, 16 rows means 15 intervals.
        # So 16 rows fit.
        # Let's try 16 rows.
        
        d = 0.072
        rows = 16
        y = np.linspace(-0.5 + d/2, 0.5 - d/2, rows)
        pts = []
        for i, y_i in enumerate(y):
            offset = 0 if i % 2 == 0 else d/2
            n_x = int((1.0 - d) / d) + 1
            x_start = -0.5 + d/2 - offset
            x = x_start + np.arange(n_x) * d
            x = x[(x >= -0.5 + d/2 - 1e-9) & (x <= 0.5 - d/2 + 1e-9)]
            for x_i in x:
                pts.append([x_i, y_i])
        
        pts = np.array(pts)
        
        if len(pts) < N:
            # Add random points inside
            while len(pts) < N:
                p = np.random.uniform(-0.5 + d, 0.5 - d, 2)
                # Check if close to existing? Just add for now, optimizer will fix.
                pts = np.vstack([pts, [p]])
        
        if len(pts) > N:
            # Keep the first N? Or random?
            # Random might break structure. Let's keep first N (top rows).
            pts = pts[:N]
            
        return pts

    def optimize_repulsion(pts, time_limit):
        """
        Optimize points using repulsive forces.
        Force on i: sum_{j != i} (c_i - c_j) / |c_i - c_j|^2 * strength
        Plus wall repulsion.
        """
        current_time = time.time()
        
        # Parameters
        lr = 0.0001
        repulsion_strength = 0.001
        wall_strength = 0.01
        damping = 0.9
        
        v = np.zeros_like(pts)
        best_pts = pts.copy()
        best_radius = compute_true_radius(pts)
        
        # Precompute pair indices? N=200, 20k pairs.
        # We can just use cdist or loops.
        
        steps = 0
        while time.time() - current_time < time_limit:
            # Compute forces
            forces = np.zeros_like(pts)
            
            # Pairwise repulsion
            # Use vectorized cdist for speed
            dists = cdist(pts, pts)
            np.fill_diagonal(dists, np.inf)
            
            # F_ij = (c_i - c_j) / d_ij^2
            # We want to push them apart.
            # If d_ij is small, force is large.
            # To avoid singularity, we can cap the force or use a softening parameter.
            # Let's use F = (c_i - c_j) / (d_ij^2 + eps)
            eps = 1e-6
            inv_d2 = 1.0 / (dists**2 + eps)
            np.fill_diagonal(inv_d2, 0)
            
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            forces += repulsion_strength * np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
            
            # Wall repulsion
            # Push away from walls if close.
            # Wall distance: 0.5 - |x|, 0.5 - |y|
            # Force towards center.
            # F_x = -sign(x) * wall_strength * (1 / (0.5 - |x| + eps)) ?
            # Simpler: F = -pts * wall_strength / (0.5 - |pts| + eps)
            # Actually, just push towards center proportional to proximity.
            
            dist_to_wall_x = 0.5 - np.abs(pts[:, 0])
            dist_to_wall_y = 0.5 - np.abs(pts[:, 1])
            
            # If dist < threshold, push inwards.
            # Let's use a smooth potential.
            # V = 1 / (dist + eps)
            # F = -dV/dx = sign(x) / (dist + eps)^2
            
            forces[:, 0] += wall_strength * np.sign(pts[:, 0]) / (dist_to_wall_x + eps)**2
            forces[:, 1] += wall_strength * np.sign(pts[:, 1]) / (dist_to_wall_y + eps)**2
            
            # Update
            v = damping * v + (1 - damping) * forces
            pts += lr * v
            
            # Clip to square (hard constraint)
            pts[:, 0] = np.clip(pts[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
            pts[:, 1] = np.clip(pts[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
            
            # Check radius
            r = compute_true_radius(pts)
            if r > best_radius:
                best_radius = r
                best_pts = pts.copy()
            
            steps += 1
            # Adaptive learning rate?
            # If no improvement for a while, reduce lr?
            
        return best_pts

    def bottleneck_local_opt(pts, time_limit):
        """
        Focus on the bottleneck pair (minimum distance) and move them apart.
        Also handle wall bottlenecks.
        """
        current_time = time.time()
        best_pts = pts.copy()
        best_radius = compute_true_radius(pts)
        
        # Identify bottleneck
        def get_bottleneck(pts):
            dists = cdist(pts, pts)
            np.fill_diagonal(dists, np.inf)
            min_idx = np.unravel_index(np.argmin(dists), dists.shape)
            min_dist = dists[min_idx]
            
            wall_dists = 0.5 - np.abs(pts)
            min_wall_idx = np.argmin(np.min(wall_dists, axis=1))
            min_wall_dist = np.min(wall_dists[min_wall_idx])
            
            if min_dist / 2.0 < min_wall_dist:
                return ('pair', min_idx, min_dist / 2.0)
            else:
                return ('wall', min_wall_idx, min_wall_dist)
        
        steps = 0
        while time.time() - current_time < time_limit:
            b_type, idx, b_val = get_bottleneck(pts)
            
            # Try to improve the bottleneck
            # Simple Nelder-Mead on the involved points?
            # Or just gradient step.
            
            # Let's do a small random perturbation of the bottleneck points
            # and accept if it improves the global radius.
            
            new_pts = pts.copy()
            
            if b_type == 'pair':
                i, j = idx
                # Move i and j apart
                direction = new_pts[i] - new_pts[j]
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                else:
                    direction = np.random.randn(2)
                    direction = direction / np.linalg.norm(direction)
                
                step = 0.001
                new_pts[i] += step * direction
                new_pts[j] -= step * direction
                
            elif b_type == 'wall':
                i = idx
                # Move i towards center
                direction = -new_pts[i]
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                else:
                    direction = np.random.randn(2)
                    direction = direction / np.linalg.norm(direction)
                
                step = 0.001
                new_pts[i] += step * direction
            
            # Check if valid and improved
            new_pts[:, 0] = np.clip(new_pts[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
            new_pts[:, 1] = np.clip(new_pts[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
            
            r_new = compute_true_radius(new_pts)
            if r_new > best_radius:
                best_radius = r_new
                best_pts = new_pts.copy()
                pts = new_pts.copy()
            else:
                # Backtrack? Or just try another small step?
                # Let's try a smaller step or random direction
                pass
            
            steps += 1
            if steps % 100 == 0:
                # Occasionally do a global repulsion step to escape local minima
                pass
            
        return best_pts

    def polish(pts, time_limit):
        """
        High-precision polish using soft-min gradient descent.
        """
        current_time = time.time()
        best_pts = pts.copy()
        best_radius = compute_true_radius(pts)
        
        beta = 50.0
        lr = 0.001
        mu = 0.9
        v = np.zeros_like(pts)
        
        def soft_min_grad(pts):
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            pair_dists = dists / 2.0
            
            wall_dists = 0.5 - np.abs(pts)
            min_wall_dist_per_point = np.min(wall_dists, axis=1)
            
            all_dists = np.concatenate([pair_dists.ravel(), min_wall_dist_per_point])
            
            m = np.max(all_dists)
            exp_vals = np.exp(-beta * (all_dists - m))
            sum_exp = np.sum(exp_vals)
            soft_min_val = m - (1.0 / beta) * np.log(sum_exp)
            weights = exp_vals / sum_exp
            
            grad = np.zeros_like(pts)
            
            # Wall gradients
            wall_weights = weights[N*N:]
            abs_x = np.abs(pts[:, 0])
            abs_y = np.abs(pts[:, 1])
            dist_x = 0.5 - abs_x
            dist_y = 0.5 - abs_y
            
            mask_x = dist_x < dist_y
            mask_y = dist_y < dist_x
            mask_eq = (dist_x == dist_y)
            
            sign_x = np.sign(pts[:, 0])
            sign_y = np.sign(pts[:, 1])
            
            grad[:, 0] += np.where(mask_x, wall_weights, 0) * (-sign_x)
            grad[:, 1] += np.where(mask_y, wall_weights, 0) * (-sign_y)
            
            grad[:, 0] += np.where(mask_eq, wall_weights * 0.5, 0) * (-sign_x)
            grad[:, 1] += np.where(mask_eq, wall_weights * 0.5, 0) * (-sign_y)
            
            # Pair gradients
            pair_weights = weights[:N*N].reshape(N, N)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                coeff = -0.5 * pair_weights / dists
            coeff = np.where(np.eye(N, dtype=bool), 0, coeff)
            
            grad_pair = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
            grad += grad_pair
            
            return soft_min_val, grad
        
        steps = 0
        while time.time() - current_time < time_limit:
            val, grad = soft_min_grad(pts)
            
            v = mu * v + (1 - mu) * grad
            pts += lr * v
            
            pts[:, 0] = np.clip(pts[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
            pts[:, 1] = np.clip(pts[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
            
            r = compute_true_radius(pts)
            if r > best_radius:
                best_radius = r
                best_pts = pts.copy()
            
            steps += 1
            # Decay lr
            lr *= 0.999
            
        return best_pts

    # Main logic
    # 1. Initialize
    pts = initial_hex_grid()
    
    # 2. Global Repulsion Optimization
    t1 = remaining_time()
    if t1 > 10:
        pts = optimize_repulsion(pts, t1 - 5)
    
    # 3. Bottleneck Local Optimization
    t2 = remaining_time()
    if t2 > 5:
        pts = bottleneck_local_opt(pts, t2 - 2)
    
    # 4. Polish
    t3 = remaining_time()
    if t3 > 2:
        pts = polish(pts, t3 - 1)
    
    # Final check
    r_final = compute_true_radius(pts)
    
    # If invalid, return initial grid
    if r_final <= 0:
        return initial_hex_grid()
    
    return pts

# EVOLVE-BLOCK-END
