"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 115s safety margin before 120s limit

    def compute_radius(pts):
        """
        Computes the maximum radius for the given points.
        r = min( min_{i<j} dist(i,j)/2, min_i (1 - dist(i, origin)) )
        """
        if len(pts) != N:
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        
        # Distance to boundary
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_pair, min_boundary)

    def generate_hex_lattice(r):
        """Generate points on a hexagonal lattice with spacing 2r that fit inside the unit circle."""
        d = 2.0 * r
        if d <= 0:
            return np.array([])
        
        # Range of indices to cover the unit circle
        k = int(1.0 / d) + 2
        
        points = []
        sqrt3 = np.sqrt(3.0)
        
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * d
                y = j * d * sqrt3 / 2.0
                dist = np.hypot(x, y)
                # Ensure the circle fits inside the unit circle
                if dist + r <= 1.0:
                    points.append([x, y])
        
        if len(points) == 0:
            return np.array([])
            
        points = np.array(points)
        
        if len(points) >= N:
            dists = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(dists, kind="stable")[:N]
            return points[indices]
        else:
            return points

    def soft_min_radius(pts, beta=50.0):
        """
        Soft-min surrogate using log-sum-exp for smooth gradients.
        Returns (soft_radius, gradient) where gradient is d(soft_radius)/d(pts).
        """
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        np.fill_diagonal(dists, np.inf)
        
        # Only consider upper triangle to avoid double counting
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]  # shape (M,)
        
        # Boundary distances
        radii = np.hypot(pts[:, 0], pts[:, 1])
        boundary_dists = 1.0 - radii  # shape (N,)
        
        # Soft min of pair_dists/2 and boundary_dists
        # soft_min(x) = -logsumexp(-x*beta)/beta
        all_vals = np.concatenate([pair_dists / 2.0, boundary_dists])
        
        # Stable log-sum-exp
        max_val = np.max(all_vals)
        exp_vals = np.exp(-beta * (all_vals - max_val))
        log_sum_exp = np.log(np.sum(exp_vals)) + beta * max_val
        soft_r = -log_sum_exp / beta
        
        # Gradient: d(soft_min)/d(x_i) = -exp(-beta*x_i) / sum(exp(-beta*x_j)) * (-beta) / beta
        # = exp(-beta*(x_i - max_val)) / sum(exp(-beta*(x_j - max_val)))
        weights = exp_vals / np.sum(exp_vals)  # shape (M+N,)
        
        # Gradient w.r.t. pts
        grad = np.zeros((N, 2))
        
        # Pair contributions
        for idx in range(len(pair_dists)):
            i, j = iu[0][idx], iu[1][idx]
            w = weights[idx]
            # d(pair_dist/2)/d(pts[i]) = (pts[i] - pts[j]) / (2 * pair_dist)
            d = pts[i] - pts[j]
            dist = pair_dists[idx]
            if dist > 1e-10:
                grad[i] += w * d / (2.0 * dist)
                grad[j] += w * (-d) / (2.0 * dist)
        
        # Boundary contributions
        for i in range(N):
            w = weights[N + i]
            # d(boundary_dist)/d(pts[i]) = -pts[i] / |pts[i]|
            r_i = radii[i]
            if r_i > 1e-10:
                grad[i] += w * (-pts[i] / r_i)
        
        return soft_r, grad

    # 1. Initialization: Find a good hexagonal lattice starting point
    r_low = 0.01
    r_high = 0.08
    
    # Ensure r_low gives enough points
    while True:
        pts = generate_hex_lattice(r_low)
        if len(pts) >= N:
            break
        r_low *= 0.8
        
    # Binary search for max r in hex lattice
    for _ in range(20):
        r_mid = (r_low + r_high) / 2.0
        pts = generate_hex_lattice(r_mid)
        if len(pts) >= N:
            r_low = r_mid
        else:
            r_high = r_mid
            
    points = generate_hex_lattice(r_low)
    
    # Fallback if something went wrong
    if len(points) < N:
        angles = np.random.rand(N) * 2 * np.pi
        radii = np.sqrt(np.random.rand(N)) * 0.5
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        points = points[:N]

    # Ensure we have exactly N points
    if len(points) > N:
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists, kind="stable")[:N]
        points = points[indices]
        
    best_points = points.copy()
    best_r = compute_radius(best_points)
    
    # 2. Optimization using soft-min gradient ascent
    # Phase 1: Coarse optimization with soft-min
    # Phase 2: Fine polishing with exact min-based local moves
    
    # Phase 1: Soft-min gradient ascent
    beta = 30.0  # Start with moderate sharpness
    lr = 0.001
    no_improve_count = 0
    max_no_improve = 500
    
    for it in range(50000):
        if time.time() > deadline - 15.0:
            break
            
        # Compute soft radius and gradient
        soft_r, grad = soft_min_radius(points, beta)
        
        # Gradient ascent step
        points_new = points + lr * grad
        
        # Clip to stay reasonably inside (avoid numerical issues)
        # But don't hard-constrain; let the objective handle it
        if not np.all(np.isfinite(points_new)):
            break
            
        # Check if this improves the exact radius occasionally
        if it % 10 == 0:
            exact_r = compute_radius(points_new)
            if exact_r > best_r:
                best_r = exact_r
                best_points = points_new.copy()
                no_improve_count = 0
            else:
                no_improve_count += 10
        else:
            no_improve_count += 1
            
        # Adaptive learning rate
        if no_improve_count > max_no_improve:
            lr *= 0.5
            no_improve_count = 0
            if lr < 1e-7:
                break
        
        # Increase beta over time to sharpen the approximation
        if it % 500 == 0:
            beta = min(beta * 1.2, 100.0)
            
        points = points_new
        
    # Phase 2: Exact local polishing
    # Use the exact min to identify tight constraints and do targeted moves
    current_points = best_points.copy()
    current_r = best_r
    
    for polish_iter in range(10000):
        if time.time() > deadline - 3.0:
            break
            
        # Compute all distances
        diff = current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        radii = np.hypot(current_points[:, 0], current_points[:, 1])
        boundary_dists = 1.0 - radii
        
        min_pair = np.min(dists)
        min_boundary = np.min(boundary_dists)
        current_r = min(min_pair / 2.0, min_boundary)
        
        if current_r > best_r:
            best_r = current_r
            best_points = current_points.copy()
            
        # Identify active constraints (within small epsilon)
        eps = 1e-6
        active_pairs = np.argwhere(dists < min_pair + eps)
        active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]
        active_boundaries = np.where(boundary_dists < min_boundary + eps)[0]
        
        if len(active_pairs) == 0 and len(active_boundaries) == 0:
            break
        
        # For each active constraint, try small moves to relax it
        # Pick a random active point and nudge it
        candidates = set()
        for p in active_pairs:
            candidates.add(p[0])
            candidates.add(p[1])
        for b in active_boundaries:
            candidates.add(b)
            
        if not candidates:
            break
            
        idx = np.random.choice(list(candidates))
        
        # Compute force direction
        force = np.zeros(2)
        
        for p in active_pairs:
            if p[0] == idx:
                other = p[1]
                d = current_points[idx] - current_points[other]
                dist = np.linalg.norm(d)
                if dist > 1e-10:
                    force += d / dist
            elif p[1] == idx:
                other = p[0]
                d = current_points[idx] - current_points[other]
                dist = np.linalg.norm(d)
                if dist > 1e-10:
                    force += d / dist
                    
        if idx in active_boundaries:
            d = -current_points[idx]
            dist = np.linalg.norm(d)
            if dist > 1e-10:
                force += d / dist
                
        norm_force = np.linalg.norm(force)
        if norm_force > 1e-10:
            force = force / norm_force
        else:
            force = np.random.randn(2)
            force = force / np.linalg.norm(force)
            
        # Try multiple step sizes
        best_step_r = current_r
        best_step_points = current_points
        
        for step in [0.002, 0.001, 0.0005, 0.0002]:
            new_points = current_points.copy()
            new_points[idx] += force * step
            new_r = compute_radius(new_points)
            if new_r > best_step_r:
                best_step_r = new_r
                best_step_points = new_points
        
        if best_step_r > current_r:
            current_points = best_step_points
        else:
            # Try random perturbation
            for _ in range(3):
                new_points = current_points.copy()
                new_points[idx] += np.random.randn(2) * 0.001
                new_r = compute_radius(new_points)
                if new_r > current_r:
                    current_points = new_points
                    break
    
    return best_points
# EVOLVE-BLOCK-END
