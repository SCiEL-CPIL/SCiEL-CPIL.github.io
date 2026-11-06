"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    def generate_hex_lattice(n):
        """
        Generate n points in the unit disk using a hexagonal lattice.
        """
        # Estimate spacing
        # Area per point ~ sqrt(3)/2 * s^2
        # N * sqrt(3)/2 * s^2 ~ pi * (1/2)^2 ? No, disk radius 1.
        # Approx s = sqrt(2 * pi / (N * sqrt(3)))
        s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
        
        # Generate grid
        points = []
        # Range of indices
        # Max radius is 1.
        # Hex grid coordinates:
        # x = s * (i + 0.5 * j)
        # y = s * (sqrt(3)/2 * j)
        
        max_idx = int(np.ceil(1.0 / s)) + 2
        
        for j in range(-max_idx, max_idx + 1):
            for i in range(-max_idx, max_idx + 1):
                x = s * (i + 0.5 * j)
                y = s * (np.sqrt(3) / 2.0 * j)
                if np.hypot(x, y) <= 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        
        # If we have more than N, we need to select N.
        # If we have less, we need to add more (or adjust s).
        # For simplicity, if len > N, take the first N? Or the ones closest to center?
        # Taking first N from a loop is arbitrary.
        # Better: If len > N, randomly sample N? Or select N with largest distance to nearest neighbor?
        # Let's just take a random subset if we have too many, or generate with smaller s if too few.
        
        if len(points) > N:
            # Select N points. To ensure good distribution, we can just take a random subset.
            # Or, better, we can try to find a subset that maximizes min distance? Too expensive.
            # Let's just take the first N from a shuffled list?
            idx = np.random.choice(len(points), N, replace=False)
            points = points[idx]
        elif len(points) < N:
            # Generate with smaller spacing
            s *= 0.9
            points = []
            max_idx = int(np.ceil(1.0 / s)) + 2
            for j in range(-max_idx, max_idx + 1):
                for i in range(-max_idx, max_idx + 1):
                    x = s * (i + 0.5 * j)
                    y = s * (np.sqrt(3) / 2.0 * j)
                    if np.hypot(x, y) <= 1.0:
                        points.append([x, y])
            points = np.array(points)
            if len(points) > N:
                idx = np.random.choice(len(points), N, replace=False)
                points = points[idx]
            else:
                # If still too few, pad with random points
                while len(points) < N:
                    angle = np.random.rand() * 2 * np.pi
                    r = np.sqrt(np.random.rand()) * 0.9
                    points = np.vstack([points, [r * np.cos(angle), r * np.sin(angle)]])
        
        return points

    def compute_radius(pts):
        """
        Computes the maximum radius for the given points.
        """
        if len(pts) != N:
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return min(min_pair, min_boundary)

    # 1. Initialization: Hexagonal Lattice
    points = generate_hex_lattice(N)
    
    # Ensure exactly N points
    if len(points) != N:
        # Fallback
        angles = np.random.rand(N) * 2 * np.pi
        radii = np.sqrt(np.random.rand(N)) * 0.9
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
    # Project to unit disk
    radii = np.hypot(points[:, 0], points[:, 1])
    outside = radii > 1.0
    if np.any(outside):
        points[outside] = points[outside] / radii[outside, np.newaxis]

    best_points = points.copy()
    best_r = compute_radius(best_points)
    
    # 2. Optimization: Force-Based Relaxation
    beta = 200.0
    step_size = 0.01
    decay_rate = 0.9999
    
    # Pre-allocate arrays if needed? N=300 is small.
    
    for it in range(20000): # Increased iterations
        if time.time() > deadline:
            break
        
        # Compute distances once
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        radii = np.hypot(points[:, 0], points[:, 1])
        boundary_dists = 1.0 - radii
        
        # Check actual radius every 100 iterations to save time
        if it % 100 == 0:
            current_r = compute_radius(points)
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
        
        # Compute soft-min weights
        x_pair = dists / 2.0
        x_bound = boundary_dists
        
        min_x_pair = np.min(x_pair)
        min_x_bound = np.min(x_bound)
        min_x = min(min_x_pair, min_x_bound)
        
        w_pair = np.exp(-beta * (x_pair - min_x))
        np.fill_diagonal(w_pair, 0.0)
        
        w_bound = np.exp(-beta * (x_bound - min_x))
        
        Z = np.sum(w_pair) + np.sum(w_bound)
        
        if Z == 0:
            break
        
        w_pair /= Z
        w_bound /= Z
        
        # Forces
        safe_dists = np.where(dists > 1e-9, dists, 1.0)
        factor = w_pair / (2.0 * safe_dists)
        np.fill_diagonal(factor, 0.0)
        
        F_pair = np.einsum('ij,ijk->ik', factor, diff)
        
        safe_radii = np.where(radii > 1e-9, radii, 1.0)
        unit_vecs = points / safe_radii[:, np.newaxis]
        F_bound = -w_bound[:, np.newaxis] * unit_vecs
        
        forces = beta * (F_pair + F_bound)
        
        # Update
        points += forces * step_size
        
        # Project
        radii = np.hypot(points[:, 0], points[:, 1])
        outside = radii > 1.0
        if np.any(outside):
            points[outside] = points[outside] / radii[outside, np.newaxis]
            
        # Decay
        step_size *= decay_rate
        
        # Perturbation
        if it % 50 == 0 and it > 0:
            # Adaptive perturbation magnitude
            pert_mag = 0.005 * (1.0 + 0.1 * np.sin(it / 50.0))
            num_perturb = N // 10
            idx = np.random.choice(N, num_perturb, replace=False)
            perturbations = np.random.randn(num_perturb, 2) * pert_mag
            points[idx] += perturbations
            radii = np.hypot(points[:, 0], points[:, 1])
            outside = radii > 1.0
            if np.any(outside):
                points[outside] = points[outside] / radii[outside, np.newaxis]

    # Final check
    final_r = compute_radius(best_points)
    if final_r > best_r:
        best_r = final_r
        best_points = best_points.copy()

    return best_points
# EVOLVE-BLOCK-END
