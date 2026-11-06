"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

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

    def generate_hex_lattice(n_target):
        """
        Generate a hexagonal lattice of points that fits inside the unit circle.
        We binary search for the spacing d such that the number of points is close to n_target.
        """
        def count_points(d):
            if d <= 0:
                return 0
            k = int(1.0 / d) + 2
            sqrt3 = np.sqrt(3.0)
            count = 0
            for j in range(-k, k + 1):
                offset = 0.5 * d if (j % 2 != 0) else 0.0
                y = j * d * sqrt3 / 2.0
                if abs(y) > 1.0:
                    continue
                for i in range(-k, k + 1):
                    x = i * d + offset
                    if x*x + y*y <= 1.0:
                        count += 1
            return count

        # Binary search for d
        d_low = 0.01
        d_high = 0.5
        best_d = None
        best_diff = float('inf')
        
        # Coarse search
        for _ in range(60):
            d_mid = (d_low + d_high) / 2.0
            c = count_points(d_mid)
            if c > n_target:
                d_low = d_mid
            else:
                d_high = d_mid
            
            diff = abs(c - n_target)
            if diff < best_diff:
                best_diff = diff
                best_d = d_mid

        # Generate points with best_d
        d = best_d
        k = int(1.0 / d) + 2
        sqrt3 = np.sqrt(3.0)
        points = []
        for j in range(-k, k + 1):
            offset = 0.5 * d if (j % 2 != 0) else 0.0
            y = j * d * sqrt3 / 2.0
            if abs(y) > 1.0:
                continue
            for i in range(-k, k + 1):
                x = i * d + offset
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        
        # If we have more than N points, keep the N closest to the center
        if len(points) > N:
            dists_to_origin = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(dists_to_origin)
            points = points[indices[:N]]
        elif len(points) < N:
            needed = N - len(points)
            angles = np.random.rand(needed) * 2 * np.pi
            radii = np.sqrt(np.random.rand(needed)) * 0.9 + 0.1
            new_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            points = np.vstack([points, new_points])

        return points

    # 1. Initialization
    points = generate_hex_lattice(N)
    
    # Ensure validity
    radii = np.hypot(points[:, 0], points[:, 1])
    outside = radii > 1.0
    if np.any(outside):
        points[outside] = points[outside] / radii[outside, np.newaxis]

    best_points = points.copy()
    best_r = compute_radius(best_points)
    
    # 2. Optimization: Potential energy minimization with adaptive step size
    # Use a repulsive potential V_rep = 1/d^4 and confining potential V_conf = k*(r-1)^2 for r>0.9
    # We minimize the sum of these potentials, which encourages points to spread out and stay inside.
    
    # Parameters
    k_rep = 1.0      # Repulsion strength
    k_conf = 100.0   # Confinement strength
    lr = 0.001       # Initial learning rate
    lr_decay = 0.9999
    momentum = 0.9   # Momentum factor
    velocity = np.zeros_like(points)
    
    for it in range(50000):
        if time.time() > deadline - 5.0:
            break
        
        # Compute current radius
        current_r = compute_radius(points)
        
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
        
        # Compute forces
        # Pairwise repulsive forces
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Repulsive force: F_rep = -dV_rep/dd * (diff/dist) = k_rep * 4/d^5 * diff/dist = 4*k_rep*diff/d^6
        # To avoid singularity, use a softened version
        safe_dists = np.where(dists > 1e-6, dists, 1.0)
        # F_rep_ij = 4 * k_rep * diff_ij / dists_ij^6
        factor = 4.0 * k_rep / (safe_dists**6)
        np.fill_diagonal(factor, 0.0)
        F_rep = np.einsum('ij,ijk->ik', factor, diff)
        
        # Confinement force: push points toward center if they are near the boundary
        radii = np.hypot(points[:, 0], points[:, 1])
        # Force is -k_conf * (r - 1) * (points/r) for r > 0.8, else 0
        # Actually, we want to keep them inside, so force should be inward when r is large
        # Let's use a smooth confining potential: V_conf = k_conf * (r^2) for r > 0.8
        # Force = -dV/dr * unit_vec = -2*k_conf*r * unit_vec
        # But we only want this to be active near the boundary. Let's use a soft threshold.
        # Simpler: just use a harmonic confinement toward the origin with strength that increases near boundary
        # Or just use a constant inward force proportional to (r - r_min) where r_min is some threshold
        
        # Let's use: F_conf = -k_conf * (radii - 0.9) * (points / radii) for radii > 0.9, else 0
        # This pushes points back when they go beyond r=0.9
        mask = radii > 0.9
        F_conf = np.zeros_like(points)
        if np.any(mask):
            safe_radii = np.where(radii[mask] > 1e-6, radii[mask], 1.0)
            F_conf[mask] = -k_conf * (radii[mask] - 0.9) * (points[mask] / safe_radii[:, np.newaxis])
        
        forces = F_rep + F_conf
        
        # Update with momentum
        velocity = momentum * velocity + lr * forces
        points += velocity
        
        # Project to unit disk
        radii = np.hypot(points[:, 0], points[:, 1])
        outside = radii > 1.0
        if np.any(outside):
            points[outside] = points[outside] / radii[outside, np.newaxis]
        
        # Decay learning rate
        lr *= lr_decay
        
        # Occasional perturbation to escape local minima
        if it % 100 == 0 and it > 0:
            num_perturb = max(5, N // 20)
            idx = np.random.choice(N, num_perturb, replace=False)
            perturbations = np.random.randn(num_perturb, 2) * 0.01
            points[idx] += perturbations
            radii = np.hypot(points[:, 0], points[:, 1])
            outside = radii > 1.0
            if np.any(outside):
                points[outside] = points[outside] / radii[outside, np.newaxis]
            # Reset velocity for perturbed points
            velocity[idx] = 0.0

    # Final Polish: Use the soft-min surrogate approach for fine-tuning
    polish_points = best_points.copy()
    polish_step = 0.00005
    beta = 2000.0
    
    for it in range(5000):
        if time.time() > deadline - 1.0:
            break
        current_r = compute_radius(polish_points)
        if current_r > best_r:
            best_r = current_r
            best_points = polish_points.copy()
        
        # Compute distances
        diff = polish_points[:, np.newaxis, :] - polish_points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        radii = np.hypot(polish_points[:, 0], polish_points[:, 1])
        boundary_dists = 1.0 - radii
        
        x_pair = dists / 2.0
        x_bound = boundary_dists
        min_x = min(np.min(x_pair), np.min(x_bound))
        
        w_pair = np.exp(-beta * (x_pair - min_x))
        np.fill_diagonal(w_pair, 0.0)
        w_bound = np.exp(-beta * (x_bound - min_x))
        Z = np.sum(w_pair) + np.sum(w_bound)
        if Z == 0 or not np.isfinite(Z):
            break
        w_pair /= Z
        w_bound /= Z
        
        safe_dists = np.where(dists > 1e-9, dists, 1.0)
        factor = w_pair / (2.0 * safe_dists)
        np.fill_diagonal(factor, 0.0)
        F_pair = np.einsum('ij,ijk->ik', factor, diff)
        
        safe_radii = np.where(radii > 1e-9, radii, 1.0)
        unit_vecs = polish_points / safe_radii[:, np.newaxis]
        F_bound = -w_bound[:, np.newaxis] * unit_vecs
        
        forces = beta * (F_pair + F_bound)
        polish_points += forces * polish_step
        
        radii = np.hypot(polish_points[:, 0], polish_points[:, 1])
        outside = radii > 1.0
        if np.any(outside):
            polish_points[outside] = polish_points[outside] / radii[outside, np.newaxis]

    final_r = compute_radius(best_points)
    if final_r > best_r:
        best_r = final_r

    return best_points
# EVOLVE-BLOCK-END
