import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin

    # 1. Initialization: Hexagonal Lattice
    def generate_hex_lattice(r):
        d = 2.0 * r
        if d <= 0:
            return np.array([])
        
        k = int(1.0 / d) + 2
        points = []
        sqrt3 = np.sqrt(3.0)
        
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * d
                y = j * d * sqrt3 / 2.0
                dist = np.hypot(x, y)
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

    # Find a good initial radius
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
    
    if len(points) < N:
        # Fallback random
        angles = np.random.rand(N) * 2 * np.pi
        radii = np.sqrt(np.random.rand(N))
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        points = points * 0.5 # Scale down

    if len(points) > N:
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists, kind="stable")[:N]
        points = points[indices]
        
    # Add small random jitter to break symmetry and avoid degenerate states
    points += np.random.normal(0, 0.001, (N, 2))
    
    best_points = points.copy()
    best_r = 0.0
    
    def compute_radius(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        
        # Efficient pairwise distance using chunked computation to save memory
        n = len(pts)
        min_pair_sq = np.inf
        chunk_size = 100
        for i in range(0, n, chunk_size):
            j = min(i + chunk_size, n)
            diff = pts[i:j, np.newaxis, :] - pts[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            # Zero out diagonal within chunk
            row_idx = np.arange(j - i)
            col_idx = np.arange(i, j)
            dists_sq[row_idx, col_idx] = np.inf
            chunk_min = np.min(dists_sq)
            if chunk_min < min_pair_sq:
                min_pair_sq = chunk_min
        
        min_pair = np.sqrt(min_pair_sq) / 2.0
        return min(min_pair, min_boundary)

    best_r = compute_radius(best_points)
    
    # Optimization Loop: Vectorized Gradient Descent on Potential Energy
    # Use smoother potential: E = sum_{i<j} 1/d_ij + lambda * sum_i 1/(1-|c_i|)
    # This avoids the extreme forces from 1/d^4
    
    def compute_forces(pts, lambda_bound):
        """Compute forces using smoother 1/d potential"""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, 1.0)
        
        # For potential 1/d, gradient is -1/d^2 * (p_i - p_j)
        # Force pushing apart: +1/d^2 * (p_i - p_j)
        dists = np.sqrt(dists_sq)
        inv_dists_sq = 1.0 / dists_sq
        mask = np.ones((N, N), dtype=bool)
        np.fill_diagonal(mask, False)
        inv_dists_sq[mask] = 0.0
        
        # Force magnitude: 1/d^2 for each pair
        factors = inv_dists_sq
        factors[mask] = 0.0
        
        # F_i = sum_j factors[i,j] * diff[i,j]
        pair_forces = np.einsum('ij,ijk->ik', factors, diff)
        
        # Boundary Force: gradient of 1/(1-|p|) is p/(|p|*(1-|p|)^2)
        radii = np.hypot(pts[:, 0], pts[:, 1])
        radii_safe = np.where(radii > 1e-6, radii, 1.0)
        radii_clipped = np.minimum(radii_safe, 1.0 - 1e-6)
        denom = radii_clipped * (1.0 - radii_clipped)**2
        denom = np.where(denom > 1e-9, denom, 1.0)
        
        bound_forces = -lambda_bound * pts / denom[:, np.newaxis]
        
        return pair_forces + bound_forces
    
    # Multi-stage optimization with different force potentials
    stages = [
        {'steps': 20000, 'step_size': 0.002, 'lambda': 3.0, 'eval_every': 20, 'max_no_improve': 80},
        {'steps': 15000, 'step_size': 0.001, 'lambda': 5.0, 'eval_every': 15, 'max_no_improve': 60},
        {'steps': 10000, 'step_size': 0.0005, 'lambda': 8.0, 'eval_every': 10, 'max_no_improve': 40},
    ]
    
    for stage_idx, stage in enumerate(stages):
        if time.time() > deadline - 10.0:
            break
            
        step_size = stage['step_size']
        lambda_bound = stage['lambda']
        eval_every = stage['eval_every']
        max_no_improve = stage['max_no_improve']
        no_improve_count = 0
        
        for it in range(stage['steps']):
            if time.time() > deadline - 5.0:
                break
                
            # Compute current radius periodically
            if it % eval_every == 0:
                current_r = compute_radius(points)
                if current_r > best_r:
                    best_r = current_r
                    best_points = points.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count > max_no_improve:
                        # Perturb to escape local minimum
                        perturb_scale = 0.01 * (1 + stage_idx * 0.5)
                        points += np.random.normal(0, perturb_scale, (N, 2))
                        # Ensure validity
                        radii = np.hypot(points[:, 0], points[:, 1])
                        outside = radii > 1.0
                        if np.any(outside):
                            points[outside] = points[outside] / radii[outside, np.newaxis]
                        
                        no_improve_count = 0
                        step_size = min(step_size * 1.3, 0.005)

            # Compute Forces
            total_forces = compute_forces(points, lambda_bound)
            
            # Adaptive step size based on gradient magnitude
            grad_mag = np.linalg.norm(total_forces, axis=1)
            grad_mag_safe = np.where(grad_mag > 1e-6, grad_mag, 1.0)
            adaptive_factor = 1.0 / (1.0 + 0.05 * grad_mag_safe)
            
            # Update with per-point adaptive step
            points += total_forces * (step_size * adaptive_factor)[:, np.newaxis]
            
            # Enforce hard constraint: inside unit circle
            new_radii = np.hypot(points[:, 0], points[:, 1])
            outside = new_radii > 1.0
            if np.any(outside):
                points[outside] = points[outside] / new_radii[outside, np.newaxis]
            
            # Decay step size
            step_size *= 0.99998
        
        # Check if we should stop early
        if best_r > 0.0534:
            break

    # Final high-precision polish pass
    if time.time() < deadline - 3.0:
        polish_steps = 3000
        polish_step = 0.0002
        lambda_bound = 10.0
        
        for it in range(polish_steps):
            if time.time() > deadline - 1.0:
                break
            
            total_forces = compute_forces(points, lambda_bound)
            points += total_forces * polish_step
            
            new_radii = np.hypot(points[:, 0], points[:, 1])
            outside = new_radii > 1.0
            if np.any(outside):
                points[outside] = points[outside] / new_radii[outside, np.newaxis]
        
        final_r = compute_radius(points)
        if final_r > best_r:
            best_r = final_r
            best_points = points.copy()

    return best_points
