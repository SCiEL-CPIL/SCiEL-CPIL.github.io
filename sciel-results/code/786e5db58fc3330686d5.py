"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # 1. Initialization: Hexagonal lattice
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

    def compute_radius(pts):
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

    # Find optimal hex lattice radius via binary search
    r_low = 0.01
    r_high = 0.08
    
    while True:
        pts = generate_hex_lattice(r_low)
        if len(pts) >= N:
            break
        r_low *= 0.8
        
    for _ in range(20):
        r_mid = (r_low + r_high) / 2.0
        pts = generate_hex_lattice(r_mid)
        if len(pts) >= N:
            r_low = r_mid
        else:
            r_high = r_mid
            
    points = generate_hex_lattice(r_low)
    
    if len(points) < N:
        angles = np.random.rand(N) * 2 * np.pi
        radii = np.sqrt(np.random.rand(N)) * 0.5
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    if len(points) > N:
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists, kind="stable")[:N]
        points = points[indices]
        
    # 2. Optimization using force-based relaxation with adaptive step size
    best_points = points.copy()
    best_r = compute_radius(best_points)
    
    # Use a spatial hash for neighbor search to speed up force computation
    def build_spatial_hash(pts, cell_size):
        """Build a spatial hash for fast neighbor queries."""
        h = {}
        for i, p in enumerate(pts):
            cx = int(np.floor(p[0] / cell_size))
            cy = int(np.floor(p[1] / cell_size))
            key = (cx, cy)
            if key not in h:
                h[key] = []
            h[key].append(i)
        return h
    
    def get_neighbors(pts, i, cell_size, spatial_hash):
        """Get neighbors of point i within cell_size distance."""
        p = pts[i]
        cx = int(np.floor(p[0] / cell_size))
        cy = int(np.floor(p[1] / cell_size))
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (cx + dx, cy + dy)
                if key in spatial_hash:
                    for j in spatial_hash[key]:
                        if j != i:
                            neighbors.append(j)
        return neighbors
    
    # Main optimization loop
    step_size = 0.005
    no_improve_count = 0
    max_no_improve = 500
    
    for it in range(20000):
        if time.time() > deadline:
            break
        
        # Compute distances and current radius
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        radii = np.hypot(points[:, 0], points[:, 1])
        bound_dists = 1.0 - radii
        
        current_r = min(np.min(dists) / 2.0, np.min(bound_dists))
        
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # If stuck, add random perturbation
        if no_improve_count > max_no_improve:
            # Perturb a few random points
            n_perturb = max(5, N // 50)
            idx = np.random.choice(N, size=n_perturb, replace=False)
            points[idx] += np.random.randn(n_perturb, 2) * 0.01
            no_improve_count = 0
            continue
        
        # Calculate forces
        # Use a target radius slightly above current to encourage expansion
        r_target = current_r * 1.002 + 0.0001
        
        forces = np.zeros_like(points)
        
        # Build spatial hash for efficient neighbor search
        cell_size = max(2.0 * r_target * 1.5, 0.05)
        spatial_hash = build_spatial_hash(points, cell_size)
        
        # Compute pairwise forces using spatial hash
        for i in range(N):
            # Boundary force: push points towards center if too close to boundary
            if radii[i] > 1.0 - r_target:
                violation = radii[i] - (1.0 - r_target)
                if violation > 0:
                    # Force proportional to violation, directed towards center
                    forces[i] -= (points[i] / radii[i]) * violation * 50.0
            
            # Pairwise repulsion forces
            neighbors = get_neighbors(points, i, cell_size, spatial_hash)
            for j in neighbors:
                if j <= i:  # Only compute each pair once
                    continue
                d = dists[i, j]
                if d < 2.0 * r_target * 1.5:
                    # Repulsive force: stronger when closer
                    diff_ij = points[i] - points[j]
                    if d > 1e-8:
                        # Force magnitude decays with distance
                        force_mag = 1.0 / (d * d)
                        # Scale to keep forces reasonable
                        force_vec = diff_ij * force_mag * 0.005
                        forces[i] += force_vec
                        forces[j] -= force_vec
        
        # Update positions with adaptive step size
        points += forces * step_size
        
        # Project points back inside unit circle (hard constraint)
        new_radii = np.hypot(points[:, 0], points[:, 1])
        outside = new_radii > 1.0
        if np.any(outside):
            points[outside] = points[outside] / new_radii[outside, np.newaxis]
        
        # Adaptive step size: decrease if radius decreased, increase if increased
        if current_r < best_r:
            step_size = max(step_size * 0.95, 0.0001)
        else:
            step_size = min(step_size * 1.02, 0.02)
    
    # Final check
    final_r = compute_radius(best_points)
    if final_r > best_r:
        best_r = final_r
        best_points = best_points.copy()

    return best_points
# EVOLVE-BLOCK-END
