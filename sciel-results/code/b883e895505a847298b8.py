"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Generate hexagonal lattice points inside a disk
    def hex_lattice(radius_limit, spacing):
        points = []
        sqrt3 = np.sqrt(3.0)
        # Determine range
        k = int(1.0 / spacing) + 3
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * sqrt3 / 2.0
                if np.hypot(x, y) <= radius_limit:
                    points.append((x, y))
        return np.array(points) if points else np.empty((0, 2))

    # Compute the packing radius for a set of points
    def compute_packing_radius(points):
        if len(points) == 0:
            return 0.0
        # Boundary constraint
        boundary_r = 1.0 - np.max(np.linalg.norm(points, axis=1))
        if boundary_r <= 0:
            return 0.0
        # Pairwise distances - use cdist for efficiency
        from scipy.spatial.distance import cdist
        dists = cdist(points, points)
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        return min(boundary_r, min_pair_dist / 2.0)

    # Force-based relaxation
    def relax(points, max_iters=5000, time_limit=None):
        n = len(points)
        if n == 0:
            return points
        best_points = points.copy()
        best_radius = compute_packing_radius(best_points)
        
        # Adaptive step size
        step_size = 0.01
        patience = 0
        max_patience = 200
        
        for iteration in range(max_iters):
            if time_limit is not None and time.time() > time_limit:
                break
            
            # Compute forces
            forces = np.zeros_like(points)
            current_radius = compute_packing_radius(points)
            
            # Pairwise repulsion for close pairs
            from scipy.spatial.distance import cdist
            dists = cdist(points, points)
            np.fill_diagonal(dists, np.inf)
            
            # Only consider pairs closer than 2 * best_radius + small margin
            threshold = 2.0 * max(best_radius, 0.01) + 0.05
            close_mask = dists < threshold
            
            # For each close pair, compute repulsive force
            for i in range(n):
                for j in range(i + 1, n):
                    if close_mask[i, j]:
                        d = dists[i, j]
                        if d < 1e-10:
                            # Random small perturbation
                            forces[i] += np.random.randn(2) * 0.01
                            forces[j] -= np.random.randn(2) * 0.01
                        else:
                            # Repulsion proportional to overlap
                            overlap = threshold - d
                            if overlap > 0:
                                direction = (points[i] - points[j]) / d
                                force_mag = overlap * 0.5
                                forces[i] += direction * force_mag
                                forces[j] -= direction * force_mag
            
            # Boundary constraint: push inward if too close to boundary
            norms = np.linalg.norm(points, axis=1)
            boundary_margin = best_radius + 0.01
            for i in range(n):
                if norms[i] > 1.0 - boundary_margin:
                    if norms[i] > 1e-10:
                        direction = -points[i] / norms[i]
                        excess = norms[i] - (1.0 - boundary_margin)
                        forces[i] += direction * excess * 0.5
            
            # Apply forces with adaptive step
            points_new = points + step_size * forces
            
            # Keep points inside the unit disk
            norms_new = np.linalg.norm(points_new, axis=1)
            max_allowed = 1.0 - 1e-10
            for i in range(n):
                if norms_new[i] > max_allowed:
                    points_new[i] = points_new[i] / norms_new[i] * max_allowed
            
            new_radius = compute_packing_radius(points_new)
            
            if new_radius > best_radius:
                best_radius = new_radius
                best_points = points_new.copy()
                points = points_new
                patience = 0
                step_size = min(step_size * 1.05, 0.05)
            else:
                patience += 1
                step_size *= 0.95
                if patience > max_patience:
                    break
        
        return best_points

    # Initial hexagonal lattice
    # Estimate spacing: area per circle ~ pi/1.5 * (1/sqrt(N))^2 roughly
    # For N circles in unit disk, average area per circle = pi/N
    # Hexagonal packing: area per circle = (sqrt(3)/2) * s^2 where s is spacing
    # So s^2 = 2*pi/(sqrt(3)*N)
    s = np.sqrt(2.0 * np.pi / (np.sqrt(3.0) * N))
    
    # Try a few different spacings
    best_points = None
    best_radius = 0.0
    
    for spacing_factor in [1.0, 0.95, 1.05, 0.9, 1.1]:
        spacing = s * spacing_factor
        # Use a slightly larger radius for lattice generation, then filter
        lattice_points = hex_lattice(1.0, spacing)
        
        if len(lattice_points) < N:
            continue
        
        # Select N points closest to center
        norms = np.linalg.norm(lattice_points, axis=1)
        idx = np.argsort(norms)[:N]
        points = lattice_points[idx].copy()
        
        r = compute_packing_radius(points)
        if r > best_radius:
            best_radius = r
            best_points = points.copy()
        
        # Relax
        relaxed = relax(points, max_iters=3000, time_limit=deadline - 5.0)
        r_relaxed = compute_packing_radius(relaxed)
        if r_relaxed > best_radius:
            best_radius = r_relaxed
            best_points = relaxed.copy()
        
        if time.time() > deadline - 10.0:
            break
    
    # Final relaxation if time permits
    if time.time() < deadline - 5.0 and best_points is not None:
        final_relaxed = relax(best_points, max_iters=5000, time_limit=deadline - 2.0)
        r_final = compute_packing_radius(final_relaxed)
        if r_final > best_radius:
            best_points = final_relaxed
    
    if best_points is None or len(best_points) != N:
        # Fallback: simple hexagonal lattice
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98
    
    return best_points
# EVOLVE-BLOCK-END
