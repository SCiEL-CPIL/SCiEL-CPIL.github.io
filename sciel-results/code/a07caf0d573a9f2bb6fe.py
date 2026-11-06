"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    def elapsed():
        return time.time() - start_time
    
    def compute_radius(points):
        """Compute the valid packing radius for given points."""
        if len(points) != N:
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        # Check all points are strictly inside unit circle
        dists = np.hypot(points[:, 0], points[:, 1])
        if np.any(dists >= 1.0):
            return 0.0
        # Compute minimum pair distance
        # Use vectorized computation in chunks to save memory
        min_pair_dist = np.inf
        for i in range(N):
            # Distance from point i to all points j > i
            diffs = points[i+1:] - points[i]
            dists_i = np.hypot(diffs[:, 0], diffs[:, 1])
            if len(dists_i) > 0:
                min_pair_dist = min(min_pair_dist, np.min(dists_i))
        if min_pair_dist == np.inf:
            return 0.0
        r = min(min_pair_dist / 2.0, 1.0 - np.max(dists))
        return max(0.0, r)
    
    def hex_lattice_init(n_points, spacing=None):
        """Generate hexagonal lattice points inside unit disk."""
        if spacing is None:
            # Estimate spacing from density
            area_per_circle = np.pi / n_points
            spacing = np.sqrt(area_per_circle / (np.sqrt(3)/2))
        
        points = []
        # Generate hex lattice points
        for j in range(-int(1.0/spacing) - 2, int(1.0/spacing) + 3):
            for i in range(-int(1.0/spacing) - 2, int(1.0/spacing) + 3):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y < 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) == 0:
            return np.zeros((n_points, 2))
        
        # If we have more points than needed, select the ones closest to center
        if len(points) > n_points:
            dists = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(dists, kind='stable')[:n_points]
            points = points[idx]
        elif len(points) < n_points:
            # Pad with random points inside disk
            needed = n_points - len(points)
            random_pts = np.random.uniform(0, 1, size=(needed, 2))
            angles = np.random.uniform(0, 2*np.pi, size=needed)
            radii = np.random.uniform(0, 0.9, size=needed)
            random_pts[:, 0] = radii * np.cos(angles)
            random_pts[:, 1] = radii * np.sin(angles)
            points = np.vstack([points, random_pts])
        
        return points
    
    def optimize(points, max_time=100.0):
        """Optimize circle packing using force-directed method."""
        points = points.copy()
        n = len(points)
        
        # Initial radius estimate
        r_current = compute_radius(points)
        if r_current <= 0:
            return points, r_current
        
        best_points = points.copy()
        best_radius = r_current
        
        # Adaptive parameters
        lr = 0.001  # learning rate
        lr_decay = 0.999
        min_lr = 1e-6
        
        # Force computation
        def compute_forces(pts):
            forces = np.zeros_like(pts)
            # Pairwise repulsion: push apart if too close
            # Target: maintain minimum distance
            target_dist = 2.0 * (1.0 - np.max(np.hypot(pts[:,0], pts[:,1]))) * 0.99
            # Actually, we want to maximize min pair dist / 2 and 1 - max dist
            # Use a simpler approach: repel all pairs, attract to center
            
            for i in range(n):
                for j in range(i+1, n):
                    diff = pts[i] - pts[j]
                    dist = np.hypot(diff[0], diff[1])
                    if dist < 1e-10:
                        # Random small perturbation
                        diff = np.random.randn(2) * 1e-6
                        dist = np.hypot(diff[0], diff[1])
                    
                    # Repulsive force: stronger when closer
                    # Use inverse square law with cutoff
                    if dist < 0.5:  # Only repel when reasonably close
                        force_mag = 1.0 / (dist * dist + 1e-8)
                        force = force_mag * diff / dist
                        forces[i] += force
                        forces[j] -= force
            
            # Attractive force to keep points inside and balanced
            # Pull toward origin with strength proportional to distance
            for i in range(n):
                dist = np.hypot(pts[i][0], pts[i][1])
                if dist > 0.5:
                    # Stronger pull when near boundary
                    pull_strength = (dist - 0.5) * 2.0
                    forces[i] -= pull_strength * pts[i] / (dist + 1e-8)
            
            return forces
        
        # Gradient descent with adaptive step size
        iteration = 0
        while elapsed() < start_time + max_time:
            forces = compute_forces(points)
            
            # Normalize forces to avoid huge steps
            force_norm = np.hypot(forces[:, 0], forces[:, 1])
            max_force = np.max(force_norm) if len(force_norm) > 0 else 1.0
            if max_force > 1e-10:
                forces = forces / max_force * 0.1
            
            # Update positions
            new_points = points + lr * forces
            
            # Enforce boundary: keep points inside unit disk
            dists = np.hypot(new_points[:, 0], new_points[:, 1])
            mask = dists > 0.999
            if np.any(mask):
                new_points[mask] = new_points[mask] * (0.999 / dists[mask])
            
            # Check if improvement
            new_radius = compute_radius(new_points)
            if new_radius > best_radius:
                best_radius = new_radius
                best_points = new_points.copy()
                points = new_points
            else:
                # Revert and reduce learning rate
                lr *= lr_decay
                if lr < min_lr:
                    break
            
            iteration += 1
        
        return best_points, best_radius
    
    # Generate initial hex lattice
    points = hex_lattice_init(N)
    
    # Optimize
    points, r = optimize(points, max_time=105.0)
    
    # Check deadline
    if elapsed() > deadline - 5.0:
        return points
    
    # Try a second optimization pass with different parameters if time allows
    if elapsed() < deadline - 30.0:
        # Slightly perturb the best solution and re-optimize
        perturbed = points + np.random.randn(N, 2) * 0.01
        # Keep inside disk
        dists = np.hypot(perturbed[:, 0], perturbed[:, 1])
        mask = dists > 0.99
        if np.any(mask):
            perturbed[mask] = perturbed[mask] * (0.99 / dists[mask])
        
        points2, r2 = optimize(perturbed, max_time=25.0)
        if r2 > r:
            points = points2
            r = r2
    
    return points
# EVOLVE-BLOCK-END
