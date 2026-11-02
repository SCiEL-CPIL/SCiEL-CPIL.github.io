import numpy as np
import time

N = 300

def construct():
        # Deadline guard
        start_time = time.time()
        deadline = 140.0  # Leave 10s margin
        
        # 1. Initialization: Hexagonal Lattice
        # Generate a dense hex lattice and pick the closest N points to origin
        def generate_hex_lattice(n_points):
            # Estimate spacing
            # Area per point ~ 1/sqrt(n)
            # Let's just generate a large grid and filter
            spacing = 0.1
            points = []
            # Loop over a range that covers unit circle
            # Radius 1, so x,y in [-1.1, 1.1]
            # Hex lattice:
            # x = i * spacing + (j % 2) * spacing / 2
            # y = j * spacing * sqrt(3) / 2
            
            # We need to find a spacing that yields at least n_points inside radius 1+spacing
            # Let's try a few spacings
            best_points = None
            for _ in range(20):
                k = int(1.0 / spacing) + 2
                rows = []
                for i in range(-k, k + 1):
                    for j in range(-k, k + 1):
                        x = (i + 0.5 * (j % 2)) * spacing
                        y = j * spacing * np.sqrt(3.0) / 2.0
                        if np.hypot(x, y) <= 1.0:
                            rows.append([x, y])
                if len(rows) >= n_points:
                    # Sort by distance to origin
                    rows = np.array(rows)
                    dists = np.hypot(rows[:, 0], rows[:, 1])
                    indices = np.argsort(dists, kind='stable')
                    selected = rows[indices[:n_points]]
                    if best_points is None or len(selected) > len(best_points):
                        best_points = selected
                    break
                spacing *= 0.95
            
            if best_points is None:
                # Fallback: Random
                return np.random.rand(n_points, 2) * 2 - 1
            return best_points

        points = generate_hex_lattice(N)
        
        # 2. Optimization Loop
        # We will try to increase the radius r.
        
        def calc_radius(pts):
            # Min distance between any two points / 2
            # And min distance to boundary
            if len(pts) < 2:
                return 1.0
            # Pairwise distances
            # N=300, so 300x300 matrix is fine
            dists = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :]) ** 2, axis=2))
            # Set diagonal to inf
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
            
            # Boundary distance
            radii = np.hypot(pts[:, 0], pts[:, 1])
            min_boundary_dist = 1.0 - np.max(radii)
            
            return min(min_pair_dist / 2.0, min_boundary_dist)

        r = calc_radius(points)
        best_points = points.copy()
        best_r = r
        
        # Optimization parameters
        step_size = 0.01
        noise_level = 0.001
        iterations = 2000
        
        # Pre-calculate indices for pairs? No, just use full matrix for simplicity and speed with numpy.
        
        for it in range(iterations):
            if time.time() > deadline:
                break
                
            # Calculate forces
            forces = np.zeros_like(points)
            
            # Pairwise repulsion
            # dists[i,j] is distance
            dists = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # We want to push apart if dist < 2*r
            # Force is proportional to overlap
            # F_ij = (2*r - dist) * (p_i - p_j) / dist
            # Only if dist < 2*r
            
            # Vectorized calculation
            # diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            # diff shape (N, N, 2)
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            
            # Overlap magnitude
            # overlap = max(0, 2*r - dists)
            overlap = np.maximum(0, 2.0 * r - dists)
            
            # Direction
            # dir = diff / dists
            # Avoid division by zero
            dists_safe = np.where(dists > 1e-8, dists, 1.0)
            dir_vec = diff / dists_safe[:, :, np.newaxis]
            
            # Force on i from j is overlap * dir
            # Sum over j
            # F_i = sum_j overlap[i,j] * dir_vec[i,j]
            # Note: dir_vec[i,j] points from j to i.
            # So if overlap > 0, we push i away from j.
            
            # Multiply overlap by direction
            # overlap shape (N, N), dir_vec shape (N, N, 2)
            # Need to broadcast overlap to (N, N, 1)
            force_pairs = overlap[:, :, np.newaxis] * dir_vec
            
            # Sum over j (axis 1)
            forces += np.sum(force_pairs, axis=1)
            
            # Boundary attraction
            radii = np.hypot(points[:, 0], points[:, 1])
            boundary_overlap = np.maximum(0, radii + r - 1.0)
            # Push towards origin: -points / radius
            # If radius is 0, force is 0
            radii_safe = np.where(radii > 1e-8, radii, 1.0)
            boundary_forces = -boundary_overlap[:, np.newaxis] * points / radii_safe[:, np.newaxis]
            forces += boundary_forces
            
            # Update points
            new_points = points + step_size * forces
            
            # Add noise
            new_points += noise_level * np.random.randn(*new_points.shape)
            
            # Check validity and new radius
            # Ensure points are inside unit circle (hard constraint for validity, but we optimize for max r)
            # The score function returns invalid if any point is outside.
            # So we must ensure new_points are inside.
            new_radii = np.hypot(new_points[:, 0], new_points[:, 1])
            if np.any(new_radii > 1.0):
                # Project back or reject?
                # Projecting might break the "equal circles" assumption if we just clip.
                # Better to reject if any point goes outside.
                # Or scale them back?
                # Let's just reject.
                continue
            
            new_r = calc_radius(new_points)
            
            if new_r > r:
                points = new_points
                r = new_r
                if r > best_r:
                    best_r = r
                    best_points = points.copy()
                
                # Decrease noise if improving
                noise_level *= 0.99
            else:
                # Keep old points, maybe increase noise to escape local min?
                noise_level = min(noise_level * 1.01, 0.01)
            
            # Adjust step size
            # If we are improving, maybe keep step size.
            # If stuck, reduce step size?
            # Simple: keep constant.
        
        return best_points
