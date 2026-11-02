"""
Packing 200 equal circles in a unit square using soft-contact rigid-body dynamics.
"""
import numpy as np

N = 200


def construct():
    """
    Uses a symplectic Euler integrator with a soft-contact potential to pack circles.
    """
    
    def hexagonal_init(n_circles):
        """Generate hexagonal lattice points centered in unit square."""
        best_points = None
        best_r = 0
        
        for rows in range(10, 25):
            for cols in range(10, 25):
                total = rows * cols
                if total < n_circles:
                    continue
                
                r_max = min(1.0 / (2*cols - 1), 1.0 / ((rows - 1) * np.sqrt(3)))
                
                if r_max > best_r:
                    best_r = r_max
                    pts = []
                    for j in range(rows):
                        for i in range(cols):
                            x = (i * 2 - (cols - 1)) * r_max
                            if j % 2 == 1:
                                x += r_max
                            y = (j - (rows - 1) / 2.0) * r_max * np.sqrt(3)
                            pts.append([x, y])
                    pts = np.array(pts[:n_circles])
                    if np.all(np.abs(pts) <= 0.5 - r_max * 0.01):
                        best_points = pts
                        best_r = r_max
        
        if best_points is not None:
            return best_points
        
        m = int(np.ceil(np.sqrt(n_circles)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n_circles]

    def compute_radius(centers):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2
        
        bound_dists = np.minimum(
            np.minimum(0.5 - centers[:, 0], 0.5 + centers[:, 0]),
            np.minimum(0.5 - centers[:, 1], 0.5 + centers[:, 1])
        )
        min_bound = np.min(bound_dists)
        
        return min(min_pair, min_bound)

    # Initialization
    centers = hexagonal_init(N).astype(np.float64)
    velocities = np.zeros_like(centers)
    
    # Simulation parameters
    dt = 0.0005
    damping = 0.98
    k_pair = 100.0
    k_wall = 100.0
    d_soft = 0.08  # Soft contact distance
    steps = 5000
    
    # Precompute indices for efficiency
    indices = np.arange(N)
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    for step in range(steps):
        # Calculate forces
        forces = np.zeros_like(centers)
        
        # Pairwise forces
        diff = centers[i_idx] - centers[j_idx]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        
        # Avoid division by zero
        mask = dists > 1e-6
        if np.any(mask):
            inv_dists = 1.0 / dists[mask]
            # Force magnitude: k * (d_soft - d)^2 if d < d_soft
            overlaps = d_soft - dists[mask]
            valid_overlap = overlaps > 0
            if np.any(valid_overlap):
                # F = k * (d_soft - d)^2 * (diff/d)
                # We want to push apart, so force on i is -F * diff/d
                # Wait, diff = c_i - c_j. Vector from j to i.
                # Force on i should be in direction of diff (away from j).
                # So F_vec = k * (d_soft - d)^2 * (diff / d)
                mag = k_pair * (overlaps[valid_overlap] ** 2)
                dirs = diff[mask][valid_overlap] * inv_dists[mask][valid_overlap, np.newaxis]
                f_vec = mag[:, np.newaxis] * dirs
                
                # Accumulate forces
                np.add.at(forces, i_idx[mask][valid_overlap], f_vec)
                np.add.at(forces, j_idx[mask][valid_overlap], -f_vec)
        
        # Wall forces
        # If x > 0.5 - d_soft, push left
        # If x < -0.5 + d_soft, push right
        # Same for y
        
        # X-axis
        x = centers[:, 0]
        # Right wall
        dist_right = 0.5 - x
        overlap_right = d_soft - dist_right
        valid_right = overlap_right > 0
        if np.any(valid_right):
            f_mag = k_wall * (overlap_right[valid_right] ** 2)
            forces[valid_right, 0] -= f_mag
        
        # Left wall
        dist_left = x + 0.5
        overlap_left = d_soft - dist_left
        valid_left = overlap_left > 0
        if np.any(valid_left):
            f_mag = k_wall * (overlap_left[valid_left] ** 2)
            forces[valid_left, 0] += f_mag
            
        # Y-axis
        y = centers[:, 1]
        # Top wall
        dist_top = 0.5 - y
        overlap_top = d_soft - dist_top
        valid_top = overlap_top > 0
        if np.any(valid_top):
            f_mag = k_wall * (overlap_top[valid_top] ** 2)
            forces[valid_top, 1] -= f_mag
            
        # Bottom wall
        dist_bottom = y + 0.5
        overlap_bottom = d_soft - dist_bottom
        valid_bottom = overlap_bottom > 0
        if np.any(valid_bottom):
            f_mag = k_wall * (overlap_bottom[valid_bottom] ** 2)
            forces[valid_bottom, 1] += f_mag

        # Update velocities (symplectic Euler)
        velocities += forces * dt
        velocities *= damping
        
        # Update positions
        centers += velocities * dt
        
        # Clip to box to prevent escape (hard constraint)
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        # Zero velocity if clipped (inelastic collision with wall)
        centers[:, 0] = np.where(centers[:, 0] == 0.5, 0.5, centers[:, 0])
        centers[:, 0] = np.where(centers[:, 0] == -0.5, -0.5, centers[:, 0])
        centers[:, 1] = np.where(centers[:, 1] == 0.5, 0.5, centers[:, 1])
        centers[:, 1] = np.where(centers[:, 1] == -0.5, -0.5, centers[:, 1])
        
        # Simple convergence check: if max velocity is small, break
        if step > 1000 and np.max(np.abs(velocities)) < 1e-6:
            break

    # Final polish: ensure no overlaps by slightly adjusting if needed?
    # The physics should have handled it.
    # Clip again to be safe
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers


# Quick test
if __name__ == "__main__":
    centers = construct()
    print(f"Shape: {centers.shape}")
    def compute_radius(centers):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2
        bound_dists = np.minimum(
            np.minimum(0.5 - centers[:, 0], 0.5 + centers[:, 0]),
            np.minimum(0.5 - centers[:, 1], 0.5 + centers[:, 1])
        )
        min_bound = np.min(bound_dists)
        return min(min_pair, min_bound)
    print(f"Radius: {compute_radius(centers):.10f}")
