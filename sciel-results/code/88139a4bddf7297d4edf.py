    def relax_positions(centers, steps=500, k_rep=1.0, k_bound=10.0, damping=0.9):
        """
        Relax positions using repulsive forces (vectorized).
        """
        centers = centers.copy()
        velocities = np.zeros_like(centers)
        dt = 0.01
        mask = ~np.eye(N, dtype=bool)
        
        for step in range(steps):
            forces = np.zeros_like(centers)
            
            # Vectorized Pairwise Repulsion
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)
            dist_sq[dist_sq < 1e-8] = 1e-8
            dist = np.sqrt(dist_sq)
            
            # F_vec = k_rep * diff / dist^3
            # dist^3 = dist * dist_sq
            inv_dist_cubed = np.zeros_like(dist_sq)
            inv_dist_cubed[mask] = 1.0 / (dist[mask] * dist_sq[mask])
            
            force_tensor = diff * inv_dist_cubed[:, :, np.newaxis]
            forces = np.sum(force_tensor * k_rep, axis=1)
            
            # Boundary force
            dists_to_center = np.linalg.norm(centers, axis=1)
            # Only apply if close to boundary
            near_boundary = dists_to_center > 0.9
            if np.any(near_boundary):
                # Force towards center
                # F = -k_bound * (dist - 0.9) * (center / dist)
                pen = dists_to_center[near_boundary] - 0.9
                unit_vecs = centers[near_boundary] / dists_to_center[near_boundary, np.newaxis]
                bound_forces = -k_bound * pen[:, np.newaxis] * unit_vecs
                forces[near_boundary] += bound_forces
            
            # Update velocities and positions
            velocities = velocities * damping + forces * dt
            centers += velocities * dt
            
            # Hard constraint: keep inside unit circle
            dists = np.linalg.norm(centers, axis=1)
            mask_out = dists > 1.0
            if np.any(mask_out):
                centers[mask_out] = centers[mask_out] / dists[mask_out, np.newaxis]
            
            # Stop early if forces are small
            if np.max(np.linalg.norm(forces, axis=1)) < 1e-6:
                break
        
        return centers
