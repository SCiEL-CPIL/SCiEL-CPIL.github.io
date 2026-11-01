            # Vectorized Pairwise Repulsion
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :] # (N, N, 3)
            dists = np.linalg.norm(diff, axis=2) # (N, N)
            
            # Mask for pairs i < j and dist < 2*target_r
            # We need to be careful with diagonal
            mask = (dists < 2 * target_r) & (dists > 1e-9)
            np.fill_diagonal(mask, False)
            
            # Calculate force magnitude
            # F = (2*r - dist)
            force_mag = np.zeros_like(dists)
            force_mag[mask] = (2 * target_r - dists[mask])
            
            # Force vector = force_mag * diff / dist
            # dist can be 0, but we masked it.
            # To avoid division by zero, use safe_dist
            safe_dists = np.where(dists > 1e-9, dists, 1.0)
            force_vec = force_mag[:, :, np.newaxis] * (diff / safe_dists[:, :, np.newaxis])
            
            # Sum forces for each point
            # force_vec[i, j] is force on i due to j
            forces = np.sum(force_vec, axis=1)
