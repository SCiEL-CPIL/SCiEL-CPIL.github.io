def _is_valid(pos):
    if not np.all(np.isfinite(pos)):
        return False
    # Check min distance
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    return min_dist_sq > 0.35**2

# In construct:
        if _is_valid(pos_hard) and energy < best_energy:
            best_energy = energy
            best_pos = pos_hard
            
    if best_pos is None:
        # Fallback to first structure
        best_pos = structures[0]
        
    return best_pos
