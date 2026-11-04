<<<<<<< SEARCH
def _is_valid(params, side, n, min_sep=1e-8, outer_angle_deg=0.0):
    """Check if the configuration is valid."""
    # Re-check side length strictly
    side_check, _ = _fast_side_and_penalty(params, n, outer_angle_deg)
    if side_check > side + 1e-9:
        return False
        
    # Check non-overlap strictly
    centers = params[:, :2]
    dists = np.linalg.norm(centers[:, np.newaxis] - centers[np.newaxis, :], axis=2)
    for i in range(n):
        for j in range(i+1, n):
            if dists[i, j] < 2.0:
                v1 = _hex_vertices(params[i, 0], params[i, 1], params[i, 2])
                v2 = _hex_vertices(params[j, 0], params[j, 1], params[j, 2])
                sep = _poly_separation(v1, v2)
                if sep < min_sep:
                    return False
    return True
=======
def _is_valid(params, side, n, min_sep=1e-8, outer_angle_deg=0.0):
    """Check if the configuration is valid."""
    # Re-check side length strictly
    side_check, _ = _fast_side_and_penalty(params, n, outer_angle_deg)
    if side_check > side + 1e-9:
        return False
        
    # Check non-overlap strictly using SAT
    centers = params[:, :2]
    dists = np.linalg.norm(centers[:, np.newaxis] - centers[np.newaxis, :], axis=2)
    for i in range(n):
        for j in range(i+1, n):
            if dists[i, j] < 2.5:
                v1 = _hex_vertices(params[i, 0], params[i, 1], params[i, 2])
                v2 = _hex_vertices(params[j, 0], params[j, 1], params[j, 2])
                sep = _sat_separation(v1, v2)
                if sep < min_sep:
                    return False
    return True
>>>>>>> REPLACE

