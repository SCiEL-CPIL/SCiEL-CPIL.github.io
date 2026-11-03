from scipy.spatial.distance import cdist

def _compute_radius(pts):
    if len(pts) == 0:
        return 0.0
    dist = cdist(pts, pts, metric='euclidean')
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    
    dist_to_bounds = np.minimum(
        np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
        np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
    )
    min_bound = np.min(dist_to_bounds)
    
    return min(min_pair, 2 * min_bound) / 2
