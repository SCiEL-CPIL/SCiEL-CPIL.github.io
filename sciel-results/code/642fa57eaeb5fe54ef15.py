def calc_score(points):
    # points: (14, 3)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    min_d2 = np.min(dists_sq)
    max_d2 = np.max(dists_sq)
    if min_d2 < 1e-9:
        return 1e10
    return max_d2 / min_d2
