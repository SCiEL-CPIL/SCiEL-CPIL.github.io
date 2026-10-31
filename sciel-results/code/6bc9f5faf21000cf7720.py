<<<<<<< SEARCH
    if best_pts is None:
        return _make_hexagonal_cluster(N)

    return best_pts
=======
    if best_pts is None:
        return _make_hexagonal_cluster(N)

    # Final sanity check: ensure points are distinct
    d = _pairwise_distances(best_pts)
    np.fill_diagonal(d, np.inf)
    if d.min() < 1e-10:
        # Points too close, add small perturbation
        rng = np.random.default_rng(12345)
        best_pts = best_pts + rng.normal(0, 1e-6, size=best_pts.shape)
    
    return best_pts
>>>>>>> REPLACE
