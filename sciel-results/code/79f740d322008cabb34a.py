        def compute_score(points):
            # Pairwise distances
            # points: (N, 2)
            # diff: (N, N, 2) -> too big? 300*300*2 = 180,000 floats. Fine.
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            # Set diagonal to inf
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            
            radii = np.linalg.norm(points, axis=1)
            min_boundary = np.min(1.0 - radii)
            
            return min(min_dist / 2.0, min_boundary)
        