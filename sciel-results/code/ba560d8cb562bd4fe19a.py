    def ensure_separation(centers, eps=1e-4):
        """
        Scales the centers relative to their centroid by (1+eps) to ensure
        adjacent hexagons have a gap > 1e-6.
        """
        if len(centers) == 0:
            return centers
        centroid = np.mean(centers, axis=0)
        scaled = centroid + (1.0 + eps) * (centers - centroid)
        return scaled
