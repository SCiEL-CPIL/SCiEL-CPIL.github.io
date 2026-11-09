<<<<<<< SEARCH
    def doubled_areas(points):
        """Return dict: doubled_area -> count (positive only), using numpy."""
        n = len(points)
        if n < 3:
            return {}
        pts = np.array(points, dtype=np.int32)
        cnt = defaultdict(int)
        # Use numpy to compute all triple determinants efficiently
        # For each pair (i,j), compute cross product magnitude with all k
        for i in range(n):
            for j in range(i + 1, n):
                # Vector from i to j
                v = pts[j] - pts[i]
                # Vectors from i to all k
                w = pts - pts[i]
                # Cross product z-component: v_x * w_y - v_y * w_x
                # w[:, 1] is y-coords of all vectors from i
                # w[:, 0] is x-coords of all vectors from i
                cross_z = v[0] * w[:, 1] - v[1] * w[:, 0]
                # Only consider k > j
                for k in range(j + 1, n):
                    d = abs(int(cross_z[k]))
                    if d > 0:
                        cnt[d] += 1
        return cnt
=======
    def doubled_areas(points):
        """Return dict: doubled_area -> count (positive only), using numpy."""
        n = len(points)
        if n < 3:
            return {}
        pts = np.array(points, dtype=np.int64)
        # Compute all C(30,3) = 4060 triple determinants at once
        # Use broadcasting: for triples (i,j,k) with i<j<k
        # det = (xj-xi)*(yk-yi) - (yj-yi)*(xk-xi)
        # We'll compute in chunks to manage memory
        cnt = defaultdict(int)
        # Precompute index arrays
        ii, jj, kk = np.triu_indices(n, k=1)
        # This gives all pairs (i,j) with i<j
        # For each such pair, we need all k > j
        # Better approach: generate all valid triples
        # n=30, C(30,3)=4060, so we can do it all at once
        # Use meshgrid approach
        # Actually, let's just loop over i and j but vectorize over k
        for i in range(n):
            for j in range(i + 1, n):
                # Vector from i to j
                v = pts[j] - pts[i]
                # Vectors from i to all k > j
                ks = np.arange(j + 1, n)
                w = pts[ks] - pts[i]
                # Cross product z-component
                cross_z = v[0] * w[:, 1] - v[1] * w[:, 0]
                abs_cross = np.abs(cross_z)
                # Count non-zero values
                nz = abs_cross[abs_cross > 0]
                if len(nz) > 0:
                    # Use bincount for fast counting
                    vals = nz.astype(np.int64)
                    counts = np.bincount(vals)
                    for val, c in zip(np.where(counts > 0)[0], counts[counts > 0]):
                        cnt[int(val)] += int(c)
        return cnt
>>>>>>> REPLACE

Now let me also improve the local search to be more effective with a larger perturbation range:

