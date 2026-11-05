<<<<<<< SEARCH
    def _obj_with_penalty(x):
        n = len(x)
        N_local = n // 3
        centers = x[:2 * N_local].reshape(N_local, 2)
        angles_deg = x[2 * N_local:]
        L = _min_outer_side(centers, angles_deg, 0.0)
        # Penalty for overlaps
        penalty = 0.0
        min_dist = np.sqrt(3.0)  # inradius*2
        for i in range(N_local):
            for j in range(i + 1, N_local):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_dist:
                    penalty += (min_dist - d) ** 2
        return L + 10.0 * penalty
=======
    def _obj_with_penalty(x):
        n = len(x)
        N_local = n // 3
        centers = x[:2 * N_local].reshape(N_local, 2)
        angles_deg = x[2 * N_local:]
        L = _min_outer_side(centers, angles_deg, 0.0)
        
        # Robust penalty for overlaps using SAT
        penalty = 0.0
        for i in range(N_local):
            v1 = _hexagon_vertices(centers[i,0], centers[i,1], np.deg2rad(angles_deg[i]))
            for j in range(i + 1, N_local):
                v2 = _hexagon_vertices(centers[j,0], centers[j,1], np.deg2rad(angles_deg[j]))
                if _polygon_overlap(v1, v2):
                    # Calculate a penalty based on the minimum distance between polygons
                    # A simple proxy: if they overlap, push them apart.
                    # To make it smooth, we can estimate the penetration depth.
                    # For now, a large fixed penalty or distance-based.
                    # Distance between centers is a poor proxy when rotated.
                    # Let's use a large constant penalty to force separation.
                    penalty += 100.0
                    
        return L + penalty
>>>>>>> REPLACE

