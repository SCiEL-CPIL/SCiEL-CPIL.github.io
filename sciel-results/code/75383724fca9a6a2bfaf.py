            valid_circs = []
            valid_rads = []
            valid_simplices = []
            
            for i, simplex in enumerate(simplices):
                p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
                c = spherical_circumcenter(p1, p2, p3)
                if c is None:
                    continue
                valid_circs.append(c)
                dot_val = np.dot(c, p1)
                dot_val = np.clip(dot_val, -1.0, 1.0)
                rad = np.arccos(dot_val)
                valid_rads.append(rad)
                valid_simplices.append(simplex)
                
            if not valid_rads:
                break
                
            valid_rads = np.array(valid_rads)
            current_max_radius = np.max(valid_rads)
            if current_max_radius < best_radius:
                best_radius = current_max_radius
                best_points = points.copy()
                
            # Sort valid facets by radius
            sorted_indices = np.argsort(valid_rads)[::-1]
            K = 3
            worst_indices = sorted_indices[:min(K, len(valid_rads))]
            
            pull = np.zeros((N, 3))
            for idx in worst_indices:
                simplex = valid_simplices[idx]
                c = valid_circs[idx]
                for v_idx in simplex:
                    p = points[v_idx]
                    pull[v_idx] += (c - p)
