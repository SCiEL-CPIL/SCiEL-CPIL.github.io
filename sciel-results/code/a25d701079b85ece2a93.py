def construct():
    # ... initialization ...
    
    # Get initial pairs
    def get_active_pairs(centres, k=10):
        # Compute distances
        dists = pdist(centres)
        # Convert to matrix
        dist_mat = squareform(dists)
        # For each row, find k smallest
        pairs = set()
        for i in range(n):
            # Exclude self
            dists_i = dist_mat[i].copy()
            dists_i[i] = np.inf
            # Get indices of k smallest
            idx = np.argpartition(dists_i, k)[:k]
            for j in idx:
                if i < j:
                    pairs.add((i, j))
                else:
                    pairs.add((j, i))
        return list(pairs)

    # ... loop ...
        # Update pairs
        current_centres = x0.reshape(n, 2)
        active_pairs = get_active_pairs(current_centres, k=15)
        
        def residuals(x):
            centres = x.reshape(n, 2)
            
            res_list = []
            
            # Pair residuals
            for i, j in active_pairs:
                d = np.linalg.norm(centres[i] - centres[j])
                val = 2.0 * r_target - d
                res_list.append(0.5 * (val + np.sqrt(val**2 + delta**2)))
                
            # Boundary residuals
            norms = np.linalg.norm(centres, axis=1)
            val_bound = r_target - (1.0 - norms)
            res_bound = 0.5 * (val_bound + np.sqrt(val_bound**2 + delta**2))
            res_list.extend(res_bound)
            
            return np.array(res_list)
        
        # Run least squares
        # ...
