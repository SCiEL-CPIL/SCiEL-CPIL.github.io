<<<<<<< SEARCH
    def improve(labels):
        """Local improvement via single-vertex flips until no improvement."""
        # Precompute adjacency row sums for speed
        improved = True
        while improved and (time.time() < deadline - 1.0):
            improved = False
            for v in range(n):
                current_label = labels[v]
                # Number of neighbors on the opposite side (would become same after flip)
                # Number of neighbors on the same side (would become opposite after flip)
                same_side = np.sum(adjacency[v] & (labels == current_label))
                opp_side = np.sum(adjacency[v] & (labels != current_label))
                
                # If we flip v, the gain in cut size is:
                # lose opp_side edges that were crossing, gain same_side edges that weren't
                gain = same_side - opp_side
                if gain > 0:
                    labels[v] = 1 - current_label
                    improved = True
                    if time.time() >= deadline - 1.0:
                        return labels
        return labels
=======
    def improve(labels):
        """Local improvement via single-vertex flips until no improvement (vectorized)."""
        improved = True
        while improved and (time.time() < deadline - 1.0):
            improved = False
            # Vectorized: for each vertex, compute gain from flipping
            # gain[v] = (# neighbors on same side as v) - (# neighbors on opposite side)
            # If gain[v] > 0, flipping v increases cut size
            same_side_counts = np.sum(adjacency * (labels[None, :] == labels[:, None]).astype(np.int8), axis=1)
            opp_side_counts = np.sum(adjacency, axis=1) - same_side_counts
            gains = same_side_counts - opp_side_counts
            
            # Find all vertices with positive gain and flip them
            flip_mask = gains > 0
            if np.any(flip_mask):
                labels[flip_mask] = 1 - labels[flip_mask]
                improved = True
                if time.time() >= deadline - 1.0:
                    return labels
        return labels
>>>>>>> REPLACE

