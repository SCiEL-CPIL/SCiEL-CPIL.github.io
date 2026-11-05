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
        """Local improvement via single-vertex flips until no improvement."""
        improved = True
        while improved and (time.time() < deadline - 1.0):
            improved = False
            # Vectorized computation of gains for all vertices
            # For vertex v with label l, gain = (neighbors on same side) - (neighbors on opposite side)
            # same_side_count[v] = sum(adjacency[v] & (labels == labels[v]))
            # opp_side_count[v] = sum(adjacency[v] & (labels != labels[v]))
            
            # Compute for label=0 vertices and label=1 vertices
            label_0 = (labels == 0)
            label_1 = (labels == 1)
            
            # For each vertex, compute neighbors on same and opposite sides
            # Using matrix operations for speed
            same_side = np.sum(adjacency * label_0[None, :], axis=1) * label_0 + \
                        np.sum(adjacency * label_1[None, :], axis=1) * label_1
            total_degree = np.sum(adjacency, axis=1)
            opp_side = total_degree - same_side
            
            gain = same_side - opp_side
            
            # Find vertices with positive gain
            candidates = np.where(gain > 0)[0]
            if len(candidates) > 0:
                # Flip all candidates with positive gain (greedy batch flip)
                labels[candidates] = 1 - labels[candidates]
                improved = True
        return labels
>>>>>>> REPLACE

