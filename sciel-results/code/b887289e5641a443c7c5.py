    def local_search(code_list, max_iters=500):
        """Try to reduce code size by swapping codewords."""
        current = list(code_list)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            # Compute coverage
            cov_count = np.zeros(N, dtype=np.int32)
            for idx in current:
                cov_count[sphere_neighbors[idx]] += 1
            
            # Try removing each codeword and see if we can add a better one
            # First, try simple removal
            to_remove = None
            for idx in current:
                if np.all(cov_count[sphere_neighbors[idx]] >= 2):
                    to_remove = idx
                    break
            
            if to_remove is not None:
                current.remove(to_remove)
                improved = True
                continue
            
            # Try swap: remove one, add another that covers more uncovered
            # Only useful if we can maintain coverage while potentially enabling further removals
            # For now, try: for each pair (i, j), check if removing i and adding j keeps coverage
            # This is O(n^2 * sphere_size) which might be too slow. Skip for now.
            # Instead, try: find a codeword that covers the most "critical" (cov_count==1) positions
            # and see if swapping helps.
            # Simple approach: try each codeword removal, then greedily add back if needed
            for i in range(len(current)):
                idx_to_remove = current[i]
                # Positions that would become uncovered
                would_uncover = np.array([j for j in sphere_neighbors[idx_to_remove] if cov_count[j] == 1])
                if len(would_uncover) == 0:
                    current.pop(i)
                    improved = True
                    break
                # Try to find a single codeword that covers all would_uncover positions
                # and is not already in current
                # This is expensive; only try if code is small enough
                if len(current) < 80 and len(would_uncover) <= 5:
                    # Check all candidates
                    for cand in range(N):
                        if cand in current:
                            continue
                        # Check if cand covers all would_uncover
                        covers_all = True
                        for w in would_uncover:
                            if not sphere_masks[cand, w]:
                                covers_all = False
                                break
                        if covers_all:
                            current[i] = cand
                            improved = True
                            break
                    if improved:
                        break
        return current
