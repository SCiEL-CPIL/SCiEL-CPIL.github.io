<<<<<<< SEARCH
    def contains_q3_in_subset(v_list):
        """Check if the induced subgraph on v_list (8 vertices) contains Q3."""
        # Build local adjacency bitmasks
        local_adj = [0] * 8
        for i in range(8):
            vi = v_list[i]
            mask = 0
            for j in range(8):
                if i != j and adj[vi][v_list[j]]:
                    mask |= (1 << j)
            local_adj[i] = mask

        for mask_a, mask_b, A_idx, B_idx in all_splits:
            covered = 0
            is_matching = True
            missing_count = 0

            for i in A_idx:
                for j in B_idx:
                    if not (local_adj[i] & (1 << j)):
                        missing_count += 1
                        if missing_count > 4:
                            is_matching = False
                            break
                        if (covered & (1 << i)) or (covered & (1 << j)):
                            is_matching = False
                            break
                        covered |= (1 << i)
                        covered |= (1 << j)
                if not is_matching:
                    break
            if is_matching:
                return True
        return False
=======
    def contains_q3_in_subset_fast(v_list):
        """Fast check: does the induced subgraph on these 8 vertices contain Q3?
        Uses precomputed local adjacency bitmasks and the 35 bipartitions.
        """
        # Build local adjacency as bitmasks (8 bits each)
        local_adj = [0] * 8
        for i in range(8):
            vi = v_list[i]
            mask = 0
            for j in range(8):
                if i != j and adj[vi][v_list[j]]:
                    mask |= (1 << j)
            local_adj[i] = mask

        for mask_a, mask_b, A_idx, B_idx in all_splits:
            # For each vertex in A, check if it can be matched to a vertex in B
            # such that all 4 edges from A to B form a perfect matching
            # We need: for each i in A, there exists j in B with edge (i,j),
            # and the matching covers all 8 vertices
            # Use a simple backtracking on the 4 vertices in A
            matched_b = [False] * 4
            def backtrack(a_idx_pos):
                if a_idx_pos == 4:
                    return True
                i = A_idx[a_idx_pos]
                for b_pos in range(4):
                    if matched_b[b_pos]:
                        continue
                    j = B_idx[b_pos]
                    if local_adj[i] & (1 << j):
                        matched_b[b_pos] = True
                        if backtrack(a_idx_pos + 1):
                            return True
                        matched_b[b_pos] = False
                return False
            if backtrack(0):
                return True
        return False

    def count_q3_subgraphs():
        """Count the number of 8-vertex subsets that contain a Q3."""
        count = 0
        for subset in all_8_subsets:
            if contains_q3_in_subset_fast(subset):
                count += 1
        return count
>>>>>>> REPLACE

