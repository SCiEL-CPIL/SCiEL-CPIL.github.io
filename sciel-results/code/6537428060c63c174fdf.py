<<<<<<< SEARCH
        rng = random.Random(seed)
        # uncovered[i] = True if t_subset i is not yet covered
        uncovered = np.ones(n_t_subsets, dtype=bool)
        n_uncovered = n_t_subsets
        blocks = []
        
        # Precompute for each point x, which t-subsets contain x
        # points_in_subset[x] = list of t-subset indices that contain point x
        points_in_subset = [[] for _ in range(v)]
        for i, subset in enumerate(all_t_subsets):
            for x in subset:
                points_in_subset[x].append(i)
=======
        rng = random.Random(seed)
        # uncovered[i] = True if t_subset i is not yet covered
        uncovered = np.ones(n_t_subsets, dtype=bool)
        n_uncovered = n_t_subsets
        blocks = []
        
        # Note: points_in_subset is not actually used in the current greedy logic!
        # The gain calculation uses t_subset_to_idx and uncovered directly.
        # So we can remove this block to save time/memory, though it's small.
>>>>>>> REPLACE

Actually, looking at the code, `points_in_subset` is defined but never used in the `greedy_cover` function body shown. The gain calculation is:
