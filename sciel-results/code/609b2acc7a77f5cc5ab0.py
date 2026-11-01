<<<<<<< SEARCH
    # 2) Randomized greedy restarts with pruning.
    # Each restart: shuffle candidate order, run greedy, prune, keep best.
    num_restarts = 2000  # cap; will be limited by deadline
    for r in range(num_restarts):
        if time.time() > deadline:
            break

        rng = random.Random(r)
        candidates = list(range(N))
        rng.shuffle(candidates)

        covered = 0
        chosen = []

        while covered != full_mask:
            best_gain = -1
            best_idx = -1
            for idx in candidates:
                new_covered = spheres[idx] & ~covered
                gain = new_covered.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            if best_gain == 0:
                break
            chosen.append(best_idx)
            covered |= spheres[best_idx]

        pruned_greedy = prune(chosen)
        if len(pruned_greedy) < best_size:
            best_size = len(pruned_greedy)
            best_code = [all_words[i] for i in pruned_greedy]

        # Early exit if we hit the known optimum of 73
        if best_size <= 73:
            break
=======
    # 2) Randomized greedy restarts with pruning and local improvement.
    # Each restart: shuffle candidate order, run greedy with active list,
    # apply local improvement, prune, keep best.
    num_restarts = 5000  # cap; will be limited by deadline
    for r in range(num_restarts):
        if time.time() > deadline:
            break

        rng = random.Random(r)
        # Create a shuffled list of candidates
        candidates = list(range(N))
        rng.shuffle(candidates)

        covered = 0
        chosen = []
        active = list(candidates)  # active candidates to consider

        while covered != full_mask and active:
            best_gain = -1
            best_pos = -1
            for pos, idx in enumerate(active):
                new_covered = spheres[idx] & ~covered
                gain = new_covered.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_pos = pos
            if best_gain == 0:
                break
            best_idx = active.pop(best_pos)
            chosen.append(best_idx)
            covered |= spheres[best_idx]

            # Remove candidates that no longer add coverage
            new_active = []
            for idx in active:
                if (spheres[idx] & ~covered).bit_count() > 0:
                    new_active.append(idx)
            active = new_active

        pruned_greedy = prune(chosen)
        
        # Apply local improvement to try to reduce size further
        improved = local_improve(pruned_greedy)
        
        if len(improved) < best_size:
            best_size = len(improved)
            best_code = [all_words[i] for i in improved]
        elif len(pruned_greedy) < best_size:
            best_size = len(pruned_greedy)
            best_code = [all_words[i] for i in pruned_greedy]

        # Early exit if we hit the known optimum of 73
        if best_size <= 73:
            break
>>>>>>> REPLACE

