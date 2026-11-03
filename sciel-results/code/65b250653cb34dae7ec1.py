<<<<<<< SEARCH
    def advanced_local_search(selected, time_limit):
        """
        More aggressive local search using reverse_cover for faster candidate generation.
        """
        selected = selected[:]
        if not selected:
            return selected

        counts = compute_counts(selected)
        selected_set = set(selected)

        for _ in range(50):
            if time.time() > time_limit:
                break
            improved = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                m = masks[w]

                # Find uniquely covered words
                unique_words = []
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        unique_words.append(pos)
                    mm ^= lowbit

                if not unique_words:
                    # Remove directly
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    selected.pop(idx)
                    selected_set.remove(w)
                    improved = True
                    continue

                if len(unique_words) > 6:
                    continue

                # Use reverse_cover to find candidates that cover all unique words
                candidate_set = set()
                for uw in unique_words:
                    for cw in reverse_cover[uw]:
                        if cw not in selected_set:
                            candidate_set.add(cw)

                best_w2 = -1
                best_score = -1
                for w2 in candidate_set:
                    if (masks[w2] & m) != m:  # Must cover all of w's unique words
                        continue
                    # Score: how many new words it covers
                    score = (masks[w2] & ~m).bit_count()
                    if score > best_score:
                        best_score = score
                        best_w2 = w2

                if best_w2 >= 0 and best_score > 0:
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    mm = masks[best_w2]
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] += 1
                        mm ^= lowbit
                    selected[idx] = best_w2
                    selected_set.remove(w)
                    selected_set.add(best_w2)
                    improved = True

            if not improved:
                break

        return selected
=======
    def algorithm_x_search(selected, time_limit):
        """
        Custom Algorithm X variant with bit-parallel operations to find a smaller
        covering code. Uses the fact that each word is covered by a small set of
        potential codewords to reduce branching factor.
        """
        selected = selected[:]
        if not selected or time.time() > time_limit:
            return selected

        # Start with current solution, try to reduce via backtracking
        # We'll use a greedy+backtrack hybrid approach
        current_size = len(selected)
        if current_size < 150:
            return selected

        # Build a reduced problem: only consider words that are "critical"
        # and their potential replacements
        counts = compute_counts(selected)
        selected_set = set(selected)

        # Find words with high redundancy (count >= 3) - good candidates for removal
        redundant_words = []
        for w in selected:
            m = masks[w]
            min_count = total_space
            mm = m
            while mm:
                lowbit = mm & -mm
                pos = lowbit.bit_length() - 1
                if counts[pos] < min_count:
                    min_count = counts[pos]
                mm ^= lowbit
            if min_count >= 3:
                redundant_words.append((min_count, w))

        # Sort by minimum count (most redundant first)
        redundant_words.sort(reverse=True)

        improved = False
        for min_count, w in redundant_words[:20]:
            if time.time() > time_limit:
                break
            if w not in selected_set:
                continue

            m = masks[w]

            # Find uniquely covered words
            unique_mask = 0
            mm = m
            while mm:
                lowbit = mm & -mm
                pos = lowbit.bit_length() - 1
                if counts[pos] == 1:
                    unique_mask |= lowbit
                mm ^= lowbit

            if unique_mask == 0:
                # Can remove directly
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    counts[pos] -= 1
                    mm ^= lowbit
                selected.remove(w)
                selected_set.remove(w)
                improved = True
                continue

            # Use reverse_cover to find minimal set of replacements
            unique_words = []
            mm = unique_mask
            while mm:
                lowbit = mm & -mm
                pos = lowbit.bit_length() - 1
                unique_words.append(pos)
                mm ^= lowbit

            if len(unique_words) > 8:
                continue

            # Find candidate replacements using reverse_cover
            candidate_sets = []
            for uw in unique_words:
                candidates = []
                for cw in reverse_cover[uw]:
                    if cw not in selected_set:
                        candidates.append(cw)
                candidate_sets.append(candidates)

            # Try to find a single replacement that covers all unique words
            best_w2 = -1
            best_gain = -1
            for cs in candidate_sets:
                for w2 in cs:
                    if (masks[w2] & unique_mask) == unique_mask:
                        gain = (masks[w2] & ~m).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_w2 = w2

            if best_w2 >= 0:
                # Remove w, add best_w2
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    counts[pos] -= 1
                    mm ^= lowbit
                mm = masks[best_w2]
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    counts[pos] += 1
                    mm ^= lowbit
                selected.remove(w)
                selected.append(best_w2)
                selected_set.remove(w)
                selected_set.add(best_w2)
                improved = True
                continue

            # Try 2-for-1: find two codewords to replace w
            if len(unique_words) <= 4:
                best_pair = None
                best_pair_gain = -1
                for i in range(len(candidate_sets)):
                    for w2 in candidate_sets[i][:10]:
                        for j in range(i+1, len(candidate_sets)):
                            for w3 in candidate_sets[j][:10]:
                                if w2 == w3:
                                    continue
                                if (masks[w2] | masks[w3]) & unique_mask == unique_mask:
                                    gain = ((masks[w2] | masks[w3]) & ~m).bit_count()
                                    if gain > best_pair_gain:
                                        best_pair_gain = gain
                                        best_pair = (w2, w3)

                if best_pair and best_pair_gain > 0:
                    w2, w3 = best_pair
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    for w_new in (w2, w3):
                        mm = masks[w_new]
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] += 1
                            mm ^= lowbit
                    selected.remove(w)
                    selected.extend([w2, w3])
                    selected_set.remove(w)
                    selected_set.update([w2, w3])
                    improved = True

        if improved:
            # Polish the result
            selected = polish(selected, time_limit)

        return selected
>>>>>>> REPLACE

