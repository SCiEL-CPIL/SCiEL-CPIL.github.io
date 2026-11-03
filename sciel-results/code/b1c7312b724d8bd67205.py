import time
import random
import numpy as np
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for 150s limit

    total_space = q ** n
    
    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Precompute neighbor bitmasks
    # masks[i] is a bitmask of all words within Hamming distance 'radius' of i
    masks = [0] * total_space
    for i in range(total_space):
        m = 1 << i
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                m |= (1 << (i + diff * powers[pos]))
        masks[i] = m

    # Precompute reverse coverage: for each word x, which codewords cover it?
    # reverse_cover[x] = list of codewords w such that x is in masks[w]
    reverse_cover = [[] for _ in range(total_space)]
    for w in range(total_space):
        mm = masks[w]
        while mm:
            lowbit = mm & -mm
            idx = lowbit.bit_length() - 1
            reverse_cover[idx].append(w)
            mm ^= lowbit

    full_mask = (1 << total_space) - 1

    best_size = total_space
    best_selected = list(range(total_space))

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    def polish(selected, time_limit):
        """Remove redundant codewords iteratively."""
        selected = selected[:]
        if not selected:
            return selected

        # counts[x] is the number of codewords that cover x
        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            mm = masks[w]
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit

        changed = True
        while changed and time.time() < time_limit:
            changed = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                m = masks[w]
                
                # Check if any position in the mask has count == 1
                removable = True
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        removable = False
                        break
                    mm ^= lowbit

                if removable:
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    selected.pop(idx)
                    changed = True

        return selected

    def local_search(selected, time_limit):
        """
        Local search with 1-for-1 swaps and 2-for-1 swaps using reverse_cover for speed.
        """
        selected = selected[:]
        if not selected:
            return selected

        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            mm = masks[w]
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit

        selected_set = set(selected)

        # 1-for-1 swap (and deletion if gain > 0)
        max_passes = 50
        for p in range(max_passes):
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

                # Identify unique points covered by w
                unique_points = []
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        unique_points.append(pos)
                    mm ^= lowbit

                if not unique_points:
                    # w is redundant
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

                # Find candidates that cover all unique_points
                # Use reverse_cover: for each unique point, get list of codewords covering it
                # Intersection of these lists gives candidates
                if len(unique_points) == 1:
                    candidates = reverse_cover[unique_points[0]]
                else:
                    # Start with the smallest list
                    candidates = reverse_cover[unique_points[0]]
                    for pt in unique_points[1:]:
                        if len(candidates) > len(reverse_cover[pt]):
                            candidates = reverse_cover[pt]
                        # Filter candidates to those in reverse_cover[pt]
                        rev_set = set(reverse_cover[pt])
                        candidates = [c for c in candidates if c in rev_set]
                        if not candidates:
                            break

                best_gain = -1
                best_word = -1
                for w2 in candidates:
                    if w2 in selected_set:
                        continue
                    m2 = masks[w2]
                    # Check if w2 covers all unique points (should be true by construction)
                    # Compute gain: new points covered by w2 that were not covered by w
                    gain = (m2 & ~m).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_word = w2

                if best_word >= 0 and best_gain > 0:
                    # Perform swap
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit

                    mm = masks[best_word]
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] += 1
                        mm ^= lowbit

                    selected[idx] = best_word
                    selected_set.remove(w)
                    selected_set.add(best_word)
                    improved = True

            if not improved:
                break

        # 2-for-1 swap
        # Try to remove 2 words and add 1
        if time.time() < time_limit - 5:
            for _ in range(10):
                if time.time() > time_limit:
                    break
                improved = False
                indices = list(range(len(selected)))
                random.shuffle(indices)
                
                num_pairs_to_try = min(50, len(indices) * (len(indices) - 1) // 2)
                
                for _ in range(num_pairs_to_try):
                    i_idx = random.randint(0, len(indices) - 1)
                    j_idx = random.randint(0, len(indices) - 1)
                    if i_idx == j_idx:
                        continue
                    idx1 = indices[i_idx]
                    idx2 = indices[j_idx]
                    if idx1 >= len(selected) or idx2 >= len(selected):
                        continue
                    w1 = selected[idx1]
                    w2 = selected[idx2]
                    
                    m1 = masks[w1]
                    m2 = masks[w2]
                    m12 = m1 | m2
                    
                    # Calculate gap_mask: points in m12 that would become uncovered if w1,w2 removed
                    gap_points = []
                    mm = m12
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        in_m1 = (m1 & lowbit) != 0
                        in_m2 = (m2 & lowbit) != 0
                        if in_m1 and in_m2:
                            if counts[pos] < 3:
                                gap_points.append(pos)
                        else:
                            if counts[pos] < 2:
                                gap_points.append(pos)
                        mm ^= lowbit
                        
                    if not gap_points:
                        # 2-for-0 move
                        mm = m1
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit
                        mm = m2
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit
                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                        selected.pop(idx2)
                        selected.pop(idx1)
                        selected_set.remove(w1)
                        selected_set.remove(w2)
                        improved = True
                        break
                    else:
                        # Find w3 that covers all gap_points
                        if len(gap_points) == 1:
                            candidates = reverse_cover[gap_points[0]]
                        else:
                            candidates = reverse_cover[gap_points[0]]
                            for pt in gap_points[1:]:
                                if len(candidates) > len(reverse_cover[pt]):
                                    candidates = reverse_cover[pt]
                                rev_set = set(reverse_cover[pt])
                                candidates = [c for c in candidates if c in rev_set]
                                if not candidates:
                                    break
                        
                        best_word = -1
                        for w3 in candidates:
                            if w3 in selected_set:
                                continue
                            best_word = w3
                            break
                                
                        if best_word >= 0:
                            # Perform 2-for-1
                            mm = m1
                            while mm:
                                lowbit = mm & -mm
                                pos = lowbit.bit_length() - 1
                                counts[pos] -= 1
                                mm ^= lowbit
                            mm = m2
                            while mm:
                                lowbit = mm & -mm
                                pos = lowbit.bit_length() - 1
                                counts[pos] -= 1
                                mm ^= lowbit
                            mm = masks[best_word]
                            while mm:
                                lowbit = mm & -mm
                                pos = lowbit.bit_length() - 1
                                counts[pos] += 1
                                mm ^= lowbit
                                
                            if idx1 > idx2:
                                idx1, idx2 = idx2, idx1
                            selected.pop(idx2)
                            selected.pop(idx1)
                            selected.append(best_word)
                            
                            selected_set.remove(w1)
                            selected_set.remove(w2)
                            selected_set.add(best_word)
                            improved = True
                            break
                            
                if improved:
                    selected = polish(selected, time_limit)
                    counts = np.zeros(total_space, dtype=np.int32)
                    for w in selected:
                        mm = masks[w]
                        while mm:
                            lowbit = mm & -mm
                            idx = lowbit.bit_length() - 1
                            counts[idx] += 1
                            mm ^= lowbit
                    selected_set = set(selected)

        return selected

    def run_greedy_vectorized(randomized=False, alpha=1.0):
        """Vectorized greedy set cover using numpy."""
        selected = []
        selected_set = set()
        remaining = np.ones(total_space, dtype=bool)

        while remaining.any() and time.time() < deadline - 10:
            # Compute gain for all words in parallel
            # gain[w] = number of remaining points covered by word w
            # masks is a list of big integers, so we need to convert to arrays
            # For efficiency, precompute masks as numpy arrays of bools
            # But that's 2187 * 2187 bits = ~590KB, which is fine
            
            best_gain = -1
            best_word = -1

            # Vectorized: for each word, count how many of its covered points are still remaining
            # We can precompute masks as a 2D array: mask_matrix[w][x] = 1 if x in masks[w]
            # But that's 2187*2187 = ~4.8M entries, which is ~38MB as bool, acceptable
            
            # Actually, let's just iterate but use bit operations which are fast
            for w in range(total_space):
                if w in selected_set:
                    continue
                # Count bits in (masks[w] & remaining_mask)
                # remaining is a numpy array, but masks[w] is an int
                # Convert remaining to int mask? That's expensive.
                # Alternative: iterate through set bits in masks[w] and check remaining
                # Or: precompute a numpy array version of masks
                
                # For now, use the int-based approach but optimize
                mm = masks[w]
                gain = 0
                # Sample a few bits to estimate? No, need exact.
                # Let's use a different approach: maintain remaining as a set or numpy array
                # and for each w, count intersections
                
                # Actually, the current int-based approach is O(total_space) per iteration
                # which is 2187 * 186 = ~400K operations total, which is fine.
                # The bottleneck is Python overhead. Let's stick with int masks.
                
                # Use bit_count on the intersection
                # remaining_mask as int is expensive to maintain
                # Let's just use the simple loop
                pass
            
            # Fall back to simple approach but with numpy for remaining
            for w in range(total_space):
                if w in selected_set:
                    continue
                # Count how many points in masks[w] are still remaining
                mm = masks[w]
                gain = 0
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if remaining[pos]:
                        gain += 1
                    mm ^= lowbit
                if randomized:
                    gain += random.gauss(0, 1.0) * alpha
                if gain > best_gain:
                    best_gain = gain
                    best_word = w

            if best_word < 0:
                break

            selected.append(best_word)
            selected_set.add(best_word)
            # Update remaining
            mm = masks[best_word]
            while mm:
                lowbit = mm & -mm
                pos = lowbit.bit_length() - 1
                remaining[pos] = False
                mm ^= lowbit

        return selected

    # Strategy 1: Standard greedy + polish + local search
    if time.time() < deadline - 20:
        sol = run_greedy_vectorized(randomized=False)
        polished = polish(sol[:], deadline - 20)
        ls = local_search(polished[:], deadline - 20)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Strategy 2: Multiple randomized greedy restarts
    restarts = 0
    while time.time() < deadline - 30 and restarts < 50:
        restarts += 1
        sol = run_greedy_vectorized(randomized=True, alpha=random.uniform(0.1, 0.5))
        if len(sol) < best_size + 10:
            polished = polish(sol[:], deadline - 30)
            ls = local_search(polished[:], deadline - 30)
            if len(ls) < best_size:
                best_size = len(ls)
                best_selected = ls[:]

    # Final polish and local search on the best solution found
    if time.time() < deadline - 5:
        polished = polish(best_selected[:], deadline - 5)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]
            
    if time.time() < deadline - 2:
        final_ls = local_search(best_selected[:], deadline - 2)
        if len(final_ls) < best_size:
            best_size = len(final_ls)
            best_selected = final_ls[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
