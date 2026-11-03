import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    # Evaluator kills at 150s; leave safety margin
    deadline = start_time + 142.0

    total_space = q ** n

    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Precompute neighbor lists (sparse representation)
    # Each word covers itself + n*(q-1) neighbors
    neighbors = []
    for i in range(total_space):
        nbrs = [i]
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                nbrs.append(i + diff * powers[pos])
        neighbors.append(nbrs)

    # Also build bitmasks for greedy (needed for fast gain computation)
    masks = [0] * total_space
    for i in range(total_space):
        m = 0
        for nb in neighbors[i]:
            m |= (1 << nb)
        masks[i] = m

    # Precompute coverers for fast local search
    # coverers[x] = list of words w such that x is in neighbors[w]
    coverers = [[] for _ in range(total_space)]
    for w in range(total_space):
        for nb in neighbors[w]:
            coverers[nb].append(w)

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
        Local search with 1-for-1 swaps and 2-for-0/1 swaps.
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

        def update_counts_remove(w):
            mm = masks[w]
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] -= 1
                mm ^= lowbit

        def update_counts_add(w):
            mm = masks[w]
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit

        # 1-for-1 swap and removal
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

                unique_words = []
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        unique_words.append(pos)
                    mm ^= lowbit

                if not unique_words:
                    # Can remove directly
                    update_counts_remove(w)
                    selected.pop(idx)
                    selected_set.remove(w)
                    improved = True
                    continue

                # Try to swap
                # Intersection of coverers for unique words
                candidates = None
                for uw in unique_words:
                    if candidates is None:
                        candidates = set(coverers[uw])
                    else:
                        candidates.intersection_update(coverers[uw])
                    
                    if not candidates:
                        break

                if not candidates:
                    continue

                candidates.discard(w)
                
                best_gain = -1
                best_word = -1
                for w2 in candidates:
                    if w2 in selected_set:
                        continue
                    m2 = masks[w2]
                    gain = (m2 & ~m).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_word = w2

                if best_word >= 0 and best_gain > 0:
                    update_counts_remove(w)
                    update_counts_add(best_word)
                    selected[idx] = best_word
                    selected_set.remove(w)
                    selected_set.add(best_word)
                    improved = True

            if not improved:
                break

        # 2-for-0/1 swap
        if time.time() < time_limit - 5:
            for _ in range(20):
                if time.time() > time_limit:
                    break
                improved = False
                indices = list(range(len(selected)))
                random.shuffle(indices)

                # Sample pairs to avoid O(N^2)
                num_pairs = min(200, len(indices) * (len(indices)-1) // 2)
                for _ in range(num_pairs):
                    if time.time() > time_limit:
                        break
                    i1 = random.randint(0, len(indices)-1)
                    i2 = random.randint(0, len(indices)-1)
                    if i1 == i2:
                        continue
                    idx1 = indices[i1]
                    idx2 = indices[i2]
                    if idx1 >= len(selected) or idx2 >= len(selected):
                        continue
                    
                    w1 = selected[idx1]
                    w2 = selected[idx2]
                    
                    m1 = masks[w1]
                    m2 = masks[w2]
                    
                    # Find U: words that become uncovered if w1, w2 removed
                    # U = { x in N(w1) U N(w2) | counts[x] - I(x in N(w1)) - I(x in N(w2)) == 0 }
                    U = []
                    # Iterate over union of neighbors
                    # To avoid duplicates, we can iterate over m1 | m2
                    mm = m1 | m2
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        mm ^= lowbit
                        
                        c = counts[pos]
                        in1 = 1 if (m1 & lowbit) else 0
                        in2 = 1 if (m2 & lowbit) else 0
                        new_c = c - in1 - in2
                        if new_c == 0:
                            U.append(pos)
                    
                    if not U:
                        # 2-for-0: Remove both
                        update_counts_remove(w1)
                        update_counts_remove(w2)
                        # Remove from list carefully
                        if idx1 < idx2:
                            selected.pop(idx2)
                            selected.pop(idx1)
                        else:
                            selected.pop(idx1)
                            selected.pop(idx2)
                        selected_set.remove(w1)
                        selected_set.remove(w2)
                        improved = True
                        # Need to re-shuffle indices or break? 
                        # Breaking inner loop is safer as indices are now invalid
                        break

                    else:
                        # 2-for-1: Find w3 covering U
                        candidates = None
                        for uw in U:
                            if candidates is None:
                                candidates = set(coverers[uw])
                            else:
                                candidates.intersection_update(coverers[uw])
                            if not candidates:
                                break
                        
                        if not candidates:
                            continue
                        
                        # Exclude w1, w2 and existing selected
                        candidates.discard(w1)
                        candidates.discard(w2)
                        candidates -= selected_set
                        
                        if not candidates:
                            continue
                            
                        # Pick best w3 (max gain)
                        best_gain = -1
                        best_word = -1
                        # Gain is number of words in N(w3) that are NOT in N(w1) U N(w2)
                        # But since we are replacing, any valid w3 works. 
                        # To ensure we don't lose coverage of other words, we already checked U.
                        # We might prefer a w3 that covers more "extra" words to potentially allow future removals.
                        union_m12 = m1 | m2
                        for w3 in candidates:
                            m3 = masks[w3]
                            gain = (m3 & ~union_m12).bit_count()
                            if gain > best_gain:
                                best_gain = gain
                                best_word = w3
                                
                        if best_word >= 0:
                            update_counts_remove(w1)
                            update_counts_remove(w2)
                            update_counts_add(best_word)
                            
                            if idx1 < idx2:
                                selected.pop(idx2)
                                selected[idx1] = best_word
                            else:
                                selected.pop(idx1)
                                selected[idx2] = best_word
                                
                            selected_set.remove(w1)
                            selected_set.remove(w2)
                            selected_set.add(best_word)
                            improved = True
                            break # Break inner loop due to index invalidation

                if not improved:
                    # If no improvement in this pass, we might break outer loop?
                    # Let's just continue to next pass, maybe randomization helps
                    pass

        return selected

    def run_greedy(randomized=False, alpha=1.0):
        """Greedy set cover with optional randomization and alpha weighting."""
        covered = 0
        selected = []
        remaining = full_mask

        while remaining and time.time() < deadline - 5:
            best_gain = -1
            candidates = []

            for w in range(total_space):
                gain = (masks[w] & remaining).bit_count()
                if randomized:
                    gain = gain + random.gauss(0, 1.0) * alpha
                if gain > best_gain:
                    best_gain = gain
                    candidates = [w]
                elif gain == best_gain and not randomized:
                    candidates.append(w)

            if not candidates:
                break

            if randomized and len(candidates) > 1:
                best_word = random.choice(candidates)
            else:
                best_word = candidates[0]

            selected.append(best_word)
            covered |= masks[best_word]
            remaining = full_mask & ~covered

        return selected

    # Strategy 1: Standard greedy + polish
    if time.time() < deadline - 10:
        sol = run_greedy(randomized=False)
        polished = polish(sol[:], deadline - 10)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    # Strategy 2: Multiple randomized greedy restarts with polish
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break

        sol = run_greedy(randomized=True, alpha=random.uniform(0.1, 0.5))
        if len(sol) < best_size + 10:  # Only polish if close
            polished = polish(sol[:], deadline - 20)
            if len(polished) < best_size:
                best_size = len(polished)
                best_selected = polished[:]

    # Strategy 3: Local search on best solution
    if time.time() < deadline - 5:
        ls = local_search(best_selected[:], deadline - 5)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Strategy 4: Polish + local search + polish again
    if time.time() < deadline - 3:
        polished = polish(best_selected[:], deadline - 3)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]
        if time.time() < deadline - 2:
            ls2 = local_search(best_selected[:], deadline - 2)
            if len(ls2) < best_size:
                best_size = len(ls2)
                best_selected = ls2[:]

    # Final polish
    if time.time() < deadline - 1:
        polished = polish(best_selected[:], deadline - 1)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
