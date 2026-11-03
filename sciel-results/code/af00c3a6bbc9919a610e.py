import time
import random
import math
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    # Evaluator kills at 150s; leave safety margin
    deadline = start_time + 140.0

    total_space = q ** n

    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Precompute neighbor bitmasks for all words
    # masks[i] is a bitmask of all words within distance 1 of i
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

    full_mask = (1 << total_space) - 1

    # Precompute reverse cover: for each word, which codewords cover it
    # This helps in finding replacements faster (Algorithm X style constraint propagation)
    reverse_cover = [[] for _ in range(total_space)]
    for cw_idx in range(total_space):
        m = masks[cw_idx]
        while m:
            low = m & (-m)
            w_idx = low.bit_length() - 1
            reverse_cover[w_idx].append(cw_idx)
            m ^= low

    best_size = total_space
    best_selected = list(range(total_space))

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    def compute_counts(selected):
        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            mm = masks[w]
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit
        return counts

    def polish(selected, time_limit):
        """Remove redundant codewords iteratively."""
        selected = selected[:]
        if not selected:
            return selected

        counts = compute_counts(selected)
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

    def local_search(selected, time_limit, use_sa=False):
        """
        Local search with 1-for-1 swaps and 2-for-1 swaps.
        Uses reverse_cover for faster candidate generation.
        """
        selected = selected[:]
        if not selected:
            return selected

        counts = compute_counts(selected)
        selected_set = set(selected)
        
        # Helper to find best replacement for a word w to cover unique_mask
        def find_replacement(w, unique_mask, exclude_set):
            if unique_mask == 0:
                return -1, 0
            
            # Pick a word from unique_mask to guide the search
            # Use the word with the fewest covering codewords for better pruning
            # For speed, just pick the first one
            mm = unique_mask
            first_low = mm & -mm
            target_word = first_low.bit_length() - 1
            
            best_gain = -1
            best_word = -1
            
            # Iterate candidates from reverse cover of target_word
            for w2 in reverse_cover[target_word]:
                if w2 in exclude_set:
                    continue
                m2 = masks[w2]
                if (m2 & unique_mask) == unique_mask:
                    # Gain is number of new words covered by w2 that were not covered by w
                    # Note: w is being removed, so we look at coverage provided by w2
                    # that is not provided by any other currently selected word?
                    # Actually, gain in objective function (size) is 0 for 1-for-1 swap.
                    # We want to maintain validity. The "gain" here is heuristic:
                    # prefer w2 that covers more words that are currently covered ONLY by w?
                    # Or just any valid swap.
                    # To break ties or prefer better structure, we can look at
                    # how many words w2 covers that are currently covered by >1 words (redundancy reduction)
                    # vs words covered by 1 word (risk).
                    # Simple heuristic: prefer w2 that covers more total words? No, all cover 15.
                    # Prefer w2 that overlaps less with existing selected words (to spread out)?
                    # Let's stick to the first valid one found, or randomize if multiple.
                    # For determinism in SA, we might want to evaluate a cost.
                    # Cost = - (number of words in unique_mask covered by w2) ... all cover all.
                    # Let's just return the first valid one found.
                    return w2, 1
            
            return -1, 0

        # 1-for-1 swap and removal
        for _ in range(200):
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

                unique_mask = 0
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        unique_mask |= lowbit
                    mm ^= lowbit
                
                if unique_mask == 0:
                    # Can just remove
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

                # Find replacement C that covers unique_mask
                best_word, _ = find_replacement(w, unique_mask, selected_set)
                
                if best_word >= 0:
                    # Swap w with best_word
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

        # 2-for-1 swap (limited)
        if time.time() < time_limit - 5:
            for _ in range(50):
                if time.time() > time_limit:
                    break
                improved = False
                indices = list(range(len(selected)))
                random.shuffle(indices)
                
                # Sample a few pairs
                for i_idx in range(min(100, len(indices))):
                    idx1 = indices[i_idx]
                    if idx1 >= len(selected): continue
                    w1 = selected[idx1]
                    
                    # Pick a random partner w2
                    idx2 = random.randint(0, len(selected)-1)
                    if idx2 == idx1: continue
                    w2 = selected[idx2]
                    
                    m1 = masks[w1]
                    m2 = masks[w2]
                    
                    # Calculate U_AB: words that would become uncovered if w1, w2 removed
                    # A word is in U_AB if:
                    # 1. It is covered by w1 or w2
                    # 2. AND (count[word] == 1) OR (count[word] == 2 AND it's covered by both w1 and w2)
                    
                    u_ab = 0
                    # Words covered by w1
                    mm = m1
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        if counts[pos] == 1:
                            u_ab |= lowbit
                        elif counts[pos] == 2:
                            # Check if covered by w2
                            if (m2 & lowbit) != 0:
                                u_ab |= lowbit
                        mm ^= lowbit
                        
                    # Words covered by w2 but not w1 (already handled by count==1 above, 
                    # but if count==2 and covered by w1, it's safe. 
                    # If count==1, it's in u_ab. 
                    # If count>=2 and not covered by w1, it's safe.)
                    # So we only need to check m1 for count==1 and count==2&in_m2.
                    # What about words in m2 but not m1?
                    # If count==1, they are in m2, not m1. 
                    # My loop above only iterates m1. 
                    # I need to iterate m2 as well for count==1 cases not in m1.
                    
                    mm = m2 & ~m1
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        if counts[pos] == 1:
                            u_ab |= lowbit
                        mm ^= lowbit

                    if u_ab == 0:
                        # Can remove both
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
                        # Remove from list
                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                        selected.pop(idx2)
                        selected.pop(idx1)
                        selected_set.remove(w1)
                        selected_set.remove(w2)
                        improved = True
                        break
                        
                    if u_ab.bit_count() > 8:
                        continue # Too hard to cover with 1 word

                    # Find C that covers u_ab
                    best_word, _ = find_replacement(w1, u_ab, selected_set)
                    # Note: find_replacement uses w1's mask logic, but here we need to cover u_ab.
                    # The function logic is generic: find word covering mask.
                    
                    if best_word >= 0:
                        # Remove w1, w2, Add best_word
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
                            
                        # Update list
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
                    # Polish after a 2-for-1 move
                    selected = polish(selected, time_limit)
                    # Rebuild counts and set
                    counts = compute_counts(selected)
                    selected_set = set(selected)

        return selected

    def run_greedy(randomized=False, alpha=1.0):
        """Greedy set cover."""
        covered = 0
        selected = []
        remaining = full_mask
        selected_set = set()

        while remaining and time.time() < deadline - 5:
            best_gain = -1
            best_word = -1

            for w in range(total_space):
                if w in selected_set:
                    continue
                gain = (masks[w] & remaining).bit_count()
                if randomized:
                    gain = gain + random.gauss(0, 1.0) * alpha
                if gain > best_gain:
                    best_gain = gain
                    best_word = w

            if best_word < 0:
                break

            selected.append(best_word)
            selected_set.add(best_word)
            covered |= masks[best_word]
            remaining = full_mask & ~covered

        return selected

    # Strategy 1: Standard greedy + polish + local search
    if time.time() < deadline - 10:
        sol = run_greedy(randomized=False)
        polished = polish(sol[:], deadline - 10)
        ls = local_search(polished[:], deadline - 10)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Strategy 2: Iterated Local Search
    # Use the best solution found so far as the starting point for ILS
    current_best = best_selected[:]
    
    # Allow time for ILS
    ils_deadline = deadline - 5
    temperature = 1.0
    cooling_rate = 0.95
    
    while time.time() < ils_deadline:
        # Perturb
        perturbed = current_best[:]
        # Remove k random codewords
        k = random.randint(3, 8)
        if len(perturbed) > k:
            to_remove = random.sample(perturbed, k)
            perturbed_set = set(perturbed)
            for w in to_remove:
                perturbed.remove(w)
                perturbed_set.remove(w)
            # Add k random codewords to maintain size roughly
            for _ in range(k):
                new_w = random.randint(0, total_space - 1)
                if new_w not in perturbed_set:
                    perturbed.append(new_w)
                    perturbed_set.add(new_w)
        
        # Polish and Local Search
        polished = polish(perturbed[:], ils_deadline)
        ls = local_search(polished[:], ils_deadline)
        
        if len(ls) < len(current_best):
            current_best = ls[:]
            if len(current_best) < best_size:
                best_size = len(current_best)
                best_selected = current_best[:]
            # Reset temperature on improvement
            temperature = 1.0
        else:
            # Accept worse solution with probability based on temperature?
            # For simplicity, just reject and perturb again, or accept if close.
            # Since we always restart from best, this is effectively ILS with random restarts from best.
            pass
        
        temperature *= cooling_rate

    # Final polish
    if time.time() < deadline - 2:
        polished = polish(best_selected[:], deadline - 2)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
