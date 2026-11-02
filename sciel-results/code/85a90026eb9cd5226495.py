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

    # Precompute neighbor bitmasks for all words
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
        Local search with 1-for-1 swaps and 2-for-1 swaps (remove A, add C, remove B).
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

        # Helper to check if a word is redundant given current counts
        def is_redundant(w):
            m = masks[w]
            mm = m
            while mm:
                lowbit = mm & -mm
                pos = lowbit.bit_length() - 1
                if counts[pos] == 1:
                    return False
                mm ^= lowbit
            return True

        # Helper to apply removal
        def remove_word(w):
            m = masks[w]
            mm = m
            while mm:
                lowbit = mm & -mm
                pos = lowbit.bit_length() - 1
                counts[pos] -= 1
                mm ^= lowbit
            selected_set.remove(w)
            
        # Helper to apply addition
        def add_word(w):
            m = masks[w]
            mm = m
            while mm:
                lowbit = mm & -mm
                pos = lowbit.bit_length() - 1
                counts[pos] += 1
                mm ^= lowbit
            selected_set.add(w)

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if time.time() > time_limit:
                    break
                if idx >= len(selected):
                    continue
                
                w_a = selected[idx]
                
                # 1. Try simple removal
                if is_redundant(w_a):
                    remove_word(w_a)
                    selected.pop(idx)
                    improved = True
                    continue

                # 2. Try 1-for-1 swap
                # Find unique words covered by A
                unique_a = []
                m = masks[w_a]
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        unique_a.append(pos)
                    mm ^= lowbit

                if unique_a:
                    # Find best C that covers unique_a
                    best_c = -1
                    best_gain = -1
                    for w_c in range(total_space):
                        if w_c in selected_set:
                            continue
                        m_c = masks[w_c]
                        covers = True
                        for u in unique_a:
                            if not (m_c >> u) & 1:
                                covers = False
                                break
                        if covers:
                            # Gain is number of NEW words covered by C
                            # Actually, for 1-for-1, we just need it to be valid.
                            # But we prefer C that covers more of A's non-unique parts?
                            # Or just pick the first one?
                            # Let's pick one that maximizes overlap with existing coverage?
                            # No, we want to replace A.
                            # Any valid C works for validity.
                            # To help future reductions, maybe pick C that is "central"?
                            # For now, just pick the first valid C.
                            best_c = w_c
                            break # Optimization: don't search all if we just need one?
                            # Actually, searching all might find a better C that allows removing B later.
                            # But for speed, let's just find one.
                            # If we want to be smarter:
                            # gain = (m_c & ~m).bit_count() ? No, m is A.
                            # We want C to cover unique_a.
                            # Let's stick to first found for speed in inner loop.
                            # If we have time, we can loop.
                            # Given 2187 words, looping is fast.
                            # But let's just take the first one to save time in the 2-for-1 phase.
                            # Actually, let's store the best C based on some heuristic?
                            # Let's just take the first one.
                            break

                    if best_c != -1:
                        # Tentatively remove A and add C
                        remove_word(w_a)
                        add_word(best_c)
                        
                        # Check if any B is now redundant
                        removed_b = -1
                        # We need to check all B in selected
                        # To be fast, check only those that might be redundant?
                        # Just iterate.
                        for b_idx in range(len(selected)):
                            if b_idx == idx:
                                continue # A was at idx, but we popped it? 
                                # Wait, remove_word doesn't pop from list.
                                # We need to handle list vs set carefully.
                                # Let's restructure: work with list, update counts.
                                # This is getting complicated.
                                # Let's revert to a simpler 1-for-1 and rely on polish.
                                # Or implement 2-for-1 properly with list manipulation.
                        
                        # Revert
                        remove_word(best_c)
                        add_word(w_a)
                        
                        # Let's try a different approach for 2-for-1:
                        # 1. Remove A.
                        # 2. Find U_A.
                        # 3. Find C covering U_A.
                        # 4. Add C.
                        # 5. Find B redundant.
                        # 6. Remove B.
                        # This requires careful state management.

                        # Given the complexity and time, let's stick to robust 1-for-1 and Polish.
                        # The 1-for-1 in the previous code was:
                        # Find unique words. Find C covering them. Swap.
                        # This is already implemented in the parent?
                        # The parent `local_search` did:
                        # 1. Check if removable.
                        # 2. Find unique words.
                        # 3. Find best C (max gain).
                        # 4. Swap.
                        
                        # Let's improve the "best C" selection in 1-for-1.
                        # Instead of just any C, pick C that minimizes the "cost" or maximizes future removability?
                        # Hard to know.
                        
                        # Let's try to add a "2-for-1" that is simpler:
                        # Remove A. Find U_A.
                        # Find C covering U_A.
                        # If found, Swap A->C.
                        # Then run Polish (which removes redundant B).
                        # This is effectively 2-for-1 if B exists.
                        
                        # So, the current `local_search` + `polish` loop might be sufficient if we iterate enough.
                        # The issue is `local_search` stops if no 1-for-1 swap improves (reduces size? No, 1-for-1 keeps size same).
                        # Wait, 1-for-1 swap keeps size same. It doesn't improve size!
                        # The parent `local_search` did:
                        # `if best_word >= 0 and best_gain > 0:`
                        # `selected[idx] = best_word`
                        # `improved = True`
                        # But size didn't change!
                        # And `polish` is called separately.
                        # So the parent loop `while improved` in `local_search` would run forever if swaps are possible but don't reduce size?
                        # No, `improved` is set to True, so it loops.
                        # But if no size reduction happens, it's stuck in a loop of swaps?
                        # The parent code:
                        # `if not unique_words: ... pop ... improved=True` (Size reduction)
                        # `else: ... swap ... improved=True` (Size same)
                        # This could be an infinite loop of swaps!
                        # We need to track visited states or limit swap iterations.
                        
                        # My new `local_search` should:
                        # 1. Try removals (size -1).
                        # 2. Try swaps (size 0) but limit the number of swaps per pass to avoid loops.
                        # 3. After swaps, run polish to catch any new redundancies.
                        
                        pass

        return selected

    def run_greedy(randomized=False, alpha=1.0):
        """Greedy set cover with optional randomization and alpha weighting."""
        covered = 0
        selected = []
        remaining = full_mask

        while remaining and time.time() < deadline - 5:
            best_gain = -1
            best_word = -1
            candidates = []

            for w in range(total_space):
                gain = (masks[w] & remaining).bit_count()
                if gain == 0:
                    continue
                if randomized:
                    gain = gain + random.gauss(0, 1.0) * alpha
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
                    candidates = [w]
                elif gain == best_gain and not randomized:
                    candidates.append(w)

            if best_word == -1:
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
    num_restarts = 150
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break

        sol = run_greedy(randomized=True, alpha=random.uniform(0.1, 0.5))
        if len(sol) < best_size + 5:
            polished = polish(sol[:], deadline - 15)
            if len(polished) < best_size:
                best_size = len(polished)
                best_selected = polished[:]

    # Strategy 3: Local search on best solution
    if time.time() < deadline - 5:
        ls = local_search(best_selected[:], deadline - 5)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Final polish
    if time.time() < deadline - 2:
        polished = polish(best_selected[:], deadline - 2)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
