import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0

    total_space = q ** n

    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Precompute neighbor bitmasks for each word
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
        """Remove redundant codewords iteratively with multiple passes."""
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
        Local search with 1-for-1 swaps and 2-for-1 swaps.
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

        # 1-for-1 swap and removal
        for _ in range(80):
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

                best_gain = -1
                best_word = -1
                for w2 in range(total_space):
                    if w2 in selected_set:
                        continue
                    m2 = masks[w2]
                    if (m2 & unique_mask) == unique_mask:
                        gain = (m2 & ~m).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_word = w2

                if best_word >= 0 and best_gain > 0:
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
        if time.time() < time_limit - 8:
            for _ in range(15):
                if time.time() > time_limit:
                    break
                improved = False
                indices = list(range(len(selected)))
                random.shuffle(indices)

                for i_idx in range(min(40, len(indices))):
                    idx1 = indices[i_idx]
                    if idx1 >= len(selected):
                        continue
                    w1 = selected[idx1]

                    idx2 = random.randint(0, len(selected) - 1)
                    if idx2 == idx1:
                        continue
                    w2 = selected[idx2]

                    m1 = masks[w1]
                    m2 = masks[w2]

                    u_ab = 0
                    inter = m1 & m2
                    mm = inter
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        if counts[pos] == 2:
                            u_ab |= lowbit
                        mm ^= lowbit

                    diff1 = m1 & ~m2
                    mm = diff1
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        if counts[pos] == 1:
                            u_ab |= lowbit
                        mm ^= lowbit

                    diff2 = m2 & ~m1
                    mm = diff2
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        if counts[pos] == 1:
                            u_ab |= lowbit
                        mm ^= lowbit

                    if u_ab == 0:
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

                    if u_ab.bit_count() > 8:
                        continue

                    best_word = -1
                    for w3 in range(total_space):
                        if w3 in selected_set:
                            continue
                        m3 = masks[w3]
                        if (m3 & u_ab) == u_ab:
                            best_word = w3
                            break

                    if best_word >= 0:
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

    def run_greedy(randomized=False, alpha=1.0):
        """Greedy set cover."""
        covered = 0
        selected = []
        remaining = full_mask
        selected_set = set()

        while remaining and time.time() < deadline - 8:
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

    def solve_min_vertex_cover(selected, time_limit):
        """
        Solve minimum vertex cover in bipartite graph:
        Left nodes = uncovered words (words with count==1 that would be orphaned)
        Right nodes = candidate codewords
        Edge from word i to codeword j if codeword j covers word i AND word i has count 1 (only covered by j)
        
        Actually, for this problem, we use a different approach:
        Find words that are "critical" (covered by few codewords), then use max-flow
        to find the minimum set of codewords to keep/add.
        
        Simplified: Use ILP but with better initial bounds from LP relaxation.
        """
        if not selected or time.time() > time_limit:
            return selected[:]

        m = len(selected)
        if m < 10:
            return selected[:]

        # Build constraint matrix
        from scipy.sparse import csr_matrix
        
        rows = []
        cols = []
        data = []
        
        for j, w in enumerate(selected):
            mm = masks[w]
            while mm:
                lowbit = mm & -mm
                i = lowbit.bit_length() - 1
                rows.append(i)
                cols.append(j)
                data.append(1)
                mm ^= lowbit
        
        A = csr_matrix((data, (rows, cols)), shape=(total_space, m))
        
        c = np.ones(m)
        lb = np.ones(total_space)
        ub = np.full(total_space, np.inf)
        lb_vars = np.zeros(m)
        ub_vars = np.ones(m)
        integrality = np.ones(m)
        
        try:
            res = milp(
                c=c,
                constraints=LinearConstraint(A, lb, ub),
                integrality=integrality,
                bounds=Bounds(lb_vars, ub_vars),
                options={"time_limit": max(1.0, time_limit - time.time() - 2.0)}
            )
            
            if res.status in (0, 1):
                x = res.x
                new_selected = [selected[j] for j in range(m) if x[j] > 0.5]
                if new_selected:
                    covered = 0
                    for w in new_selected:
                        covered |= masks[w]
                    if covered == full_mask:
                        return new_selected
                        
        except Exception:
            pass
        
        return selected[:]

    def structural_reduction(selected, time_limit):
        """
        Identify codewords that can be replaced to reduce count.
        Find words covered by exactly 2 codewords - one of them might be replaceable.
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

        # Find words with count == 1: these are "unique" to one codeword
        # If we remove that codeword, we must add another that covers this word
        # Try to find codewords where the unique words can be covered by fewer new codewords
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                m = masks[w]

                # Find words uniquely covered by w
                unique_words = 0
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        unique_words |= lowbit
                    mm ^= lowbit

                if unique_words == 0:
                    # Can remove w entirely
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

                # Try to find one or two codewords that cover all unique_words
                # and have good coverage of other words
                uw_count = unique_words.bit_count()
                if uw_count > 6:
                    continue

                # Try single replacement
                best_single = -1
                best_single_gain = -1
                for w2 in range(total_space):
                    if w2 in selected_set:
                        continue
                    m2 = masks[w2]
                    if (m2 & unique_words) != unique_words:
                        continue
                    # Check what we gain: m2 covers new words not in m
                    new_cover = (m2 & ~m).bit_count()
                    if new_cover > best_single_gain:
                        best_single_gain = new_cover
                        best_single = w2

                if best_single >= 0:
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    mm = masks[best_single]
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] += 1
                        mm ^= lowbit
                    selected[idx] = best_single
                    selected_set.remove(w)
                    selected_set.add(best_single)
                    improved = True
                    continue

                # Try two replacements (2-for-1)
                if uw_count > 3 and uw_count <= 8:
                    best_pair = (-1, -1)
                    best_pair_gain = -1
                    # Sample candidates
                    candidates = [w2 for w2 in range(total_space) if w2 not in selected_set]
                    random.shuffle(candidates)
                    candidates = candidates[:200]

                    for i_c in range(min(100, len(candidates))):
                        w2 = candidates[i_c]
                        m2 = masks[w2]
                        remaining_uw = unique_words & ~m2
                        if remaining_uw == 0:
                            # w2 alone covers all unique words
                            new_cover = (m2 & ~m).bit_count()
                            if new_cover > best_pair_gain:
                                best_pair_gain = new_cover
                                best_pair = (w2, -1)
                            continue
                        if remaining_uw.bit_count() > 4:
                            continue
                        for j_c in range(i_c + 1, min(i_c + 50, len(candidates))):
                            w3 = candidates[j_c]
                            m3 = masks[w3]
                            if (m3 & remaining_uw) != remaining_uw:
                                continue
                            new_cover = ((m2 | m3) & ~m).bit_count()
                            if new_cover > best_pair_gain:
                                best_pair_gain = new_cover
                                best_pair = (w2, w3)

                    if best_pair[0] >= 0:
                        mm = m
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit
                        selected.pop(idx)
                        selected_set.remove(w)
                        selected.append(best_pair[0])
                        selected_set.add(best_pair[0])
                        mm = masks[best_pair[0]]
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] += 1
                            mm ^= lowbit
                        if best_pair[1] >= 0:
                            selected.append(best_pair[1])
                            selected_set.add(best_pair[1])
                            mm = masks[best_pair[1]]
                            while mm:
                                lowbit = mm & -mm
                                pos = lowbit.bit_length() - 1
                                counts[pos] += 1
                                mm ^= lowbit
                        improved = True

        return selected

    # Strategy 1: Standard greedy + polish + local search
    if time.time() < deadline - 15:
        sol = run_greedy(randomized=False)
        polished = polish(sol[:], deadline - 15)
        ls = local_search(polished[:], deadline - 15)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Strategy 2: Multiple randomized greedy restarts
    for r in range(100):
        if time.time() > deadline - 30:
            break

        sol = run_greedy(randomized=True, alpha=random.uniform(0.1, 0.6))
        if len(sol) < best_size + 8:
            polished = polish(sol[:], deadline - 30)
            ls = local_search(polished[:], deadline - 30)
            if len(ls) < best_size:
                best_size = len(ls)
                best_selected = ls[:]

    # Strategy 3: Two-phase greedy (first cover high-degree words, then fill)
    if time.time() < deadline - 40:
        covered = 0
        selected = []
        remaining = full_mask
        selected_set = set()

        phase1_limit = int(total_space * 0.5)
        while remaining and len(selected) < phase1_limit and time.time() < deadline - 40:
            best_gain = -1
            best_word = -1
            for w in range(total_space):
                if w in selected_set:
                    continue
                gain = (masks[w] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            if best_word < 0:
                break
            selected.append(best_word)
            selected_set.add(best_word)
            covered |= masks[best_word]
            remaining = full_mask & ~covered

        while remaining and time.time() < deadline - 40:
            best_overlap = total_space + 1
            best_word = -1
            for w in range(total_space):
                if w in selected_set:
                    continue
                new_cover = (masks[w] & remaining).bit_count()
                overlap = (masks[w] & covered).bit_count()
                score = new_cover * 10 - overlap
                if score > best_overlap:
                    best_overlap = score
                    best_word = w
            if best_word < 0:
                break
            selected.append(best_word)
            selected_set.add(best_word)
            covered |= masks[best_word]
            remaining = full_mask & ~covered

        polished = polish(selected[:], deadline - 40)
        ls = local_search(polished[:], deadline - 40)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Strategy 4: Structural reduction on the best solution
    if time.time() < deadline - 20:
        reduced = structural_reduction(best_selected[:], deadline - 20)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_selected = reduced[:]

    # Strategy 5: MILP-based optimization on the best solution found so far
    if time.time() < deadline - 5:
        ilp_result = solve_min_vertex_cover(best_selected[:], deadline - 3)
        if len(ilp_result) < best_size:
            best_size = len(ilp_result)
            best_selected = ilp_result[:]

    # Final polish
    if time.time() < deadline - 2:
        polished = polish(best_selected[:], deadline - 2)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    # Final local search pass
    if time.time() < deadline - 1:
        final_ls = local_search(best_selected[:], deadline - 1)
        if len(final_ls) < best_size:
            best_size = len(final_ls)
            best_selected = final_ls[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
