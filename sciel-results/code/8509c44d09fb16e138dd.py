import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin

    total_space = q ** n

    # Precompute word to index and index to word
    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    # Precompute all words
    all_words = [idx_to_word(i) for i in range(total_space)]

    # Precompute cover masks as Python integers (bitsets)
    cover_masks = [0] * total_space
    for i in range(total_space):
        w = all_words[i]
        mask = 1 << i
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    power = q ** (n - 1 - pos)
                    j = i + (val - orig_val) * power
                    mask |= (1 << j)
        cover_masks[i] = mask

    full_mask = (1 << total_space) - 1

    # Precompute reverse cover: for each word, which codewords cover it
    reverse_cover = [[] for _ in range(total_space)]
    for cw_idx in range(total_space):
        m = cover_masks[cw_idx]
        while m:
            low = m & (-m)
            w_idx = low.bit_length() - 1
            reverse_cover[w_idx].append(cw_idx)
            m ^= low

    # Precompute cover sizes for candidate ordering
    cover_sizes = [cover_masks[i].bit_count() for i in range(total_space)]

    # Greedy Set Cover with random tie-breaking
    def greedy_cover_randomized():
        uncovered = full_mask
        code = []
        while uncovered:
            best_gain = 0
            candidates = []
            for i in range(total_space):
                gain = (cover_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)

            if not candidates:
                break

            best_idx = random.choice(candidates)
            code.append(best_idx)
            uncovered &= ~cover_masks[best_idx]

        return code

    # Local Search: Remove redundant codewords and perform swaps
    def local_search_reduce(code, max_unique=4, max_unique_2for1=6):
        if not code:
            return code

        current_code = code[:]
        current_code_set = set(current_code)

        # Build cover_count array
        cover_count = [0] * total_space
        for idx in current_code:
            m = cover_masks[idx]
            while m:
                low = m & (-m)
                w_idx = low.bit_length() - 1
                cover_count[w_idx] += 1
                m ^= low

        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(current_code)
            n_cw = len(current_code)

            # Phase 1: Removal
            for i in range(n_cw):
                if time.time() > deadline:
                    return current_code

                idx = current_code[i]
                mask = cover_masks[idx]

                removable = True
                m = mask
                while m:
                    low = m & (-m)
                    w_idx = low.bit_length() - 1
                    if cover_count[w_idx] <= 1:
                        removable = False
                        break
                    m ^= low

                if removable:
                    m = mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        cover_count[w_idx] -= 1
                        m ^= low
                    current_code.pop(i)
                    current_code_set.remove(idx)
                    improved = True
                    break

            # Phase 2: Swaps - try to replace one codeword with another that covers all its unique words
            if not improved:
                for i in range(n_cw):
                    if time.time() > deadline:
                        return current_code

                    idx = current_code[i]
                    mask = cover_masks[idx]

                    unique_mask = 0
                    m = mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        if cover_count[w_idx] == 1:
                            unique_mask |= (1 << w_idx)
                        m ^= low

                    if unique_mask == 0:
                        m = mask
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            cover_count[w_idx] -= 1
                            m ^= low
                        current_code.pop(i)
                        current_code_set.remove(idx)
                        improved = True
                        break

                    if unique_mask.bit_count() > max_unique:
                        continue

                    words_in_unique = []
                    m = unique_mask
                    while m:
                        low = m & (-m)
                        w_idx = low.bit_length() - 1
                        words_in_unique.append(w_idx)
                        m ^= low

                    words_in_unique.sort(key=lambda w: len(reverse_cover[w]))

                    candidates = set(reverse_cover[words_in_unique[0]])

                    for w in words_in_unique[1:]:
                        if not candidates:
                            break
                        candidates.intersection_update(reverse_cover[w])

                    candidates.discard(idx)
                    candidates -= current_code_set

                    if candidates:
                        best_candidate = None
                        best_score = -1
                        cand_list = list(candidates)
                        random.shuffle(cand_list)

                        for cand in cand_list:
                            new_cover = (cover_masks[cand] & ~cover_masks[idx]).bit_count()
                            if new_cover > best_score:
                                best_score = new_cover
                                best_candidate = cand
                                if best_score >= (cover_masks[cand]).bit_count() - 2:
                                    break

                        if best_candidate is not None:
                            new_idx = best_candidate

                            m = cover_masks[idx]
                            while m:
                                low = m & (-m)
                                w_idx = low.bit_length() - 1
                                cover_count[w_idx] -= 1
                                m ^= low

                            m = cover_masks[new_idx]
                            while m:
                                low = m & (-m)
                                w_idx = low.bit_length() - 1
                                cover_count[w_idx] += 1
                                m ^= low

                            current_code[i] = new_idx
                            current_code_set.remove(idx)
                            current_code_set.add(new_idx)

                            improved = True
                            break

            # Phase 3: 2-for-1 move - remove 2 codewords, try to cover their union with 1
            if not improved and n_cw >= 2:
                for i in range(n_cw - 1):
                    if time.time() > deadline:
                        return current_code

                    for j in range(i + 1, n_cw):
                        if time.time() > deadline:
                            return current_code

                        idx1 = current_code[i]
                        idx2 = current_code[j]

                        mask1 = cover_masks[idx1]
                        mask2 = cover_masks[idx2]

                        unique_to_1 = mask1 & ~mask2
                        unique_to_2 = mask2 & ~mask1
                        unique_mask = 0
                        m = unique_to_1 | unique_to_2
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            if cover_count[w_idx] == 1:
                                unique_mask |= (1 << w_idx)
                            m ^= low

                        if unique_mask == 0:
                            for idx in [idx1, idx2]:
                                m = cover_masks[idx]
                                while m:
                                    low = m & (-m)
                                    w_idx = low.bit_length() - 1
                                    cover_count[w_idx] -= 1
                                    m ^= low
                            current_code.pop(j)
                            current_code.pop(i)
                            current_code_set.remove(idx1)
                            current_code_set.remove(idx2)
                            improved = True
                            break

                        if unique_mask.bit_count() > max_unique_2for1:
                            continue

                        words_in_unique = []
                        m = unique_mask
                        while m:
                            low = m & (-m)
                            w_idx = low.bit_length() - 1
                            words_in_unique.append(w_idx)
                            m ^= low

                        words_in_unique.sort(key=lambda w: len(reverse_cover[w]))

                        candidates = set(reverse_cover[words_in_unique[0]])
                        for w in words_in_unique[1:]:
                            if not candidates:
                                break
                            candidates.intersection_update(reverse_cover[w])

                        candidates.discard(idx1)
                        candidates.discard(idx2)
                        candidates -= current_code_set

                        if candidates:
                            best_candidate = None
                            best_score = -1
                            cand_list = list(candidates)
                            random.shuffle(cand_list)

                            for cand in cand_list:
                                new_cover = (cover_masks[cand] & (unique_to_1 | unique_to_2)).bit_count()
                                if new_cover > best_score:
                                    best_score = new_cover
                                    best_candidate = cand

                            if best_candidate is not None:
                                for idx in [idx1, idx2]:
                                    m = cover_masks[idx]
                                    while m:
                                        low = m & (-m)
                                        w_idx = low.bit_length() - 1
                                        cover_count[w_idx] -= 1
                                        m ^= low

                                m = cover_masks[best_candidate]
                                while m:
                                    low = m & (-m)
                                    w_idx = low.bit_length() - 1
                                    cover_count[w_idx] += 1
                                    m ^= low

                                current_code.pop(j)
                                current_code.pop(i)
                                current_code.append(best_candidate)
                                current_code_set.remove(idx1)
                                current_code_set.remove(idx2)
                                current_code_set.add(best_candidate)
                                improved = True
                                break

                    if improved:
                        break

        return current_code

    # Branch-and-Bound for Minimum Weight Dominating Set
    def branch_and_bound(uncovered_mask, current_size, best_size, current_code, depth, max_depth=8):
        if time.time() > deadline:
            return current_size, current_code

        if current_size >= best_size:
            return best_size, current_code

        if uncovered_mask == 0:
            return current_size, current_code[:]

        if depth >= max_depth:
            return current_size, current_code

        # Lower bound 1: simple volume bound
        max_cover = max(cover_sizes) if cover_sizes else 1
        vol_lb = current_size + (uncovered_mask.bit_count() + max_cover - 1) // max_cover

        if vol_lb >= best_size:
            return best_size, current_code

        # Choose next codeword: order by potential to reduce uncovered set
        best_gain = 0
        candidates = []
        for i in range(total_space):
            gain = (cover_masks[i] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                candidates = [i]
            elif gain == best_gain and gain > 0:
                candidates.append(i)

        if not candidates:
            return best_size, current_code

        # Limit candidates for efficiency
        if len(candidates) > 20:
            candidates.sort(key=lambda i: cover_sizes[i], reverse=True)
            candidates = candidates[:20]

        random.shuffle(candidates)

        for cand in candidates:
            if time.time() > deadline:
                break

            new_uncovered = uncovered_mask & ~cover_masks[cand]
            new_code = current_code + [cand]
            new_size, new_best_code = branch_and_bound(new_uncovered, current_size + 1, best_size, new_code, depth + 1, max_depth)

            if new_size < best_size:
                best_size = new_size
                current_code = new_best_code

        return best_size, current_code

    # Initial greedy construction
    best_code = greedy_cover_randomized()
    best_code = local_search_reduce(best_code, max_unique=4, max_unique_2for1=6)
    best_size = len(best_code)

    # Try to improve by removing one codeword and re-solving with B&B
    if time.time() < deadline - 40.0:
        for i in range(len(best_code)):
            if time.time() > deadline - 15.0:
                break

            test_code = best_code[:i] + best_code[i+1:]
            test_set = set(test_code)

            uncovered = 0
            for idx in test_code:
                uncovered |= cover_masks[idx]
            uncovered = full_mask & ~uncovered

            if uncovered == 0:
                best_code = test_code
                best_size = len(best_code)
                continue

            if uncovered.bit_count() < 50:  # Only try B&B if the gap is small
                bnb_size, bnb_code = branch_and_bound(uncovered, len(test_code), best_size, test_code, 0, max_depth=6)

                if bnb_size < best_size:
                    best_size = bnb_size
                    best_code = bnb_code

    # Final local search with tighter parameters
    best_code = local_search_reduce(best_code, max_unique=3, max_unique_2for1=4)

    # Run restarts if time permits
    if time.time() < deadline - 20.0:
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            cand_code = greedy_cover_randomized()
            cand_code = local_search_reduce(cand_code, max_unique=4, max_unique_2for1=6)
            if len(cand_code) < len(best_code):
                best_code = cand_code

    return [all_words[i] for i in best_code]
