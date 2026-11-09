import time
import numpy as np

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 85.0
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # Fast popcount for 18-bit values using 3x6-bit lookup
    pop6 = np.array([i.bit_count() for i in range(64)], dtype=np.uint8)

    def popcount18(arr):
        lo = (arr & 0x3F).astype(np.intp)
        mid = ((arr >> 6) & 0x3F).astype(np.intp)
        hi = ((arr >> 12) & 0x3F).astype(np.intp)
        return pop6[lo].astype(np.uint8) + pop6[mid].astype(np.uint8) + pop6[hi].astype(np.uint8)

    # Precompute weight classes
    all_candidates = list(range(1, 2**N))
    all_candidates.sort(key=lambda x: (x.bit_count(), x))

    # Group by weight for faster candidate selection
    weight_groups = {}
    for c in all_candidates:
        w = c.bit_count()
        if w not in weight_groups:
            weight_groups[w] = []
        weight_groups[w].append(c)

    def check_compatible(words_list, c):
        """Check if c is compatible with all words in words_list."""
        words_arr = np.array(words_list, dtype=np.uint32)
        xor_vals = np.bitwise_xor(words_arr, np.uint32(c))
        bit_counts = popcount18(xor_vals)
        return bit_counts.min() >= MIN_DIST

    def greedy_build(start_words, candidates, deadline):
        """Greedy add candidates to start_words."""
        words_list = start_words[:]
        for c in candidates:
            if time.time() > deadline:
                break
            if check_compatible(words_list, c):
                words_list.append(c)
        return words_list

    def local_search_improve(words_list, deadline, max_swaps=5000):
        """Try to improve code size via swap and add operations."""
        words_set = set(words_list)
        improved = True
        swap_count = 0

        while improved and time.time() < deadline:
            improved = False
            swap_count += 1

            # Try adding: for each candidate weight, try to add a word
            for w in range(1, N + 1):
                if time.time() > deadline:
                    return words_list
                for c in weight_groups.get(w, []):
                    if c in words_set:
                        continue
                    if check_compatible(words_list, c):
                        words_list.append(c)
                        words_set.add(c)
                        improved = True
                        break
                if improved:
                    break

            # Try swap: remove one word, add two
            if not improved and swap_count < max_swaps:
                for i in range(len(words_list)):
                    if time.time() > deadline:
                        return words_list
                    removed = words_list[i]
                    temp_list = words_list[:i] + words_list[i+1:]
                    temp_set = words_set - {removed}

                    # Try to find two words that can be added
                    found_two = False
                    for w1 in range(1, N + 1):
                        if found_two:
                            break
                        for c1 in weight_groups.get(w1, []):
                            if c1 in temp_set:
                                continue
                            if check_compatible(temp_list, c1):
                                temp_list2 = temp_list + [c1]
                                temp_set2 = temp_set | {c1}
                                for w2 in range(w1, N + 1):
                                    for c2 in weight_groups.get(w2, []):
                                        if c2 in temp_set2 or c2 == c1:
                                            continue
                                        if check_compatible(temp_list2, c2):
                                            # Success: swap removed with c1 and c2
                                            words_list = temp_list2 + [c2]
                                            words_set = temp_set2 | {c2}
                                            improved = True
                                            found_two = True
                                            break
                                    if found_two:
                                        break
                        if found_two:
                            break
                    if improved:
                        break

        return words_list

    rng = np.random.default_rng(42)
    best_words = [0]
    best_len = 1

    for attempt in range(100):
        if time.time() > deadline - 3.0:
            break

        if attempt == 0:
            # Initial greedy
            candidates = all_candidates
            words_list = greedy_build([0], candidates, deadline - 3.0)
        elif attempt < 20:
            # Greedy with shuffled weight groups
            candidates = []
            for w in range(1, N + 1):
                group = weight_groups[w][:]
                rng.shuffle(group)
                candidates.extend(group)
            words_list = greedy_build([0], candidates, deadline - 3.0)
        else:
            # Local search improvement on current best
            if time.time() > deadline - 10.0:
                break
            words_list = local_search_improve(best_words[:], deadline - 3.0)

        if len(words_list) > best_len:
            best_len = len(words_list)
            best_words = words_list[:]

        if best_len >= 500:
            break

    return best_words
