import time
import random

def construct(q, n, radius):
    start = time.monotonic()
    deadline = start + 115.0

    total = q ** n
    # Map word to index
    def word_to_idx(w):
        idx = 0
        for x in w:
            idx = idx * q + x
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute balls as bitmasks
    # ball_mask[i] is an int with bits set for all words within distance 1 of i
    ball_masks = [0] * total
    for idx in range(total):
        w = idx_to_word(idx)
        mask = 1 << idx
        for i in range(n):
            orig = w[i]
            for v in range(q):
                if v != orig:
                    w[i] = v
                    n_idx = word_to_idx(w)
                    mask |= (1 << n_idx)
            w[i] = orig
        ball_masks[idx] = mask

    full_mask = (1 << total) - 1

    # Greedy Set Cover to get initial code
    code = []
    covered_mask = 0
    uncovered_count = total

    while uncovered_count > 0:
        best_idx = -1
        best_gain = 0
        for i in range(total):
            gain = bin(ball_masks[i] & ~covered_mask).count('1')
            if gain > best_gain:
                best_gain = gain
                best_idx = i

        if best_idx == -1 or best_gain == 0:
            break

        code.append(best_idx)
        covered_mask |= ball_masks[best_idx]
        uncovered_count = total - bin(covered_mask).count('1')

    # Local Search
    def get_union_mask(code_list):
        m = 0
        for c in code_list:
            m |= ball_masks[c]
        return m

    def try_delete_redundant(code_list, union_m):
        """
        Returns (new_code, new_union, deleted_count)
        """
        deleted = 0
        changed = True
        while changed:
            changed = False
            for i in range(len(code_list) - 1, -1, -1):
                c = code_list[i]
                # Check if c is redundant: all bits in ball_masks[c] are covered by others
                others_m = 0
                for j in range(len(code_list)):
                    if j != i:
                        others_m |= ball_masks[code_list[j]]
                if (others_m & ball_masks[c]) == ball_masks[c]:
                    # Redundant
                    code_list.pop(i)
                    deleted += 1
                    changed = True
                    break # Restart loop
        return code_list, union_m, deleted

    # Main Local Search Loop
    current_code = list(code)
    current_union = get_union_mask(current_code)

    # Initial cleanup
    current_code, current_union, _ = try_delete_redundant(current_code, current_union)

    rng = random.Random(42)
    best_size = len(current_code)
    best_code = list(current_code)

    # Precompute: for each codeword, which words does it exclusively cover?
    # exclusive[c] = ball_masks[c] & ~(OR of all other ball_masks)
    # This helps quickly determine if removing c would break coverage

    iterations = 0
    while time.monotonic() < deadline:
        iterations += 1

        # 1. Try to delete
        new_code, new_union, deleted = try_delete_redundant(list(current_code), current_union)
        if deleted > 0:
            current_code = new_code
            current_union = new_union
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
            continue # Restart loop

        # 2. Try swaps - optimized version
        swap_found = False
        c_old_list = list(current_code)
        rng.shuffle(c_old_list)

        # Precompute union without each codeword for faster swap evaluation
        # union_without[i] = OR of ball_masks[j] for all j != i
        union_without = {}
        for i, c in enumerate(current_code):
            m = 0
            for j, c2 in enumerate(current_code):
                if j != i:
                    m |= ball_masks[c2]
            union_without[c] = m

        for c_old in c_old_list:
            if time.monotonic() > deadline - 5:
                break

            # Candidates for c_new: all words not in current_code
            c_new_list = [i for i in range(total) if i not in current_code]
            rng.shuffle(c_new_list)

            base_union = union_without[c_old]

            for c_new in c_new_list:
                # Quick check: does c_new cover all the holes left by removing c_old?
                # Holes = bits in ball_masks[c_old] not covered by others
                # If c_new covers all holes, the swap is valid
                holes = ball_masks[c_old] & ~base_union
                if (ball_masks[c_new] & holes) == holes:
                    # Valid swap!
                    temp_code = [c for c in current_code if c != c_old]
                    temp_code.append(c_new)
                    temp_union = base_union | ball_masks[c_new]

                    # Try to delete redundant nodes from this new set
                    temp_code, temp_union, del_count = try_delete_redundant(temp_code, temp_union)

                    if len(temp_code) < len(current_code):
                        # Improvement!
                        current_code = temp_code
                        current_union = temp_union
                        if len(current_code) < best_size:
                            best_size = len(current_code)
                            best_code = list(current_code)
                        swap_found = True
                        break # Break c_new loop
                    elif len(temp_code) == len(current_code):
                        # Same size, but maybe different structure. 
                        # Accept with small probability to explore?
                        if rng.random() < 0.1:
                            current_code = temp_code
                            current_union = temp_union
                            swap_found = True
                            break

            if swap_found:
                break # Break c_old loop

        if not swap_found:
            # Stuck. Perturb.
            if len(current_code) > 10:
                # Try different perturbation strategies
                strategy = rng.randint(0, 2)

                if strategy == 0:
                    # Remove 1 random, repair with greedy
                    idx_to_remove = rng.randint(0, len(current_code) - 1)
                    current_code.pop(idx_to_remove)
                    current_union = get_union_mask(current_code)

                    # Greedy repair
                    while current_union != full_mask:
                        best_idx = -1
                        best_gain = 0
                        for i in range(total):
                            if i in current_code:
                                continue
                            gain = bin(ball_masks[i] & ~current_union).count('1')
                            if gain > best_gain:
                                best_gain = gain
                                best_idx = i
                        if best_idx == -1:
                            break
                        current_code.append(best_idx)
                        current_union |= ball_masks[best_idx]

                    # Cleanup
                    current_code, current_union, _ = try_delete_redundant(current_code, current_union)

                elif strategy == 1:
                    # Remove 2 random, repair with greedy
                    if len(current_code) > 12:
                        idx1 = rng.randint(0, len(current_code) - 1)
                        idx2 = rng.randint(0, len(current_code) - 1)
                        if idx1 == idx2:
                            idx2 = (idx2 + 1) % len(current_code)
                        current_code.pop(max(idx1, idx2))
                        current_code.pop(min(idx1, idx2))
                        current_union = get_union_mask(current_code)

                        # Greedy repair
                        while current_union != full_mask:
                            best_idx = -1
                            best_gain = 0
                            for i in range(total):
                                if i in current_code:
                                    continue
                                gain = bin(ball_masks[i] & ~current_union).count('1')
                                if gain > best_gain:
                                    best_gain = gain
                                    best_idx = i
                            if best_idx == -1:
                                break
                            current_code.append(best_idx)
                            current_union |= ball_masks[best_idx]

                        # Cleanup
                        current_code, current_union, _ = try_delete_redundant(current_code, current_union)

                else:
                    # Swap 2 random codewords with random new ones, then repair
                    if len(current_code) > 12:
                        idx1 = rng.randint(0, len(current_code) - 1)
                        idx2 = rng.randint(0, len(current_code) - 1)
                        if idx1 == idx2:
                            idx2 = (idx2 + 1) % len(current_code)

                        # Pick random replacements
                        not_in_code = [i for i in range(total) if i not in current_code]
                        if not_in_code:
                            new1 = rng.choice(not_in_code)
                            not_in_code.remove(new1)
                            if not_in_code:
                                new2 = rng.choice(not_in_code)

                                current_code[idx1] = new1
                                current_code[idx2] = new2
                                current_union = get_union_mask(current_code)

                                # Greedy repair if needed
                                while current_union != full_mask:
                                    best_idx = -1
                                    best_gain = 0
                                    for i in range(total):
                                        if i in current_code:
                                            continue
                                        gain = bin(ball_masks[i] & ~current_union).count('1')
                                        if gain > best_gain:
                                            best_gain = gain
                                            best_idx = i
                                    if best_idx == -1:
                                        break
                                    current_code.append(best_idx)
                                    current_union |= ball_masks[best_idx]

                                # Cleanup
                                current_code, current_union, _ = try_delete_redundant(current_code, current_union)

    return [idx_to_word(i) for i in best_code]
