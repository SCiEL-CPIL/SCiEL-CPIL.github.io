import time
import random

P = {'n': 128}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # explicit safety margin (budget is 90s)

    n = P['n']
    max_val = n

    # ---- Precompute pair sums (for the "x is a sum" case) ----
    pairs_sum = [[] for _ in range(2 * max_val + 1)]
    for a in range(1, max_val + 1):
        for b in range(a + 1, max_val + 1):
            s = a + b
            if s <= 2 * max_val:
                pairs_sum[s].append((a, b))

    # valid_range_mask[x]: bits 0..(max_val-x-1) set
    valid_range_mask = [0] * (max_val + 1)
    for x in range(1, max_val + 1):
        num_bits = max_val - x
        if num_bits > 0:
            valid_range_mask[x] = (1 << num_bits) - 1

    def is_valid_addition(x, mask):
        """Check if adding x (1-indexed) to valid bitset 'mask' keeps it valid."""
        if mask & (1 << (x - 1)):
            return True  # already present
        x_bit = 1 << (x - 1)

        # Case 1: x is a member a. Need b,c in S with b+c, x+b, x+c in S.
        A_bits = mask & (mask >> x) & valid_range_mask[x]  # {b in S | x+b in S}
        m = A_bits
        while m:
            lsb_b = m & -m
            b_idx = lsb_b.bit_length() - 1
            b_val = b_idx + 1
            m ^= lsb_b
            # C_b = {c in S | b+c in S}; need c in A_bits, c != b, with b+c in S
            C_b = mask & (mask >> b_val) & valid_range_mask[b_val]
            common = A_bits & C_b
            if common & ~lsb_b:
                return False

        # Case 2: x is a sum, a+b = x, a,b in S.
        for a, b in pairs_sum[x]:
            if not (mask & (1 << (a - 1))):
                continue
            if not (mask & (1 << (b - 1))):
                continue
            U_a = mask & (mask >> a) & valid_range_mask[a]
            U_b = mask & (mask >> b) & valid_range_mask[b]
            common = U_a & U_b
            a_bit = 1 << (a - 1)
            b_bit = 1 << (b - 1)
            if common & ~(a_bit | b_bit):
                return False

        return True

    def popcount(x):
        count = 0
        while x:
            x &= x - 1
            count += 1
        return count

    # ---- Structural start: split into H (large) + L (sum-free low) ----
    K = 64
    H = list(range(K, max_val + 1))                 # {64..128}
    max_L = (K - 1) // 2                            # 31
    start_L = max_L // 2 + 1                        # 22
    L = list(range(start_L, max_L + 1))             # {22..31}, sum-free, max pair-sum < K

    best_mask = 0
    for v in H + L:
        best_mask |= (1 << (v - 1))
    best_size = popcount(best_mask)

    def elements_of(mask):
        return [i for i in range(1, max_val + 1) if (mask & (1 << (i - 1)))]

    def unused_of(mask):
        return [i for i in range(1, max_val + 1) if not (mask & (1 << (i - 1)))]

    current_mask = best_mask
    current_size = best_size
    current_elements = elements_of(current_mask)
    unused_elements = unused_of(current_mask)

    # ---- Local search: greedy add, 1-out/2-in swaps, perturbation ----
    no_progress_count = 0
    max_no_progress = 50

    while time.time() < deadline - 3.0:
        # Greedy add
        if unused_elements:
            random.shuffle(unused_elements)
            added = False
            for x in unused_elements:
                if is_valid_addition(x, current_mask):
                    current_mask |= (1 << (x - 1))
                    current_size += 1
                    unused_elements.remove(x)
                    current_elements.append(x)
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask
                    added = True
                    no_progress_count = 0
                    break
            if added:
                continue

        # 1-out, 2-in swap
        if current_elements and len(unused_elements) >= 2:
            idx_out = random.randint(0, len(current_elements) - 1)
            x_out = current_elements[idx_out]
            mask_trial = current_mask & ~(1 << (x_out - 1))

            valid_x1 = [x for x in unused_elements if is_valid_addition(x, mask_trial)]
            if len(valid_x1) >= 2:
                random.shuffle(valid_x1)
                found = False
                max_checks = 150
                checks = 0
                for i in range(len(valid_x1)):
                    if checks > max_checks:
                        break
                    x1 = valid_x1[i]
                    mask_1 = mask_trial | (1 << (x1 - 1))
                    for j in range(i + 1, len(valid_x1)):
                        if checks > max_checks:
                            break
                        checks += 1
                        x2 = valid_x1[j]
                        if is_valid_addition(x2, mask_1):
                            current_mask = mask_1 | (1 << (x2 - 1))
                            current_size += 1
                            current_elements.remove(x_out)
                            current_elements.append(x1)
                            current_elements.append(x2)
                            unused_elements.remove(x1)
                            unused_elements.remove(x2)
                            if current_size > best_size:
                                best_size = current_size
                                best_mask = current_mask
                            found = True
                            no_progress_count = 0
                            break
                    if found:
                        break
                if found:
                    continue

        # Perturbation: remove a random element
        if current_elements:
            idx_out = random.randint(0, len(current_elements) - 1)
            x_out = current_elements[idx_out]
            current_mask &= ~(1 << (x_out - 1))
            current_size -= 1
            current_elements.remove(x_out)
            unused_elements.append(x_out)
            no_progress_count += 1

        if no_progress_count > max_no_progress:
            current_mask = best_mask
            current_size = best_size
            current_elements = elements_of(current_mask)
            unused_elements = unused_of(current_mask)
            no_progress_count = 0

    # ---- Final systematic pass on best_mask (only if time remains) ----
    if time.time() < deadline - 1.0:
        final_mask = best_mask
        final_size = best_size
        final_unused = unused_of(final_mask)
        progress = True
        while progress and time.time() < deadline - 1.0:
            progress = False
            for x in list(final_unused):
                if is_valid_addition(x, final_mask):
                    final_mask |= (1 << (x - 1))
                    final_size += 1
                    final_unused.remove(x)
                    if final_size > best_size:
                        best_size = final_size
                        best_mask = final_mask
                    progress = True

        # A few 1-out/2-in attempts on the best
        if time.time() < deadline - 1.0:
            for _ in range(20):
                if time.time() > deadline - 1.0:
                    break
                f_elems = elements_of(best_mask)
                f_unused = unused_of(best_mask)
                if not f_elems or len(f_unused) < 2:
                    break
                idx_out = random.randint(0, len(f_elems) - 1)
                x_out = f_elems[idx_out]
                mask_trial = best_mask & ~(1 << (x_out - 1))
                valid_x1 = [x for x in f_unused if is_valid_addition(x, mask_trial)]
                if len(valid_x1) >= 2:
                    random.shuffle(valid_x1)
                    for i in range(len(valid_x1)):
                        x1 = valid_x1[i]
                        mask_1 = mask_trial | (1 << (x1 - 1))
                        for j in range(i + 1, len(valid_x1)):
                            x2 = valid_x1[j]
                            if is_valid_addition(x2, mask_1):
                                new_mask = mask_1 | (1 << (x2 - 1))
                                if popcount(new_mask) > best_size:
                                    best_size = popcount(new_mask)
                                    best_mask = new_mask
                                break
                        else:
                            continue
                        break

    # ---- Convert best_mask to list of ints ----
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        m ^= lsb

    return sorted(result)
