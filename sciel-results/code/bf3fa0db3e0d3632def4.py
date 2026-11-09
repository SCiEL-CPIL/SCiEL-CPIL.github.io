import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    N = 256

    # Precompute conflict sets using NumPy for speed
    conflict_mask = [0] * (N + 1)
    nums = np.arange(1, N + 1)
    for i in range(1, N + 1):
        js = nums[nums > i]
        # (i*j) % (i+j) == 0
        conflicts = js[(i * js) % (i + js) == 0]
        for j in conflicts:
            conflict_mask[i] |= (1 << j)
            conflict_mask[j] |= (1 << i)

    # Precompute degrees
    degrees = [bin(conflict_mask[i]).count('1') for i in range(N + 1)]

    def is_valid(S):
        """Check if set S is valid (all distinct pairs compatible)."""
        S_list = list(S)
        for idx, a in enumerate(S_list):
            for b in S_list[idx + 1:]:
                if (conflict_mask[a] >> b) & 1:
                    return False
        return True

    def mask_to_list(mask):
        """Convert bitmask to list of elements."""
        result = []
        temp = mask
        while temp:
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            result.append(v)
            temp ^= lb
        return result

    def greedy_min_degree(remaining_mask, selected_mask, rng):
        """
        Greedy minimum-degree independent set using bitmasks.
        """
        remaining = remaining_mask
        selected = selected_mask

        while remaining:
            min_deg = float('inf')
            best_v = -1
            temp = remaining
            while temp:
                lb = temp & (-temp)
                v = lb.bit_length() - 1
                deg = bin(conflict_mask[v] & remaining).count('1')
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                elif deg == min_deg and rng.random() < 0.5:
                    best_v = v
                temp ^= lb

            if best_v == -1:
                break

            selected |= (1 << best_v)
            to_remove = (1 << best_v) | conflict_mask[best_v]
            remaining &= ~to_remove

        return selected

    def local_search_swap(selected_mask, rng, max_iters=50):
        """
        Local search: try to swap out an element for another that increases size.
        Also accepts neutral moves to explore the space.
        """
        best_mask = selected_mask
        best_size = bin(selected_mask).count('1')
        current_mask = selected_mask

        for iteration in range(max_iters):
            if time.time() > deadline:
                break

            selected_list = mask_to_list(current_mask)
            if not selected_list:
                break

            # Pick a random element to remove
            remove_elem = rng.choice(selected_list)
            current_mask &= ~(1 << remove_elem)

            # Try to add a random element not in current_mask
            candidates = [x for x in range(1, N + 1) if not (current_mask >> x) & 1]
            if not candidates:
                current_mask |= (1 << remove_elem)
                continue

            rng.shuffle(candidates)
            added = False
            for add_elem in candidates[:50]:
                compatible = True
                temp = current_mask
                while temp:
                    lb = temp & (-temp)
                    v = lb.bit_length() - 1
                    if (conflict_mask[add_elem] >> v) & 1:
                        compatible = False
                        break
                    temp ^= lb
                if compatible:
                    current_mask |= (1 << add_elem)
                    added = True
                    cur_size = bin(current_mask).count('1')
                    if cur_size > best_size:
                        best_size = cur_size
                        best_mask = current_mask
                    # Accept neutral moves (same size) with some probability to explore
                    elif cur_size == best_size and rng.random() < 0.3:
                        best_mask = current_mask
                    break

            if not added:
                current_mask |= (1 << remove_elem)

        return best_mask

    def two_opt_improve(selected_mask, rng, max_iters=30):
        """
        Try to improve by removing 2 elements and adding up to 3 (net +1).
        Accept neutral moves to explore.
        """
        best_mask = selected_mask
        best_size = bin(selected_mask).count('1')
        current_mask = selected_mask

        for iteration in range(max_iters):
            if time.time() > deadline:
                break

            selected_list = mask_to_list(current_mask)
            if len(selected_list) < 2:
                break

            pair = rng.sample(selected_list, 2)
            current_mask &= ~(1 << pair[0])
            current_mask &= ~(1 << pair[1])

            candidates = [x for x in range(1, N + 1) if not (current_mask >> x) & 1]
            rng.shuffle(candidates)

            added_count = 0
            for add_elem in candidates[:80]:
                if added_count >= 3:
                    break
                compatible = True
                temp = current_mask
                while temp:
                    lb = temp & (-temp)
                    v = lb.bit_length() - 1
                    if (conflict_mask[add_elem] >> v) & 1:
                        compatible = False
                        break
                    temp ^= lb
                if compatible:
                    current_mask |= (1 << add_elem)
                    added_count += 1

            cur_size = bin(current_mask).count('1')
            if cur_size > best_size:
                best_size = cur_size
                best_mask = current_mask
            elif cur_size == best_size and rng.random() < 0.2:
                best_mask = current_mask
            else:
                current_mask = best_mask

        return best_mask

    def final_polish(selected_mask, rng):
        """
        Systematic single-swap polish: try every possible swap (remove one, add one).
        """
        best_mask = selected_mask
        best_size = bin(selected_mask).count('1')

        if time.time() > deadline - 2:
            return best_mask

        selected_list = mask_to_list(best_mask)
        all_elements = list(range(1, N + 1))

        for rem_elem in selected_list:
            if time.time() > deadline - 1:
                break
            temp_mask = best_mask & ~(1 << rem_elem)
            for add_elem in all_elements:
                if time.time() > deadline - 1:
                    break
                if (temp_mask >> add_elem) & 1:
                    continue
                # Check compatibility
                compatible = True
                temp = temp_mask
                while temp:
                    lb = temp & (-temp)
                    v = lb.bit_length() - 1
                    if (conflict_mask[add_elem] >> v) & 1:
                        compatible = False
                        break
                    temp ^= lb
                if compatible:
                    new_mask = temp_mask | (1 << add_elem)
                    new_size = bin(new_mask).count('1')
                    if new_size > best_size:
                        best_size = new_size
                        best_mask = new_mask
                        selected_list = mask_to_list(best_mask)

        return best_mask

    # Main search
    best_mask = 0
    best_size = 0
    all_mask = 0
    for i in range(1, N + 1):
        all_mask |= (1 << i)

    # Strategy 1: Pure min-degree greedy from scratch
    for trial in range(200):
        if time.time() > deadline - 15:
            break

        rng = random.Random(trial * 104729 + 17)
        S_mask = greedy_min_degree(all_mask, 0, rng)
        size = bin(S_mask).count('1')

        if size > best_size:
            best_size = size
            best_mask = S_mask

        if time.time() < deadline - 10:
            S_ls = local_search_swap(S_mask, rng, max_iters=30)
            ls_size = bin(S_ls).count('1')
            if ls_size > best_size:
                best_size = ls_size
                best_mask = S_ls

    # Strategy 2: Start with odds, then add evens via min-degree on remaining
    odds_mask = 0
    evens_mask = 0
    for x in range(1, N + 1):
        if x % 2 == 1:
            odds_mask |= (1 << x)
        else:
            evens_mask |= (1 << x)

    for trial in range(100):
        if time.time() > deadline - 15:
            break

        rng = random.Random(trial * 31337 + 41)
        S_mask = odds_mask

        compatible_evens = 0
        temp = evens_mask
        while temp:
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            if not (conflict_mask[v] & odds_mask):
                compatible_evens |= lb
            temp ^= lb

        if compatible_evens:
            remaining = compatible_evens
            while remaining:
                min_deg = float('inf')
                best_v = -1
                temp = remaining
                while temp:
                    lb = temp & (-temp)
                    v = lb.bit_length() - 1
                    deg = bin(conflict_mask[v] & remaining).count('1')
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v
                    elif deg == min_deg and rng.random() < 0.5:
                        best_v = v
                    temp ^= lb

                if best_v == -1:
                    break

                S_mask |= (1 << best_v)
                to_remove = (1 << best_v) | conflict_mask[best_v]
                remaining &= ~to_remove

        size = bin(S_mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = S_mask

        if time.time() < deadline - 10:
            S_ls = local_search_swap(S_mask, rng, max_iters=30)
            ls_size = bin(S_ls).count('1')
            if ls_size > best_size:
                best_size = ls_size
                best_mask = S_ls

    # Strategy 3: Randomized min-degree with different tie-breaking
    for trial in range(100):
        if time.time() > deadline - 15:
            break

        rng = random.Random(trial * 6151 + 7)
        S_mask = greedy_min_degree(all_mask, 0, rng)
        size = bin(S_mask).count('1')

        if size > best_size:
            best_size = size
            best_mask = S_mask

        if time.time() < deadline - 10:
            S_ls = local_search_swap(S_mask, rng, max_iters=30)
            ls_size = bin(S_ls).count('1')
            if ls_size > best_size:
                best_size = ls_size
                best_mask = S_ls

    # Strategy 4: Two-opt improvement on best solution
    if time.time() < deadline - 5:
        improved = two_opt_improve(best_mask, random.Random(42), max_iters=40)
        imp_size = bin(improved).count('1')
        if imp_size > best_size:
            best_size = imp_size
            best_mask = improved

    # Strategy 5: Final systematic polish
    if time.time() < deadline - 3:
        polished = final_polish(best_mask, random.Random(123))
        pol_size = bin(polished).count('1')
        if pol_size > best_size:
            best_size = pol_size
            best_mask = polished

    # Final verification and fallback
    result = mask_to_list(best_mask)

    if not is_valid(set(result)):
        result = [x for x in range(1, N + 1) if x % 2 == 1]

    return result
