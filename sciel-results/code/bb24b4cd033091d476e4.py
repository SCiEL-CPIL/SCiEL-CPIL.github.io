import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    n = 12
    m = 40
    full_mask = (1 << n) - 1
    num_masks = 1 << n  # 4096

    # Precompute popcount
    popcount = [0] * num_masks
    for i in range(1, num_masks):
        popcount[i] = popcount[i >> 1] + (i & 1)

    # Group masks by size
    masks_by_size = [[] for _ in range(n + 1)]
    for mask in range(num_masks):
        masks_by_size[popcount[mask]].append(mask)

    # Precompute comparability matrix as a list of bytearrays for speed
    # comp[a][b] = 1 if a is strict subset of b or vice versa
    comp = [bytearray(num_masks) for _ in range(num_masks)]
    for a in range(num_masks):
        row = comp[a]
        for b in range(num_masks):
            if a != b and ((a & b) == a or (a & b) == b):
                row[b] = 1

    def calc_score(masks):
        s = 0
        for i in range(len(masks)):
            a = masks[i]
            row_a = comp[a]
            for j in range(i + 1, len(masks)):
                if row_a[masks[j]]:
                    s += 1
        return s

    def get_comparability_with_others(mask, masks_list, exclude_idx):
        row = comp[mask]
        count = 0
        for j in range(len(masks_list)):
            if j != exclude_idx and row[masks_list[j]]:
                count += 1
        return count

    # Greedy initialization: start with empty set and full set, then add masks
    # that maximize comparability with the current set
    current_masks = [0, full_mask]
    used_masks = {0, full_mask}

    for _ in range(m - 2):
        best_mask = -1
        best_score = -1
        # Try all masks, pick the one with max comparability to current set
        for mask in range(num_masks):
            if mask in used_masks:
                continue
            row = comp[mask]
            score = 0
            for cm in current_masks:
                if row[cm]:
                    score += 1
            if score > best_score:
                best_score = score
                best_mask = mask

        if best_mask != -1:
            current_masks.append(best_mask)
            used_masks.add(best_mask)
        else:
            break

    best_score = calc_score(current_masks)
    best_solution = list(current_masks)

    random.seed(42)

    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.99995
    no_improve_count = 0
    max_no_improve = 3000
    restarts = 0
    max_restarts = 30

    while time.time() < deadline and restarts < max_restarts:
        if time.time() > deadline - 3.0:
            break

        current_masks = list(best_solution)
        used_masks = set(current_masks)
        current_score = best_score

        T = 5.0
        no_improve_count = 0

        while time.time() < deadline - 3.0 and T > T_min:
            # Pick a random index to modify
            idx = random.randint(0, m - 1)
            old_mask = current_masks[idx]

            # Generate candidate with bias towards small perturbations
            r = random.random()
            if r < 0.5:
                # Flip one bit (stay close)
                if old_mask == 0:
                    bit = random.randint(0, n - 1)
                    new_mask = 1 << bit
                elif old_mask == full_mask:
                    bit = random.randint(0, n - 1)
                    new_mask = full_mask ^ (1 << bit)
                else:
                    # Try to add or remove a bit
                    if random.random() < 0.5:
                        # Remove a set bit
                        bits = [i for i in range(n) if (old_mask >> i) & 1]
                        if bits:
                            bit = random.choice(bits)
                            new_mask = old_mask ^ (1 << bit)
                        else:
                            new_mask = old_mask
                    else:
                        # Add an unset bit
                        bits = [i for i in range(n) if not ((old_mask >> i) & 1)]
                        if bits:
                            bit = random.choice(bits)
                            new_mask = old_mask | (1 << bit)
                        else:
                            new_mask = old_mask
            elif r < 0.8:
                # Flip two bits
                if old_mask == 0:
                    b1 = random.randint(0, n - 1)
                    b2 = random.randint(0, n - 1)
                    while b2 == b1:
                        b2 = random.randint(0, n - 1)
                    new_mask = (1 << b1) | (1 << b2)
                elif old_mask == full_mask:
                    b1 = random.randint(0, n - 1)
                    b2 = random.randint(0, n - 1)
                    while b2 == b1:
                        b2 = random.randint(0, n - 1)
                    new_mask = full_mask ^ (1 << b1) ^ (1 << b2)
                else:
                    all_bits = list(range(n))
                    random.shuffle(all_bits)
                    b1 = all_bits[0]
                    b2 = all_bits[1]
                    new_mask = old_mask ^ (1 << b1) ^ (1 << b2)
            else:
                # Random mask (global move)
                new_mask = random.randint(0, full_mask)

            if new_mask == old_mask or new_mask in used_masks:
                continue

            # Calculate delta using precomputed matrix
            old_row = comp[old_mask]
            new_row = comp[new_mask]
            delta = 0
            for j in range(m):
                if j == idx:
                    continue
                cm = current_masks[j]
                delta += new_row[cm] - old_row[cm]

            # Acceptance
            if delta >= 0 or random.random() < math.exp(delta / T):
                used_masks.remove(old_mask)
                used_masks.add(new_mask)
                current_masks[idx] = new_mask
                current_score += delta
                if delta > 0:
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if current_score > best_score:
                    best_score = current_score
                    best_solution = list(current_masks)
            else:
                no_improve_count += 1

            T *= cooling_rate

            if no_improve_count > max_no_improve:
                break

        restarts += 1

    # Final polish: Hill Climbing with multiple passes
    improved = True
    polish_pass = 0
    max_polish_passes = 20
    while improved and time.time() < deadline - 1.0 and polish_pass < max_polish_passes:
        improved = False
        polish_pass += 1
        for i in range(m):
            if time.time() > deadline - 1.0:
                break
            old_mask = current_masks[i]
            old_comp_count = get_comparability_with_others(old_mask, current_masks, i)

            # Try to find a better replacement by scanning all masks
            best_new_mask = -1
            best_delta = 0
            for new_mask in range(num_masks):
                if new_mask == old_mask or new_mask in used_masks:
                    continue
                new_comp_count = get_comparability_with_others(new_mask, current_masks, i)
                delta = new_comp_count - old_comp_count
                if delta > best_delta:
                    best_delta = delta
                    best_new_mask = new_mask

            if best_new_mask != -1 and best_delta > 0:
                used_masks.remove(old_mask)
                used_masks.add(best_new_mask)
                current_masks[i] = best_new_mask
                best_score += best_delta
                best_solution = list(current_masks)
                improved = True

    # Convert to list of lists
    result = []
    for mask in best_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)

    return result
