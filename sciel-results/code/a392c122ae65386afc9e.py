import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0

    N = 256
    all_nums = list(range(1, N + 1))

    # Precompute conflict_mask: conflict_mask[i] is a bitmask of indices j where i and j conflict
    conflict_mask = [0] * N

    for i in range(N):
        a = i + 1
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflict_mask[i] |= (1 << j)

    # Precompute popcount table for 16-bit chunks
    popcount = [0] * (1 << 16)
    for i in range(1, 1 << 16):
        popcount[i] = popcount[i >> 1] + (i & 1)

    def fast_popcount(x):
        count = 0
        while x:
            count += popcount[x & 0xFFFF]
            x >>= 16
        return count

    def mask_to_list(mask):
        result = []
        m = mask
        while m:
            lsb = m & (-m)
            result.append(lsb.bit_length() - 1)
            m ^= lsb
        return result

    odds_indices = [i for i in range(N) if (i + 1) % 2 == 1]
    evens_indices = [i for i in range(N) if (i + 1) % 2 == 0]

    # Start with all odds
    current_mask = 0
    for i in odds_indices:
        current_mask |= (1 << i)

    best_mask = current_mask
    best_size = fast_popcount(current_mask)

    # Phase 1: Add evens with 0 conflicts
    changed = True
    while changed and time.time() < DEADLINE - 55:
        changed = False
        for e in evens_indices:
            if time.time() > DEADLINE - 55:
                break
            if (current_mask >> e) & 1:
                continue
            if (conflict_mask[e] & current_mask) == 0:
                current_mask |= (1 << e)
                best_mask = current_mask
                best_size = fast_popcount(current_mask)
                changed = True

    # Phase 1b: Try adding evens that cause minimal conflicts (net gain)
    changed = True
    while changed and time.time() < DEADLINE - 50:
        changed = False
        for e in evens_indices:
            if time.time() > DEADLINE - 50:
                break
            if (current_mask >> e) & 1:
                continue
            conflicts = conflict_mask[e] & current_mask
            num_conflicts = fast_popcount(conflicts)
            if num_conflicts <= 1:
                new_mask = (current_mask & ~conflicts) | (1 << e)
                new_size = best_size - num_conflicts + 1
                if new_size > best_size:
                    current_mask = new_mask
                    best_mask = current_mask
                    best_size = new_size
                    changed = True

    # Phase 1c: 1-for-2 exchange pass (from inspiration 1)
    # For each u in S, find pairs (v1, v2) not in S where each only conflicts with u
    # and v1, v2 don't conflict with each other
    def try_1_for_2(mask, size):
        """Try 1-for-2 exchanges. Returns (new_mask, new_size) if improved."""
        S_list = mask_to_list(mask)
        random.shuffle(S_list)
        for u in S_list:
            if time.time() > DEADLINE - 45:
                break
            # Find all v not in mask where u is the ONLY conflict
            cands = []
            for v in range(N):
                if (mask >> v) & 1:
                    continue
                if (conflict_mask[v] & mask) == (1 << u):
                    cands.append(v)
            if len(cands) >= 2:
                for i in range(len(cands)):
                    for j in range(i + 1, len(cands)):
                        v1, v2 = cands[i], cands[j]
                        if not ((conflict_mask[v1] >> v2) & 1):
                            # Valid 1-for-2: remove u, add v1 and v2
                            new_mask = (mask & ~(1 << u)) | (1 << v1) | (1 << v2)
                            new_size = size + 1
                            return new_mask, new_size
        return mask, size

    for _ in range(3):
        if time.time() > DEADLINE - 45:
            break
        new_mask, new_size = try_1_for_2(current_mask, best_size)
        if new_size > best_size:
            current_mask = new_mask
            best_mask = new_mask
            best_size = new_size

    # Phase 2: Simulated Annealing with improved moves including 1-for-2
    T = 2.0
    cooling = 0.9998
    max_iters = 300000

    current_size = best_size
    in_set = mask_to_list(current_mask)
    in_set_set = set(in_set)
    evens_not_in = [i for i in evens_indices if i not in in_set_set]
    odds_not_in = [i for i in odds_indices if i not in in_set_set]

    for i_iter in range(max_iters):
        if time.time() > DEADLINE - 3:
            break

        r = random.random()
        if r < 0.40:
            # Single add move
            if random.random() < 0.7 and evens_not_in:
                e = random.choice(evens_not_in)
            elif odds_not_in:
                e = random.choice(odds_not_in)
            elif evens_not_in:
                e = random.choice(evens_not_in)
            else:
                continue

            conflicts = conflict_mask[e] & current_mask
            num_conflicts = fast_popcount(conflicts)
            new_size = current_size - num_conflicts + 1

            if new_size > current_size:
                current_mask = (current_mask & ~conflicts) | (1 << e)
                current_size = new_size
                in_set.append(e)
                in_set_set.add(e)
                if e in evens_not_in:
                    evens_not_in.remove(e)
                elif e in odds_not_in:
                    odds_not_in.remove(e)
                for c in mask_to_list(conflicts):
                    in_set.remove(c)
                    in_set_set.discard(c)
                    if c % 2 == 0:
                        evens_not_in.append(c)
                    else:
                        odds_not_in.append(c)
                if current_size > best_size:
                    best_mask = current_mask
                    best_size = current_size
            else:
                delta = new_size - current_size
                if delta == 0:
                    if random.random() < 0.4:
                        current_mask = (current_mask & ~conflicts) | (1 << e)
                        in_set.append(e)
                        in_set_set.add(e)
                        if e in evens_not_in:
                            evens_not_in.remove(e)
                        elif e in odds_not_in:
                            odds_not_in.remove(e)
                        for c in mask_to_list(conflicts):
                            in_set.remove(c)
                            in_set_set.discard(c)
                            if c % 2 == 0:
                                evens_not_in.append(c)
                            else:
                                odds_not_in.append(c)
                else:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_mask = (current_mask & ~conflicts) | (1 << e)
                        current_size = new_size
                        in_set.append(e)
                        in_set_set.add(e)
                        if e in evens_not_in:
                            evens_not_in.remove(e)
                        elif e in odds_not_in:
                            odds_not_in.remove(e)
                        for c in mask_to_list(conflicts):
                            in_set.remove(c)
                            in_set_set.discard(c)
                            if c % 2 == 0:
                                evens_not_in.append(c)
                            else:
                                odds_not_in.append(c)
        elif r < 0.55:
            # 1-for-2 exchange move (key improvement from inspiration 1)
            if not in_set:
                continue
            u = random.choice(in_set)
            # Find candidates where u is the only conflict
            cands = []
            for v in range(N):
                if (current_mask >> v) & 1:
                    continue
                if (conflict_mask[v] & current_mask) == (1 << u):
                    cands.append(v)
            if len(cands) >= 2:
                # Try to find a non-conflicting pair
                random.shuffle(cands)
                found = False
                for i in range(min(len(cands), 20)):
                    for j in range(i + 1, min(len(cands), 20)):
                        v1, v2 = cands[i], cands[j]
                        if not ((conflict_mask[v1] >> v2) & 1):
                            # Execute 1-for-2
                            current_mask = (current_mask & ~(1 << u)) | (1 << v1) | (1 << v2)
                            current_size += 1
                            in_set.remove(u)
                            in_set_set.discard(u)
                            if u % 2 == 0:
                                evens_not_in.append(u)
                            else:
                                odds_not_in.append(u)
                            in_set.append(v1)
                            in_set.append(v2)
                            in_set_set.add(v1)
                            in_set_set.add(v2)
                            if v1 in evens_not_in:
                                evens_not_in.remove(v1)
                            elif v1 in odds_not_in:
                                odds_not_in.remove(v1)
                            if v2 in evens_not_in:
                                evens_not_in.remove(v2)
                            elif v2 in odds_not_in:
                                odds_not_in.remove(v2)
                            if current_size > best_size:
                                best_mask = current_mask
                                best_size = current_size
                            found = True
                            break
                    if found:
                        break
        elif r < 0.70:
            # Swap move: remove 1, add 1
            if not in_set:
                continue
            rm_idx = random.choice(in_set)
            temp_mask = current_mask & ~(1 << rm_idx)

            candidates = [idx for idx in range(N) if not ((temp_mask >> idx) & 1)]
            if not candidates:
                continue
            e = random.choice(candidates)

            conflicts_with_temp = conflict_mask[e] & temp_mask
            if conflicts_with_temp == 0:
                if random.random() < 0.3:
                    current_mask = temp_mask | (1 << e)
                    in_set.remove(rm_idx)
                    in_set_set.discard(rm_idx)
                    in_set.append(e)
                    in_set_set.add(e)
                    if rm_idx % 2 == 0:
                        evens_not_in.append(rm_idx)
                    else:
                        odds_not_in.append(rm_idx)
                    if e in evens_not_in:
                        evens_not_in.remove(e)
                    elif e in odds_not_in:
                        odds_not_in.remove(e)
        elif r < 0.85:
            # Multi-element exchange: remove k, add 1
            k = random.choice([2, 3])
            if len(in_set) < k:
                continue
            removed = random.sample(in_set, k)

            temp_mask = current_mask
            for rm in removed:
                temp_mask &= ~(1 << rm)

            candidates = [idx for idx in range(N) if not ((temp_mask >> idx) & 1)]
            if not candidates:
                continue

            zero_conflict = [idx for idx in candidates if (conflict_mask[idx] & temp_mask) == 0]
            if zero_conflict:
                e = random.choice(zero_conflict)
            else:
                e = random.choice(candidates)

            conflicts_with_temp = conflict_mask[e] & temp_mask
            if conflicts_with_temp != 0:
                num_conf = fast_popcount(conflicts_with_temp)
                new_size = current_size - k - num_conf + 1
                if new_size <= current_size:
                    continue

            new_size = current_size - k + 1

            if new_size > current_size:
                current_mask = temp_mask | (1 << e)
                current_size = new_size
                for rm in removed:
                    in_set.remove(rm)
                    in_set_set.discard(rm)
                    if rm % 2 == 0:
                        evens_not_in.append(rm)
                    else:
                        odds_not_in.append(rm)
                in_set.append(e)
                in_set_set.add(e)
                if e in evens_not_in:
                    evens_not_in.remove(e)
                elif e in odds_not_in:
                    odds_not_in.remove(e)
                if current_size > best_size:
                    best_mask = current_mask
                    best_size = current_size
            else:
                delta = new_size - current_size
                if delta == 0:
                    if random.random() < 0.3:
                        current_mask = temp_mask | (1 << e)
                        for rm in removed:
                            in_set.remove(rm)
                            in_set_set.discard(rm)
                            if rm % 2 == 0:
                                evens_not_in.append(rm)
                            else:
                                odds_not_in.append(rm)
                        in_set.append(e)
                        in_set_set.add(e)
                        if e in evens_not_in:
                            evens_not_in.remove(e)
                        elif e in odds_not_in:
                            odds_not_in.remove(e)
                else:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_mask = temp_mask | (1 << e)
                        current_size = new_size
                        for rm in removed:
                            in_set.remove(rm)
                            in_set_set.discard(rm)
                            if rm % 2 == 0:
                                evens_not_in.append(rm)
                            else:
                                odds_not_in.append(rm)
                        in_set.append(e)
                        in_set_set.add(e)
                        if e in evens_not_in:
                            evens_not_in.remove(e)
                        elif e in odds_not_in:
                            odds_not_in.remove(e)
        else:
            # Random removal
            if in_set:
                r_idx = random.choice(in_set)
                current_mask &= ~(1 << r_idx)
                current_size -= 1
                in_set.remove(r_idx)
                in_set_set.discard(r_idx)
                if r_idx % 2 == 0:
                    evens_not_in.append(r_idx)
                else:
                    odds_not_in.append(r_idx)

        T *= cooling
        if T < 0.01:
            T = 0.01

        # Restart mechanism
        if i_iter % 30000 == 0 and i_iter > 0:
            if current_size < best_size - 2:
                current_mask = best_mask
                current_size = best_size
                in_set = mask_to_list(current_mask)
                in_set_set = set(in_set)
                evens_not_in = [i for i in evens_indices if i not in in_set_set]
                odds_not_in = [i for i in odds_indices if i not in in_set_set]
                T = 1.0

    # Final polish: greedy add any element with 0 conflicts
    changed = True
    while changed and time.time() < DEADLINE - 1:
        changed = False
        for idx in range(N):
            if time.time() > DEADLINE - 1:
                break
            if (best_mask >> idx) & 1:
                continue
            if (conflict_mask[idx] & best_mask) == 0:
                best_mask |= (1 << idx)
                best_size += 1
                changed = True

    # Second polish: 1-for-2 on best
    if time.time() < DEADLINE - 0.5:
        new_mask, new_size = try_1_for_2(best_mask, best_size)
        if new_size > best_size:
            best_mask = new_mask
            best_size = new_size

    # Third polish: try adding elements that cause at most 1 conflict if net gain
    changed = True
    while changed and time.time() < DEADLINE - 0.2:
        changed = False
        for idx in range(N):
            if time.time() > DEADLINE - 0.2:
                break
            if (best_mask >> idx) & 1:
                continue
            conflicts = conflict_mask[idx] & best_mask
            num_conf = fast_popcount(conflicts)
            if num_conf <= 1:
                new_mask = (best_mask & ~conflicts) | (1 << idx)
                new_size = best_size - num_conf + 1
                if new_size > best_size:
                    best_mask = new_mask
                    best_size = new_size
                    changed = True

    result = sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])

    return result
