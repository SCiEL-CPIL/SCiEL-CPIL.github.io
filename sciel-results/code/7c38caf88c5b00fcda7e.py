import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 87.0  # Safety margin for 90s limit

    n = 256

    # Precompute 3-APs and 4-APs with bitmasks
    # Element i (1-indexed) corresponds to bit (i-1)

    aps3 = []
    aps4 = []

    elem_to_aps3 = [[] for _ in range(n + 1)]
    elem_to_aps4 = [[] for _ in range(n + 1)]

    # Generate 3-APs
    ap_id = 0
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            if c <= n:
                mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
                aps3.append((a, b, c, mask))
                elem_to_aps3[a].append(ap_id)
                elem_to_aps3[b].append(ap_id)
                elem_to_aps3[c].append(ap_id)
                ap_id += 1

    # Generate 4-APs
    ap_id = 0
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= n:
                mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
                aps4.append((a, b, c, e, mask))
                elem_to_aps4[a].append(ap_id)
                elem_to_aps4[b].append(ap_id)
                elem_to_aps4[c].append(ap_id)
                elem_to_aps4[e].append(ap_id)
                ap_id += 1

    num_aps3 = len(aps3)
    num_aps4 = len(aps4)

    # Precompute masks as list for fast access
    aps3_masks = [x[3] for x in aps3]
    aps4_masks = [x[4] for x in aps4]

    def count_3aps(A_mask):
        count = 0
        for mask in aps3_masks:
            if (A_mask & mask) == mask:
                count += 1
        return count

    def count_4aps(A_mask):
        count = 0
        for mask in aps4_masks:
            if (A_mask & mask) == mask:
                count += 1
        return count

    def is_valid_add(A_mask, x):
        bit_x = 1 << (x - 1)
        for ap_idx in elem_to_aps4[x]:
            mask = aps4_masks[ap_idx]
            others_mask = mask ^ bit_x
            if (A_mask & others_mask) == others_mask:
                return False
        return True

    def run_greedy(seed):
        rng = random.Random(seed)
        A_mask = 0
        A_set = set()
        candidates = list(range(1, n + 1))
        rng.shuffle(candidates)

        for x in candidates:
            if is_valid_add(A_mask, x):
                A_mask |= (1 << (x - 1))
                A_set.add(x)

        return A_mask, A_set

    def local_search(A_mask, A_set, time_limit):
        # Fast local search using bitmask operations
        # We track the current 3-AP count incrementally

        current_score = count_3aps(A_mask)
        in_set = [False] * (n + 1)
        for x in A_set:
            in_set[x] = True

        no_improvement = 0
        max_no_improvement = 300  # Allow some stagnation for diversity

        while time.time() < time_limit:
            best_delta = 0
            best_move = None
            best_u = None
            best_v = None
            best_type = None

            # Check Add moves
            for v in range(1, n + 1):
                if in_set[v]:
                    continue
                if not is_valid_add(A_mask, v):
                    continue

                bit_v = 1 << (v - 1)
                gained = 0
                for ap_idx in elem_to_aps3[v]:
                    mask = aps3_masks[ap_idx]
                    others_mask = mask ^ bit_v
                    if (A_mask & others_mask) == others_mask:
                        gained += 1

                if gained > best_delta:
                    best_delta = gained
                    best_move = 'add'
                    best_v = v
                    best_u = None
                    best_type = 'add'

            # Check Swap moves - sample a subset if set is large
            if len(A_set) > 60:
                # Sample swaps for speed
                rng_local = random.Random()
                sampled_u = rng_local.sample(list(A_set), min(40, len(A_set)))
            else:
                sampled_u = list(A_set)

            for u in sampled_u:
                bit_u = 1 << (u - 1)
                for v in range(1, n + 1):
                    if v == u or in_set[v]:
                        continue

                    # Check 4-AP validity for swap
                    valid = True
                    bit_v = 1 << (v - 1)
                    for ap_idx in elem_to_aps4[v]:
                        mask = aps4_masks[ap_idx]
                        others_mask = mask ^ bit_v
                        if (A_mask & others_mask) == others_mask:
                            if (bit_u & others_mask) == 0:
                                valid = False
                                break

                    if not valid:
                        continue

                    # Calculate 3-AP delta
                    lost = 0
                    for ap_idx in elem_to_aps3[u]:
                        mask = aps3_masks[ap_idx]
                        others_mask = mask ^ bit_u
                        if (A_mask & others_mask) == others_mask:
                            lost += 1

                    gained = 0
                    for ap_idx in elem_to_aps3[v]:
                        mask = aps3_masks[ap_idx]
                        others_mask = mask ^ bit_v
                        if (A_mask & others_mask) == others_mask:
                            if (bit_u & others_mask) == 0:
                                gained += 1

                    delta = gained - lost

                    if delta > best_delta:
                        best_delta = delta
                        best_move = 'swap'
                        best_u = u
                        best_v = v
                        best_type = 'swap'

            if best_move is None:
                no_improvement += 1
                if no_improvement > max_no_improvement:
                    break
                # Try a random perturbation to escape local optimum
                if no_improvement % 50 == 0:
                    # Random swap
                    if len(A_set) > 2:
                        u = random.choice(list(A_set))
                        # Find a valid v to swap with
                        for _ in range(20):
                            v = random.randint(1, n)
                            if v not in A_set and v != u:
                                bit_u = 1 << (u - 1)
                                bit_v = 1 << (v - 1)
                                valid = True
                                for ap_idx in elem_to_aps4[v]:
                                    mask = aps4_masks[ap_idx]
                                    others_mask = mask ^ bit_v
                                    if (A_mask & others_mask) == others_mask:
                                        if (bit_u & others_mask) == 0:
                                            valid = False
                                            break
                                if valid:
                                    A_mask &= ~(1 << (u - 1))
                                    A_set.remove(u)
                                    in_set[u] = False
                                    A_mask |= (1 << (v - 1))
                                    A_set.add(v)
                                    in_set[v] = True
                                    current_score = count_3aps(A_mask)
                                    break
            else:
                # Apply best move
                if best_type == 'add':
                    A_mask |= (1 << (best_v - 1))
                    A_set.add(best_v)
                    in_set[best_v] = True
                else:
                    A_mask &= ~(1 << (best_u - 1))
                    A_set.remove(best_u)
                    in_set[best_u] = False
                    A_mask |= (1 << (best_v - 1))
                    A_set.add(best_v)
                    in_set[best_v] = True

                current_score += best_delta
                no_improvement = 0

        return A_mask, A_set, current_score

    # Main logic
    best_mask = 0
    best_set = set()
    best_score = 0

    # Try several random initializations with different strategies
    num_initializations = 15
    elapsed = time.time() - start_time
    remaining = deadline - start_time - 3.0  # Reserve 3s for final verification
    time_per_init = max(remaining / num_initializations, 1.0)

    for i in range(num_initializations):
        if time.time() > deadline - 3:
            break

        # Vary the greedy strategy
        if i < 5:
            A_mask, A_set = run_greedy(i)
        elif i < 10:
            # Reverse greedy
            rng = random.Random(100 + i)
            A_mask = 0
            A_set = set()
            candidates = list(range(1, n + 1))
            rng.shuffle(candidates)
            candidates.reverse()
            for x in candidates:
                if is_valid_add(A_mask, x):
                    A_mask |= (1 << (x - 1))
                    A_set.add(x)
        else:
            # Weighted greedy: prefer elements with more 3-AP connections
            rng = random.Random(200 + i)
            A_mask = 0
            A_set = set()
            candidates = list(range(1, n + 1))
            # Sort by number of 3-APs they participate in (descending) with some randomness
            weights = [(len(elem_to_aps3[x]), x) for x in candidates]
            rng.shuffle(weights)
            weights.sort(key=lambda w: -w[0])
            candidates = [w[1] for w in weights]
            for x in candidates:
                if is_valid_add(A_mask, x):
                    A_mask |= (1 << (x - 1))
                    A_set.add(x)

        # Refine with local search
        ls_time_limit = min(time.time() + time_per_init * 0.85, deadline - 2)
        A_mask, A_set, score = local_search(A_mask, A_set, ls_time_limit)

        if score > best_score:
            best_score = score
            best_mask = A_mask
            best_set = A_set

    # Final verification
    if count_4aps(best_mask) != 0:
        # Fallback: simple greedy
        A_mask, A_set = run_greedy(0)
        best_mask = A_mask
        best_set = A_set
        best_score = count_3aps(best_mask)

    # If we have time, do one more round of local search on the best
    if time.time() < deadline - 5:
        A_mask, A_set, score = local_search(best_mask, best_set, deadline - 1)
        if score > best_score:
            best_score = score
            best_mask = A_mask
            best_set = A_set

    return sorted(best_set)
