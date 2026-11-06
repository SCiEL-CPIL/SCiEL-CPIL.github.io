import time
import random
import math

def construct():
    start_time = time.time()
    # Safety margin before the 60s limit
    time_limit = 58.0

    N = 256
    M = N + 1  # 1-indexed

    # Precompute 3-APs and 4-APs
    three_aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))

    four_aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))

    # Data structures for fast lookup
    # For each element x, list of (ap_idx, position) for 3-APs
    elem_to_3aps = [[] for _ in range(M)]
    for i, (a, b, c) in enumerate(three_aps):
        elem_to_3aps[a].append((i, 0))
        elem_to_3aps[b].append((i, 1))
        elem_to_3aps[c].append((i, 2))

    # For each element x, list of (ap_idx, position) for 4-APs
    elem_to_4aps = [[] for _ in range(M)]
    for i, (a, b, c, d) in enumerate(four_aps):
        elem_to_4aps[a].append((i, 0))
        elem_to_4aps[b].append((i, 1))
        elem_to_4aps[c].append((i, 2))
        elem_to_4aps[d].append((i, 3))

    # State
    in_set = [False] * M
    current_score = 0

    # Helper to check validity of adding x
    def try_add(x):
        if in_set[x]:
            return False, 0
        # Check 4-AP constraints
        for ap_idx, pos in elem_to_4aps[x]:
            a, b, c, d = four_aps[ap_idx]
            if pos == 0:
                if in_set[b] and in_set[c] and in_set[d]:
                    return False, 0
            elif pos == 1:
                if in_set[a] and in_set[c] and in_set[d]:
                    return False, 0
            elif pos == 2:
                if in_set[a] and in_set[b] and in_set[d]:
                    return False, 0
            else:
                if in_set[a] and in_set[b] and in_set[c]:
                    return False, 0

        # Calculate score delta
        delta = 0
        for ap_idx, pos in elem_to_3aps[x]:
            a, b, c = three_aps[ap_idx]
            if pos == 0:
                if in_set[b] and in_set[c]:
                    delta += 1
            elif pos == 1:
                if in_set[a] and in_set[c]:
                    delta += 1
            else:
                if in_set[a] and in_set[b]:
                    delta += 1
        return True, delta

    def try_remove(x):
        if not in_set[x]:
            return False, 0
        # Calculate score loss
        delta = 0
        for ap_idx, pos in elem_to_3aps[x]:
            a, b, c = three_aps[ap_idx]
            if pos == 0:
                if in_set[b] and in_set[c]:
                    delta += 1
            elif pos == 1:
                if in_set[a] and in_set[c]:
                    delta += 1
            else:
                if in_set[a] and in_set[b]:
                    delta += 1
        return True, -delta

    def greedy_init():
        nonlocal in_set, current_score
        in_set = [False] * M
        current_score = 0
        # Randomized greedy
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            valid, delta = try_add(x)
            if valid:
                in_set[x] = True
                current_score += delta

    def hill_climb():
        nonlocal in_set, current_score
        improved = True
        while improved and time.time() < time_limit - 2:
            improved = False
            # Try additions
            for x in range(1, N + 1):
                if not in_set[x]:
                    valid, delta = try_add(x)
                    if valid and delta > 0:
                        in_set[x] = True
                        current_score += delta
                        improved = True

            # Try swaps
            in_set_list = [x for x in range(1, N + 1) if in_set[x]]
            not_in_set_list = [x for x in range(1, N + 1) if not in_set[x]]
            for x_out in in_set_list:
                for x_in in not_in_set_list:
                    # Check validity
                    in_set[x_out] = False
                    valid_add, delta_add = try_add(x_in)
                    if valid_add:
                        # Calculate loss
                        loss = 0
                        for ap_idx, pos in elem_to_3aps[x_out]:
                            a, b, c = three_aps[ap_idx]
                            if pos == 0:
                                if in_set[b] and in_set[c]:
                                    loss += 1
                            elif pos == 1:
                                if in_set[a] and in_set[c]:
                                    loss += 1
                            else:
                                if in_set[a] and in_set[b]:
                                    loss += 1
                        net_delta = delta_add - loss
                        if net_delta > 0:
                            in_set[x_in] = True
                            current_score += net_delta
                            improved = True
                            break
                        else:
                            in_set[x_out] = True
                    else:
                        in_set[x_out] = True
                if improved:
                    break

    # Main Loop: Multi-start with Simulated Annealing
    best_score = -1
    best_in_set = None

    random.seed(42)

    # Number of restarts
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > time_limit - 10:
            break

        greedy_init()
        hill_climb()

        if current_score > best_score:
            best_score = current_score
            best_in_set = in_set[:]

        # Simulated Annealing on the current state
        T = 5.0
        cooling = 0.999
        min_T = 0.01
        all_elements = list(range(1, N + 1))

        while time.time() < time_limit - 5:
            if T < min_T:
                T = min_T

            move_type = random.random()

            if move_type < 0.3:  # Add
                candidates = [x for x in all_elements if not in_set[x]]
                if candidates:
                    x = random.choice(candidates)
                    valid, delta = try_add(x)
                    if valid:
                        if delta > 0 or (delta < 0 and random.random() < math.exp(delta / T)) or (delta == 0 and random.random() < 0.3):
                            in_set[x] = True
                            current_score += delta
                            if current_score > best_score:
                                best_score = current_score
                                best_in_set = in_set[:]

            elif move_type < 0.5:  # Remove
                if any(in_set[1:]):
                    x = random.choice([x for x in all_elements if in_set[x]])
                    valid, delta = try_remove(x)
                    if valid:
                        if random.random() < math.exp(delta / T):
                            in_set[x] = False
                            current_score += delta

            else:  # Swap
                if any(in_set[1:]):
                    x_out = random.choice([x for x in all_elements if in_set[x]])
                    candidates = [x for x in all_elements if not in_set[x]]
                    if candidates:
                        x_in = random.choice(candidates)
                        # Check validity of swap
                        # 1. Calculate loss_remove on current state (with x_out)
                        loss = 0
                        for ap_idx, pos in elem_to_3aps[x_out]:
                            a, b, c = three_aps[ap_idx]
                            if pos == 0:
                                if in_set[b] and in_set[c]:
                                    loss += 1
                            elif pos == 1:
                                if in_set[a] and in_set[c]:
                                    loss += 1
                            else:
                                if in_set[a] and in_set[b]:
                                    loss += 1

                        # 2. Remove x_out
                        in_set[x_out] = False
                        # 3. Check if x_in is valid
                        valid_add, delta_add = try_add(x_in)
                        if valid_add:
                            net_delta = delta_add - loss
                            if net_delta > 0 or random.random() < math.exp(net_delta / T):
                                in_set[x_in] = True
                                current_score += net_delta
                                if current_score > best_score:
                                    best_score = current_score
                                    best_in_set = in_set[:]
                            else:
                                # Revert
                                in_set[x_out] = True
                        else:
                            # Invalid, revert
                            in_set[x_out] = True

            T *= cooling

    # Final Polish on the best found state
    if best_in_set is not None:
        in_set = best_in_set[:]
        # Recalculate score to be sure
        current_score = 0
        for x in range(1, N + 1):
            if in_set[x]:
                # Count 3-APs where x is the smallest element? No, just count all.
                # To avoid double counting, count only if x is the first element of the AP.
                # Or just use the delta logic? No, that's for changes.
                # Let's just iterate all 3-APs and check if all 3 are in set.
                pass

        # Efficient recount
        # For each 3-AP, check if all 3 are in set
        for a, b, c in three_aps:
            if in_set[a] and in_set[b] and in_set[c]:
                current_score += 1

        # Run a final hill climb to squeeze out any remaining gains
        hill_climb()

        if current_score > best_score:
            best_score = current_score
            best_in_set = in_set[:]

    # Convert best_in_set to sorted list
    result = []
    for x in range(1, N + 1):
        if best_in_set[x]:
            result.append(x)
    return sorted(result)
