import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit

    N = 256

    # Precompute 4-APs
    four_ap_by_elem = [[] for _ in range(N)]
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            a0 = a - 1
            b0 = a + d - 1
            c0 = a + 2 * d - 1
            d0 = a + 3 * d - 1
            mask = (1 << a0) | (1 << b0) | (1 << c0) | (1 << d0)
            four_ap_by_elem[a0].append(mask)
            four_ap_by_elem[b0].append(mask)
            four_ap_by_elem[c0].append(mask)
            four_ap_by_elem[d0].append(mask)

    # Precompute 3-AP neighbors
    three_ap_by_elem = [[] for _ in range(N)]
    for d in range(1, (N - 1) // 2 + 1):
        max_a = N - 2 * d
        for a in range(1, max_a + 1):
            a0 = a - 1
            b0 = a + d - 1
            c0 = a + 2 * d - 1
            three_ap_by_elem[a0].append((b0, c0))
            three_ap_by_elem[b0].append((a0, c0))
            three_ap_by_elem[c0].append((a0, b0))

    def can_add(state, x):
        for mask in four_ap_by_elem[x]:
            if (state & mask) == (mask ^ (1 << x)):
                return False
        return True

    def calc_delta_add(state, x):
        delta = 0
        for u, v in three_ap_by_elem[x]:
            if ((state >> u) & 1) and ((state >> v) & 1):
                delta += 1
        return delta

    def count_3aps(state):
        count = 0
        for x in range(N):
            if (state >> x) & 1:
                for u, v in three_ap_by_elem[x]:
                    if ((state >> u) & 1) and ((state >> v) & 1):
                        count += 1
        return count // 3

    # --- Phase 1: Density-Heuristic Greedy Construction with Structural Tie-Breaking ---
    best_state = 0
    best_score = 0

    num_restarts = 5

    for r in range(num_restarts):
        if time.time() > deadline - 20.0:
            break

        state = 0
        current_score = 0
        available = list(range(N))
        random.shuffle(available)

        # Greedy with structural density tie-breaking
        while available and time.time() < deadline - 15.0:
            best_x = -1
            best_gain = -1
            best_struct = -1
            best_idx = -1

            for idx, x in enumerate(available):
                if not can_add(state, x):
                    continue
                gain = calc_delta_add(state, x)
                # Structural density: count potential future 3-APs
                struct = 0
                for u, v in three_ap_by_elem[x]:
                    if ((state >> u) & 1) != ((state >> v) & 1):
                        struct += 1  # One neighbor present, one missing -> potential
                # Combine: primary gain, secondary structural
                if gain > best_gain or (gain == best_gain and struct > best_struct):
                    best_gain = gain
                    best_struct = struct
                    best_x = x
                    best_idx = idx

            if best_x == -1:
                break

            state |= (1 << best_x)
            current_score += best_gain
            available.pop(best_idx)

        if current_score > best_score:
            best_score = current_score
            best_state = state

    # --- Phase 2: Deterministic Targeted Local Search ---
    current_state = best_state
    current_score = best_score

    # Helper: try to add any element with positive gain
    def try_add_positive(state, score):
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            for x in range(N):
                if (state >> x) & 1:
                    continue
                if not can_add(state, x):
                    continue
                delta = calc_delta_add(state, x)
                if delta > 0:
                    state |= (1 << x)
                    score += delta
                    changed = True
        return state, score

    # Helper: perform one pass of best 2-swap
    def best_swap_pass(state, score):
        state_indices = [i for i in range(N) if (state >> i) & 1]
        non_state_indices = [i for i in range(N) if not (state >> i) & 1]
        
        if not state_indices or not non_state_indices:
            return state, score, False

        # Precompute loss for each x in state
        losses = {}
        for x in state_indices:
            loss = 0
            for u, v in three_ap_by_elem[x]:
                if ((state >> u) & 1) and ((state >> v) & 1):
                    loss += 1
            losses[x] = loss

        best_delta = 0
        best_x = -1
        best_y = -1

        for x in state_indices:
            state_minus_x = state & ~(1 << x)
            for y in non_state_indices:
                # Check validity of adding y to state_minus_x
                valid = True
                for mask in four_ap_by_elem[y]:
                    other_mask = mask ^ (1 << y)
                    if (state_minus_x & other_mask) == other_mask:
                        valid = False
                        break
                if not valid:
                    continue

                gain = 0
                for u, v in three_ap_by_elem[y]:
                    if ((state_minus_x >> u) & 1) and ((state_minus_x >> v) & 1):
                        gain += 1

                delta = gain - losses[x]
                if delta > best_delta:
                    best_delta = delta
                    best_x = x
                    best_y = y

        if best_x != -1:
            new_state = (state & ~(1 << best_x)) | (1 << best_y)
            return new_state, score + best_delta, True
        return state, score, False

    # Iterative local search: alternate add and swap
    max_ls_iters = 200
    for it in range(max_ls_iters):
        if time.time() > deadline - 5.0:
            break

        # Try to add
        current_state, current_score = try_add_positive(current_state, current_score)

        # Try to swap
        new_state, new_score, improved = best_swap_pass(current_state, current_score)
        if improved:
            current_state = new_state
            current_score = new_score
        else:
            # Try a random perturbation to escape local minimum
            if time.time() > deadline - 10.0:
                break
            state_indices = [i for i in range(N) if (current_state >> i) & 1]
            non_state_indices = [i for i in range(N) if not (current_state >> i) & 1]
            if not state_indices or not non_state_indices:
                break
            x = random.choice(state_indices)
            y = random.choice(non_state_indices)
            # Check validity
            state_minus_x = current_state & ~(1 << x)
            valid = True
            for mask in four_ap_by_elem[y]:
                other_mask = mask ^ (1 << y)
                if (state_minus_x & other_mask) == other_mask:
                    valid = False
                    break
            if valid:
                gain = 0
                for u, v in three_ap_by_elem[y]:
                    if ((state_minus_x >> u) & 1) and ((state_minus_x >> v) & 1):
                        gain += 1
                loss = 0
                for u, v in three_ap_by_elem[x]:
                    if ((current_state >> u) & 1) and ((current_state >> v) & 1):
                        loss += 1
                delta = gain - loss
                # Accept with probability based on delta (soft-max surrogate)
                if delta >= 0 or random.random() < math.exp(delta / 2.0):
                    current_state = (current_state & ~(1 << x)) | (1 << y)
                    current_score += delta

    # Final saturation
    current_state, current_score = try_add_positive(current_state, current_score)

    # Update best
    if current_score > best_score:
        best_score = current_score
        best_state = current_state

    # Convert to list of 1-based integers
    result = []
    for i in range(N):
        if (best_state >> i) & 1:
            result.append(i + 1)

    return sorted(result)
