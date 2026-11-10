import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    start_time = time.time()
    DEADLINE = 115.0  # safety margin from 120s

    # Convert to numpy arrays for speed
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)

    # Filter to feasible items
    feasible_mask = (weights > 0) & (weights <= remaining_capacity)
    feasible_indices = np.where(feasible_mask)[0]

    # Also check for zero-weight positive-value items
    zero_weight_pos = (weights == 0) & (values > 0)
    if np.any(zero_weight_pos):
        return int(np.where(zero_weight_pos)[0][0])

    if len(feasible_indices) == 0:
        return -1

    f_weights = weights[feasible_indices]
    f_values = values[feasible_indices]
    f_indices = feasible_indices
    m = len(f_indices)

    # Compute densities
    densities = f_values / f_weights

    # Sort by density descending
    order = np.argsort(-densities)
    sorted_f_weights = f_weights[order]
    sorted_f_values = f_values[order]
    sorted_f_indices = f_indices[order]
    sorted_densities = densities[order]

    # Greedy fill baseline
    greedy_val = 0.0
    greedy_set = set()
    rem_cap = remaining_capacity
    for i in range(m):
        if sorted_f_weights[i] <= rem_cap:
            greedy_val += sorted_f_values[i]
            greedy_set.add(sorted_f_indices[i])
            rem_cap -= sorted_f_weights[i]

    best_total = greedy_val
    best_first = sorted_f_indices[0]

    # Try each of the top candidates as the first pick, then greedy fill the rest
    num_candidates = min(100, m)
    for ci in range(num_candidates):
        idx_first = sorted_f_indices[ci]
        w_first = sorted_f_weights[ci]
        v_first = sorted_f_values[ci]
        if w_first > remaining_capacity:
            continue
        rem_after = remaining_capacity - w_first
        total = v_first
        for j in range(m):
            if j == ci:
                continue
            if sorted_f_weights[j] <= rem_after:
                total += sorted_f_values[j]
                rem_after -= sorted_f_weights[j]
        if total > best_total:
            best_total = total
            best_first = idx_first

    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return int(best_first)

    # Local search: 1-swap improvement on the greedy solution
    # Build lookup dicts
    w_dict = dict(zip(f_indices, f_weights))
    v_dict = dict(zip(f_indices, f_values))
    d_dict = dict(zip(f_indices, densities))

    current_solution = list(greedy_set)
    current_value = greedy_val
    current_weight = remaining_capacity - rem_cap

    improved = True
    max_iters = 200
    it = 0
    while improved and it < max_iters:
        improved = False
        it += 1
        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            break

        best_swap_val = current_value
        best_swap_out = None
        best_swap_in = None

        for idx_out in current_solution:
            w_out = w_dict[idx_out]
            v_out = v_dict[idx_out]
            new_weight = current_weight - w_out
            new_cap = remaining_capacity - new_weight

            # Try adding each item not in solution
            for idx_in in f_indices:
                if idx_in in greedy_set:
                    continue
                w_in = w_dict[idx_in]
                v_in = v_dict[idx_in]
                if w_in <= new_cap:
                    new_val = current_value - v_out + v_in
                    if new_val > best_swap_val + 1e-9:
                        best_swap_val = new_val
                        best_swap_out = idx_out
                        best_swap_in = idx_in

        if best_swap_out is not None:
            greedy_set.remove(best_swap_out)
            greedy_set.add(best_swap_in)
            current_value = best_swap_val
            current_weight += w_dict[best_swap_in] - w_dict[best_swap_out]
            improved = True

    if current_value > best_total:
        best_total = current_value
        # Pick the highest density item in the solution as first
        best_density = -1
        for idx in greedy_set:
            if d_dict[idx] > best_density:
                best_density = d_dict[idx]
                best_first = idx

    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return int(best_first)

    # Try a DP-based approach with scaled weights
    SCALE = 50
    cap_scaled = int(round(remaining_capacity * SCALE))

    if 0 < cap_scaled < 20000:
        # 1D DP
        dp = np.zeros(cap_scaled + 1, dtype=np.float64)
        # Track which item was added at each state for backtracking
        # Use a 2D array: track[i][w] = original index or -1
        # To save memory, use int32
        track = np.full((m, cap_scaled + 1), -1, dtype=np.int32)

        for i in range(m):
            w_scaled = int(round(f_weights[i] * SCALE))
            if w_scaled <= 0 or w_scaled > cap_scaled:
                continue
            v = f_values[i]
            # Process in reverse
            for cw in range(cap_scaled, w_scaled - 1, -1):
                prev_val = dp[cw - w_scaled] + v
                if prev_val > dp[cw] + 1e-9:
                    dp[cw] = prev_val
                    track[i, cw] = f_indices[i]

            elapsed = time.time() - start_time
            if elapsed > DEADLINE:
                break

        # Reconstruct
        if elapsed <= DEADLINE:
            optimal_items = set()
            cw = cap_scaled
            for i in range(m - 1, -1, -1):
                idx = track[i, cw]
                if idx != -1:
                    optimal_items.add(idx)
                    # Find the scaled weight
                    pos = np.where(f_indices == idx)[0]
                    if len(pos) > 0:
                        cw -= int(round(f_weights[pos[0]] * SCALE))

            dp_total = dp[cap_scaled]
            if dp_total > best_total + 1e-9:
                best_total = dp_total
                # Pick highest density item from optimal solution
                best_density = -1
                for idx in optimal_items:
                    if d_dict[idx] > best_density:
                        best_density = d_dict[idx]
                        best_first = idx

    # Final check: verify best_first is feasible
    if best_first < 0 or best_first >= n:
        return -1
    if weights[best_first] > remaining_capacity:
        # Fallback to first feasible item
        for idx in f_indices:
            if weights[idx] <= remaining_capacity:
                best_first = idx
                break
        else:
            return -1

    return int(best_first)
