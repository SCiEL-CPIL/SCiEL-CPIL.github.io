import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 110s margin out of 150s total

    # Convert to numpy arrays for vectorization
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    C = float(remaining_capacity)

    # Filter feasible items (weight <= capacity)
    feasible_mask = w <= C
    if not np.any(feasible_mask):
        return -1

    feasible_indices = np.where(feasible_mask)[0]
    w_f = w[feasible_mask]
    v_f = v[feasible_mask]
    n_f = len(w_f)

    if n_f == 0:
        return -1

    # Compute densities for all feasible items
    d_f = np.zeros(n_f)
    non_zero_w = w_f > 0
    d_f[non_zero_w] = v_f[non_zero_w] / w_f[non_zero_w]
    d_f[~non_zero_w] = np.where(v_f[~non_zero_w] > 0, np.inf, 0.0)

    # Sort all feasible items by density descending (for the LP bound)
    sort_idx = np.argsort(-d_f, kind='mergesort')
    w_s = w_f[sort_idx]
    v_s = v_f[sort_idx]
    d_s = d_f[sort_idx]

    # Precompute prefix sums of weights and values in sorted order
    pw = np.cumsum(w_s)
    pv = np.cumsum(v_s)

    def get_lp_value(cap, exclude_local_idx):
        """
        Calculate the fractional knapsack value for capacity 'cap'
        using the pre-sorted items, EXCLUDING the item at local index exclude_local_idx.
        """
        if cap <= 1e-9:
            return 0.0

        # Build the sorted list excluding the item
        # We need to find the position of exclude_local_idx in the sorted order
        # sort_idx maps sorted position -> original feasible index
        # We need: for each sorted position p, the original feasible index is sort_idx[p]
        # exclude_local_idx is an original feasible index
        # Find which sorted position has sort_idx[p] == exclude_local_idx

        # Since n_f <= 100, we can just build the filtered arrays
        # But for speed, let's use the precomputed sorted arrays and skip the excluded item

        # Find position of excluded item in sorted order
        # sort_idx[p] = original feasible index p maps to
        # We want p such that sort_idx[p] == exclude_local_idx
        # This is the inverse permutation

        # For small n, just build filtered arrays
        mask = np.ones(n_f, dtype=bool)
        # exclude_local_idx is the index in w_f (feasible array)
        mask[exclude_local_idx] = False
        # But we need to apply this to the sorted order
        # sorted position p has original index sort_idx[p]
        # We exclude if sort_idx[p] == exclude_local_idx
        keep_mask = (sort_idx != exclude_local_idx)
        w_kept = w_s[keep_mask]
        v_kept = v_s[keep_mask]
        d_kept = d_s[keep_mask]

        if len(w_kept) == 0:
            return 0.0

        pw_kept = np.cumsum(w_kept)
        pv_kept = np.cumsum(v_kept)

        # Find the largest index k such that pw_kept[k] <= cap
        k = np.searchsorted(pw_kept, cap, side='right') - 1

        if k < 0:
            # Cap is less than the weight of the first kept item
            if cap <= 0:
                return 0.0
            return cap * d_kept[0]

        val = pv_kept[k]
        remaining_cap = cap - pw_kept[k]

        # If there is a next item, take a fraction of it
        if k < len(w_kept) - 1:
            next_d = d_kept[k + 1]
            val += remaining_cap * next_d

        return val

    best_score = -1.0
    best_i_local = -1

    # Lagrangian relaxation: for each candidate item i,
    # compute v[i] + LP_bound(C - w[i], excluding item i)
    # This properly accounts for opportunity cost
    for i in range(n_f):
        rem_cap = C - w_f[i]
        if rem_cap < 0:
            continue
        lp_rem = get_lp_value(rem_cap, i)
        score = v_f[i] + lp_rem

        if score > best_score:
            best_score = score
            best_i_local = i

        # Check time
        if time.time() > deadline:
            break

    # Fallback: if the Lagrangian approach didn't find anything, use density greedy
    if best_i_local == -1:
        for i in range(n_f):
            if w_f[i] > C:
                continue
            best_i_local = i
            break

    if best_i_local == -1:
        return -1

    return int(feasible_indices[best_i_local])
