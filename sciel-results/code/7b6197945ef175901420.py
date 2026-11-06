"""
Maximum 3-AP count in a 4-AP-free subset of [1,256].
Uses simulated annealing with efficient bitmask operations and
multiple restarts with diverse initial solutions.
"""
import time
import random
import math

N = 256
TIME_LIMIT = 85.0  # safety margin below 90s


def construct():
    start_time = time.time()
    rng = random.Random(12345)

    # Precompute all 4-APs and 3-APs (0-indexed)
    four_aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(0, N - 3 * d):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))

    three_aps = []
    for d in range(1, (N - 2) // 2 + 1):
        for a in range(0, N - 2 * d):
            three_aps.append((a, a + d, a + 2 * d))

    # For each element, precompute which 3-APs it's in (as index list)
    elem_to_3ap_indices = [[] for _ in range(N)]
    for idx, (a, b, c) in enumerate(three_aps):
        elem_to_3ap_indices[a].append(idx)
        elem_to_3ap_indices[b].append(idx)
        elem_to_3ap_indices[c].append(idx)

    # For each element, precompute which 4-APs it's in (as index list)
    elem_to_4ap_indices = [[] for _ in range(N)]
    for idx, (a, b, c, d) in enumerate(four_aps):
        elem_to_4ap_indices[a].append(idx)
        elem_to_4ap_indices[b].append(idx)
        elem_to_4ap_indices[c].append(idx)
        elem_to_4ap_indices[d].append(idx)

    def count_3aps_bitmask(S):
        """Count 3-APs in bitmask S (0-indexed)."""
        cnt = 0
        for a, b, c in three_aps:
            if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1:
                cnt += 1
        return cnt

    def count_4aps_bitmask(S):
        """Count 4-APs fully contained in bitmask S."""
        cnt = 0
        for a, b, c, d in four_aps:
            if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1 and (S >> d) & 1:
                cnt += 1
        return cnt

    def is_safe_add(S, x):
        """Check if adding x to S (bitmask) creates any 4-AP involving x."""
        for ap_idx in elem_to_4ap_indices[x]:
            a, b, c, d = four_aps[ap_idx]
            # Check if the other 3 elements are all in S
            mask = 0
            for y in (a, b, c, d):
                if y != x:
                    mask |= (1 << y)
            if (S & mask) == mask:
                return False
        return True

    def gain_3aps(S, x):
        """Number of new 3-APs formed by adding x to S."""
        gain = 0
        for ap_idx in elem_to_3ap_indices[x]:
            a, b, c = three_aps[ap_idx]
            mask = 0
            for y in (a, b, c):
                if y != x:
                    mask |= (1 << y)
            if (S & mask) == mask:
                gain += 1
        return gain

    def loss_3aps(S, x):
        """Number of 3-APs lost by removing x from S (x must be in S)."""
        loss = 0
        for ap_idx in elem_to_3ap_indices[x]:
            a, b, c = three_aps[ap_idx]
            mask = 0
            for y in (a, b, c):
                if y != x:
                    mask |= (1 << y)
            if (S & mask) == mask:
                loss += 1
        return loss

    def greedy_construct(seed_order=None):
        """Greedy: add elements in order of decreasing 3-AP weight, skip if creates 4-AP."""
        S = 0
        if seed_order is None:
            # Weight: number of 3-APs each element participates in
            weights = [len(elem_to_3ap_indices[x]) for x in range(N)]
            order = sorted(range(N), key=lambda x: (-weights[x], x))
        else:
            order = seed_order
        for x in order:
            if is_safe_add(S, x):
                S |= (1 << x)
        return S

    def repair_4aps(S):
        """Remove elements to eliminate all 4-APs."""
        changed = True
        while changed:
            changed = False
            for a, b, c, d in four_aps:
                if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1 and (S >> d) & 1:
                    # Remove the element with lowest 3-AP weight
                    weights = [len(elem_to_3ap_indices[x]) for x in (a, b, c, d)]
                    worst_idx = min(range(4), key=lambda i: weights[i])
                    worst = (a, b, c, d)[worst_idx]
                    S &= ~(1 << worst)
                    changed = True
                    break
        return S

    def simulated_annealing(S_init, time_limit, T_start=10.0, T_end=0.1, alpha=0.999):
        """Simulated annealing to maximize 3-AP count while maintaining no 4-APs."""
        current_S = S_init
        current_3ap = count_3aps_bitmask(current_S)
        current_4ap = count_4aps_bitmask(current_S)

        # Ensure feasibility at start
        if current_4ap > 0:
            current_S = repair_4aps(current_S)
            current_3ap = count_3aps_bitmask(current_S)
            current_4ap = count_4aps_bitmask(current_S)

        best_S = current_S
        best_score = current_3ap

        T = T_start

        while time.time() < time_limit:
            # Pick an element to flip
            # Bias toward adding non-members or removing members
            if rng.random() < 0.5:
                # Try to add a random non-member
                non_members = [x for x in range(N) if not ((current_S >> x) & 1)]
                if non_members:
                    x = rng.choice(non_members)
                else:
                    x = rng.randint(0, N - 1)
            else:
                # Try to remove a random member
                members = [x for x in range(N) if (current_S >> x) & 1]
                if members:
                    x = rng.choice(members)
                else:
                    x = rng.randint(0, N - 1)

            # Flip x
            if (current_S >> x) & 1:
                # Remove x
                new_S = current_S & ~(1 << x)
                delta_3ap = -loss_3aps(current_S, x)
            else:
                # Add x
                new_S = current_S | (1 << x)
                delta_3ap = gain_3aps(current_S, x)

            new_3ap = current_3ap + delta_3ap
            new_4ap = count_4aps_bitmask(new_S)

            # Score: 3-AP count minus large penalty for 4-AP violations
            penalty = 1000
            old_score = current_3ap - penalty * current_4ap
            new_score = new_3ap - penalty * new_4ap

            delta = new_score - old_score

            # Accept if improving, or with probability exp(delta/T)
            if delta >= 0 or (T > T_end and rng.random() < math.exp(delta / T)):
                current_S = new_S
                current_3ap = new_3ap
                current_4ap = new_4ap

                # Track best feasible solution
                if current_4ap == 0 and current_3ap > best_score:
                    best_score = current_3ap
                    best_S = current_S

            T *= alpha
            if T < T_end:
                T = T_end

        return best_S, best_score

    # Phase 1: Greedy construction
    S_greedy = greedy_construct()
    S_greedy = repair_4aps(S_greedy)
    score_greedy = count_3aps_bitmask(S_greedy)

    # Phase 2: Simulated annealing on greedy solution
    deadline = start_time + TIME_LIMIT - 5
    S_sa, score_sa = simulated_annealing(S_greedy, min(deadline, start_time + 30))
    if score_sa > score_greedy:
        S_best = S_sa
        best_score = score_sa
    else:
        S_best = S_greedy
        best_score = score_greedy

    # Phase 3: Multiple restarts with diverse initial solutions
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 2:
            break

        # Generate diverse initial solutions
        if i == 0:
            S_init = S_best
        else:
            # Randomized greedy with different orderings
            order = list(range(N))
            rng.shuffle(order)
            S_init = greedy_construct(order)
            S_init = repair_4aps(S_init)

        # Allocate time for this restart
        time_left = deadline - time.time()
        if time_left < 2.0:
            break
        restart_time = start_time + min(time_left - 1.0, 4.0)

        S_sa, score_sa = simulated_annealing(S_init, restart_time)

        if score_sa > best_score:
            best_score = score_sa
            S_best = S_sa

    # Final local search on best solution
    if time.time() < deadline - 1:
        S_final, score_final = simulated_annealing(S_best, deadline, T_start=5.0, T_end=0.05, alpha=0.9995)
        if score_final > best_score:
            best_score = score_final
            S_best = S_final

    # Verify no 4-APs
    assert count_4aps_bitmask(S_best) == 0, "4-AP violation found!"

    # Convert bitmask to sorted list (1-indexed)
    result = [x + 1 for x in range(N) if (S_best >> x) & 1]

    if not result:
        # Fallback
        S_fb = 0
        for x in range(N):
            if is_safe_add(S_fb, x):
                S_fb |= (1 << x)
        result = [x + 1 for x in range(N) if (S_fb >> x) & 1]

    return result
