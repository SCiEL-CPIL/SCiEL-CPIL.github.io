"""Structural change: WalkSAT-style MaxSAT for 4-AP-free set maximizing 3-AP count, with strict hill-climbing polish."""
import time
import random
import math

N = 256
TIME_LIMIT = 84.0


def construct():
    start_time = time.time()
    rng = random.Random(42)

    # Precompute all 4-APs and 3-APs (0-indexed)
    four_aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(0, N - 3 * d):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))

    three_aps = []
    for d in range(1, (N - 2) // 2 + 1):
        for a in range(0, N - 2 * d):
            three_aps.append((a, a + d, a + 2 * d))

    # For each element, which 4-APs does it participate in? Store other elements
    elem_to_4aps = [[] for _ in range(N)]
    for ap in four_aps:
        for i, x in enumerate(ap):
            others = tuple(ap[j] for j in range(4) if j != i)
            elem_to_4aps[x].append(others)

    # For each element, which 3-APs does it participate in? Store other elements
    elem_to_3aps = [[] for _ in range(N)]
    for ap in three_aps:
        for i, x in enumerate(ap):
            others = tuple(ap[j] for j in range(3) if j != i)
            elem_to_3aps[x].append(others)

    # Precompute weights
    elem_3ap_weight = [len(elem_to_3aps[x]) for x in range(N)]
    elem_4ap_weight = [len(elem_to_4aps[x]) for x in range(N)]

    # Build 4-AP adjacency: for each 4-AP, store its elements
    # Build 3-AP adjacency: for each 3-AP, store its elements
    # Also build: for each pair of elements, which 3-APs they share (for fast gain computation)
    # This is expensive, so we'll use a different approach

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
        for others in elem_to_4aps[x]:
            ok = True
            for y in others:
                if not ((S >> y) & 1):
                    ok = False
                    break
            if ok:
                return False
        return True

    def gain_3aps(S, x):
        """Number of new 3-APs formed by adding x to S."""
        gain = 0
        for others in elem_to_3aps[x]:
            ok = True
            for y in others:
                if not ((S >> y) & 1):
                    ok = False
                    break
            if ok:
                gain += 1
        return gain

    def loss_3aps(S, x):
        """Number of 3-APs lost by removing x from S (x must be in S)."""
        loss = 0
        for others in elem_to_3aps[x]:
            ok = True
            for y in others:
                if not ((S >> y) & 1):
                    ok = False
                    break
            if ok:
                loss += 1
        return loss

    def repair_4aps(S):
        """Remove elements to eliminate all 4-APs, prioritizing low 3-AP weight."""
        changed = True
        while changed:
            changed = False
            for a, b, c, d in four_aps:
                if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1 and (S >> d) & 1:
                    elems = (a, b, c, d)
                    worst_idx = min(range(4), key=lambda i: (elem_3ap_weight[elems[i]], -elem_4ap_weight[elems[i]]))
                    worst = elems[worst_idx]
                    S &= ~(1 << worst)
                    changed = True
                    break
        return S

    def greedy_construct(order):
        """Greedy: add elements in given order, skip if creates 4-AP."""
        S = 0
        for x in order:
            if is_safe_add(S, x):
                S |= (1 << x)
        return S

    def strict_hill_climb(S, max_iters=500, deadline=None):
        """Strict hill-climbing: try all single adds and swaps, take best improving move."""
        current_S = S
        current_score = count_3aps_bitmask(current_S)

        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break

            members = [x for x in range(N) if (current_S >> x) & 1]
            non_members = [x for x in range(N) if not ((current_S >> x) & 1)]

            if not members or not non_members:
                break

            best_move = None
            best_gain = 0

            # Try single adds
            for x in non_members:
                if not is_safe_add(current_S, x):
                    continue
                g = gain_3aps(current_S, x)
                if g > best_gain:
                    best_gain = g
                    best_move = ('add', x, None)

            # Try swaps: remove x, add y
            for x_remove in members:
                S_after_remove = current_S & ~(1 << x_remove)
                loss = loss_3aps(current_S, x_remove)
                for x_add in non_members:
                    if not is_safe_add(S_after_remove, x_add):
                        continue
                    gain = gain_3aps(S_after_remove, x_add)
                    net = gain - loss
                    if net > best_gain:
                        best_gain = net
                        best_move = ('swap', x_add, x_remove)

            if best_move is None or best_gain <= 0:
                break

            if best_move[0] == 'add':
                current_S |= (1 << best_move[1])
            else:
                current_S = (current_S & ~(1 << best_move[2])) | (1 << best_move[1])
            current_score += best_gain

        return current_S, current_score

    def walksat(S_init, time_budget):
        """WalkSAT-style: maintain a 4-AP violation counter, flip variables to reduce violations and maximize 3-APs."""
        current_S = S_init
        current_3ap = count_3aps_bitmask(current_S)
        current_4ap = count_4aps_bitmask(current_S)

        # Ensure feasibility first
        if current_4ap > 0:
            current_S = repair_4aps(current_S)
            current_3ap = count_3aps_bitmask(current_S)
            current_4ap = count_4aps_bitmask(current_S)

        best_S = current_S
        best_score = current_3ap

        deadline = time.time() + time_budget
        T = 5.0
        alpha = 0.998
        T_min = 0.3

        polish_interval = 150
        step_count = 0

        while time.time() < deadline:
            step_count += 1

            # Periodic strict hill-climb polish
            if step_count % polish_interval == 0:
                polished_S, polished_score = strict_hill_climb(current_S, max_iters=80, deadline=deadline)
                if polished_score > current_3ap:
                    current_S = polished_S
                    current_3ap = polished_score
                    current_4ap = count_4aps_bitmask(current_S)
                if current_3ap > best_score:
                    best_score = current_3ap
                    best_S = current_S
                continue

            members = [x for x in range(N) if (current_S >> x) & 1]
            non_members = [x for x in range(N) if not ((current_S >> x) & 1)]

            if not members or not non_members:
                break

            # WalkSAT: pick a variable to flip
            # With probability proportional to "violation pressure", pick from elements in 4-APs
            # Otherwise pick randomly

            move_type = rng.random()

            if move_type < 0.5:
                # Swap: remove x, add y
                x_remove = rng.choice(members)
                x_add = rng.choice(non_members)
                S_after_remove = current_S & ~(1 << x_remove)
                if not is_safe_add(S_after_remove, x_add):
                    T *= alpha
                    if T < T_min:
                        T = T_min
                    continue
                loss = loss_3aps(current_S, x_remove)
                gain = gain_3aps(S_after_remove, x_add)
                delta_3ap = gain - loss
                new_S = S_after_remove | (1 << x_add)
                new_3ap = current_3ap + delta_3ap
                new_4ap = count_4aps_bitmask(new_S)

                penalty = 1000
                old_score = current_3ap - penalty * current_4ap
                new_score = new_3ap - penalty * new_4ap
                delta = new_score - old_score

                if delta >= 0 or (T > T_min and rng.random() < math.exp(delta / T)):
                    current_S = new_S
                    current_3ap = new_3ap
                    current_4ap = new_4ap
                    if current_4ap == 0 and current_3ap > best_score:
                        best_score = current_3ap
                        best_S = current_S
                T *= alpha
                if T < T_min:
                    T = T_min

            elif move_type < 0.75:
                # Single add
                x = rng.choice(non_members)
                if not is_safe_add(current_S, x):
                    T *= alpha
                    if T < T_min:
                        T = T_min
                    continue
                gain = gain_3aps(current_S, x)
                new_S = current_S | (1 << x)
                new_3ap = current_3ap + gain
                new_4ap = count_4aps_bitmask(new_S)

                penalty = 1000
                old_score = current_3ap - penalty * current_4ap
                new_score = new_3ap - penalty * new_4ap
                delta = new_score - old_score

                if delta >= 0 or (T > T_min and rng.random() < math.exp(delta / T)):
                    current_S = new_S
                    current_3ap = new_3ap
                    current_4ap = new_4ap
                    if current_4ap == 0 and current_3ap > best_score:
                        best_score = current_3ap
                        best_S = current_S
                T *= alpha
                if T < T_min:
                    T = T_min

            else:
                # Single remove
                x = rng.choice(members)
                loss = loss_3aps(current_S, x)
                new_S = current_S & ~(1 << x)
                new_3ap = current_3ap - loss
                new_4ap = count_4aps_bitmask(new_S)

                penalty = 1000
                old_score = current_3ap - penalty * current_4ap
                new_score = new_3ap - penalty * new_4ap
                delta = new_score - old_score

                if delta >= 0 or (T > T_min and rng.random() < math.exp(delta / T)):
                    current_S = new_S
                    current_3ap = new_3ap
                    current_4ap = new_4ap
                    if current_4ap == 0 and current_3ap > best_score:
                        best_score = current_3ap
                        best_S = current_S
                T *= alpha
                if T < T_min:
                    T = T_min

        # Final strict hill-climb
        if time.time() < deadline - 1:
            polished_S, polished_score = strict_hill_climb(best_S, max_iters=200, deadline=deadline)
            if polished_score > best_score:
                best_score = polished_score
                best_S = polished_S

        return best_S, best_score

    # Phase 1: Greedy construction
    order = sorted(range(N), key=lambda x: (-elem_3ap_weight[x], x))
    S_greedy = greedy_construct(order)
    S_greedy = repair_4aps(S_greedy)
    score_greedy = count_3aps_bitmask(S_greedy)

    # Phase 2: Strict hill-climb on greedy
    S_shc, score_shc = strict_hill_climb(S_greedy, max_iters=300)
    if score_shc > score_greedy:
        S_best = S_shc
        best_score = score_shc
    else:
        S_best = S_greedy
        best_score = score_greedy

    # Phase 3: WalkSAT with multiple restarts
    deadline = start_time + TIME_LIMIT - 5
    num_restarts = 12

    for i in range(num_restarts):
        if time.time() > deadline - 2:
            break

        if i == 0:
            S_init = S_best
        else:
            # Diverse initial solutions
            if i % 4 == 1:
                # Weighted greedy with random tie-breaking
                S_init = 0
                order2 = list(range(N))
                rng.shuffle(order2)
                for x in order2:
                    if is_safe_add(S_init, x):
                        S_init |= (1 << x)
                S_init = repair_4aps(S_init)
            elif i % 4 == 2:
                # Random subset then repair
                S_init = 0
                for x in range(N):
                    if rng.random() < 0.5:
                        S_init |= (1 << x)
                S_init = repair_4aps(S_init)
            elif i % 4 == 3:
                # Perturb best
                S_init = S_best
                for _ in range(rng.randint(8, 20)):
                    x = rng.randint(0, N - 1)
                    if (S_init >> x) & 1:
                        S_init &= ~(1 << x)
                    else:
                        S_init |= (1 << x)
                S_init = repair_4aps(S_init)
            else:
                # Greedy with different weight ordering
                S_init = 0
                order3 = sorted(range(N), key=lambda x: (-elem_4ap_weight[x], x))
                for x in order3:
                    if is_safe_add(S_init, x):
                        S_init |= (1 << x)
                S_init = repair_4aps(S_init)

        time_left = deadline - time.time()
        if time_left < 1.0:
            break
        restart_time = min(time_left - 1.0, 5.0)

        S_ws, score_ws = walksat(S_init, restart_time)

        if score_ws > best_score:
            best_score = score_ws
            S_best = S_ws

    # Final strict hill-climb on best
    if time.time() < deadline - 1:
        S_final, score_final = strict_hill_climb(S_best, max_iters=300, deadline=deadline)
        if score_final > best_score:
            best_score = score_final
            S_best = S_final

    # Verify no 4-APs
    assert count_4aps_bitmask(S_best) == 0, "4-AP violation found!"

    # Convert bitmask to sorted list (1-indexed)
    result = [x + 1 for x in range(N) if (S_best >> x) & 1]

    if not result:
        S_fb = 0
        for x in range(N):
            if is_safe_add(S_fb, x):
                S_fb |= (1 << x)
        result = [x + 1 for x in range(N) if (S_fb >> x) & 1]

    return result
