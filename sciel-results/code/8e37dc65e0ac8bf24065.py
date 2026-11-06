"""4-AP-free set maximizing 3-AP count: structured multi-start + full 2-opt local search."""
import time
import random
import math

N = 256
TIME_LIMIT = 85.0  # safety margin below 90s


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

    # For each element, store the OTHER elements in each 4-AP it participates in
    elem_to_4aps = [[] for _ in range(N)]
    for ap in four_aps:
        for i, x in enumerate(ap):
            others = tuple(ap[j] for j in range(4) if j != i)
            elem_to_4aps[x].append(others)

    # For each element, store the OTHER elements in each 3-AP it participates in
    elem_to_3aps = [[] for _ in range(N)]
    for ap in three_aps:
        for i, x in enumerate(ap):
            others = tuple(ap[j] for j in range(3) if j != i)
            elem_to_3aps[x].append(others)

    # Precompute weights
    elem_3ap_weight = [len(elem_to_3aps[x]) for x in range(N)]
    elem_4ap_weight = [len(elem_to_4aps[x]) for x in range(N)]

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

    def local_search_2opt(S, max_iters=500):
        """Full 2-opt local search: try all (remove x, add y) pairs to improve 3-AP count.
        Maintains no 4-APs constraint. Uses incremental scoring."""
        current_S = S
        current_score = count_3aps_bitmask(current_S)

        for it in range(max_iters):
            if time.time() - start_time > TIME_LIMIT - 3:
                break

            improved = False

            # Get members and non-members
            members = [x for x in range(N) if (current_S >> x) & 1]
            non_members = [x for x in range(N) if not ((current_S >> x) & 1)]

            # Try single adds
            for x in non_members:
                if not is_safe_add(current_S, x):
                    continue
                new_score = current_score + gain_3aps(current_S, x)
                if new_score > current_score:
                    current_S |= (1 << x)
                    current_score = new_score
                    improved = True
                    break

            if improved:
                continue

            # Try swaps: remove x, add y (full 2-opt)
            best_swap = None
            best_gain = 0
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
                        best_swap = (x_remove, x_add)

            if best_swap is not None and best_gain > 0:
                x_remove, x_add = best_swap
                current_S = (current_S & ~(1 << x_remove)) | (1 << x_add)
                current_score += best_gain
                improved = True
                continue

            # No improvement possible
            break

        return current_S, current_score

    def local_search_2swap(S, max_iters=200):
        """Enhanced local search with 2-swap moves (remove two, add two) for escaping local optima."""
        current_S = S
        current_score = count_3aps_bitmask(current_S)

        for it in range(max_iters):
            if time.time() - start_time > TIME_LIMIT - 3:
                break

            improved = False

            members = [x for x in range(N) if (current_S >> x) & 1]
            non_members = [x for x in range(N) if not ((current_S >> x) & 1)]

            # Try single adds
            for x in non_members:
                if not is_safe_add(current_S, x):
                    continue
                new_score = current_score + gain_3aps(current_S, x)
                if new_score > current_score:
                    current_S |= (1 << x)
                    current_score = new_score
                    improved = True
                    break
            if improved:
                continue

            # Try single swaps
            best_swap = None
            best_gain = 0
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
                        best_swap = (x_remove, x_add)

            if best_swap is not None and best_gain > 0:
                x_remove, x_add = best_swap
                current_S = (current_S & ~(1 << x_remove)) | (1 << x_add)
                current_score += best_gain
                improved = True
                continue

            # Try 2-swap: remove two elements, add two elements
            if len(members) >= 2 and len(non_members) >= 2:
                best_2swap = None
                best_2gain = 0
                # Sample a subset of pairs to keep it tractable
                rng.shuffle(members)
                rng.shuffle(non_members)
                members_sample = members[:min(30, len(members))]
                non_members_sample = non_members[:min(30, len(non_members))]

                for i in range(len(members_sample)):
                    x1 = members_sample[i]
                    for j in range(i + 1, len(members_sample)):
                        x2 = members_sample[j]
                        S_after_remove = current_S & ~(1 << x1) & ~(1 << x2)
                        loss = loss_3aps(current_S, x1) + loss_3aps(S_after_remove | (1 << x1), x2)
                        # More accurate: loss of 3-APs when removing both
                        # loss = number of 3-APs that had x1 or x2 (and are lost when both removed)
                        # Simpler approximation: sum of individual losses minus overlap
                        for x3 in non_members_sample:
                            if not is_safe_add(S_after_remove, x3):
                                continue
                            for x4 in non_members_sample:
                                if x4 == x3:
                                    continue
                                if not is_safe_add(S_after_remove | (1 << x3), x4):
                                    continue
                                S_new = S_after_remove | (1 << x3) | (1 << x4)
                                gain = gain_3aps(S_after_remove, x3) + gain_3aps(S_after_remove | (1 << x3), x4)
                                net = gain - loss
                                if net > best_2gain:
                                    best_2gain = net
                                    best_2swap = (x1, x2, x3, x4)

                if best_2swap is not None and best_2gain > 0:
                    x1, x2, x3, x4 = best_2swap
                    current_S = (current_S & ~(1 << x1) & ~(1 << x2)) | (1 << x3) | (1 << x4)
                    current_score += best_2gain
                    improved = True
                    continue

            # No improvement possible
            break

        return current_S, current_score

    def walksat(S_init, time_limit):
        """WalkSAT-style local search with simulated annealing for escaping local optima."""
        current_S = S_init
        current_3ap = count_3aps_bitmask(current_S)
        current_4ap = count_4aps_bitmask(current_S)

        if current_4ap > 0:
            current_S = repair_4aps(current_S)
            current_3ap = count_3aps_bitmask(current_S)
            current_4ap = count_4aps_bitmask(current_S)

        best_S = current_S
        best_score = current_3ap

        T = 3.0
        alpha = 0.997

        while time.time() < time_limit:
            # Pick a variable to flip
            if current_4ap > 0:
                conflict_elems = set()
                for a, b, c, d in four_aps:
                    if (current_S >> a) & 1 and (current_S >> b) & 1 and (current_S >> c) & 1 and (current_S >> d) & 1:
                        conflict_elems.update([a, b, c, d])
                if conflict_elems:
                    x = rng.choice(list(conflict_elems))
                else:
                    x = rng.randint(0, N - 1)
            else:
                if rng.random() < 0.5:
                    non_members = [x for x in range(N) if not ((current_S >> x) & 1)]
                    if non_members:
                        x = rng.choice(non_members)
                    else:
                        x = rng.randint(0, N - 1)
                else:
                    members = [x for x in range(N) if (current_S >> x) & 1]
                    if members:
                        x = rng.choice(members)
                    else:
                        x = rng.randint(0, N - 1)

            # Flip x
            if (current_S >> x) & 1:
                new_S = current_S & ~(1 << x)
                delta_3ap = -loss_3aps(current_S, x)
            else:
                new_S = current_S | (1 << x)
                delta_3ap = gain_3aps(current_S, x)

            new_3ap = current_3ap + delta_3ap
            new_4ap = count_4aps_bitmask(new_S)

            penalty = 1000
            old_score = current_3ap - penalty * current_4ap
            new_score = new_3ap - penalty * new_4ap

            delta = new_score - old_score

            if delta >= 0 or (T > 0.01 and rng.random() < math.exp(delta / T)):
                current_S = new_S
                current_3ap = new_3ap
                current_4ap = new_4ap

                if current_4ap == 0 and current_3ap > best_score:
                    best_score = current_3ap
                    best_S = current_S

            T *= alpha
            if T < 0.1:
                T = 0.1

        return best_S, best_score

    # Phase 1: Multiple diverse greedy constructions
    S_best = 0
    best_score = 0

    # Structured ordering: by 3-AP weight descending
    order_by_weight = sorted(range(N), key=lambda x: (-elem_3ap_weight[x], x))
    S_greedy = greedy_construct(order_by_weight)
    S_greedy = repair_4aps(S_greedy)
    score_greedy = count_3aps_bitmask(S_greedy)
    if score_greedy > best_score:
        best_score = score_greedy
        S_best = S_greedy

    # Randomized greedy with several seeds
    for seed in range(5):
        if time.time() > TIME_LIMIT - 20:
            break
        rng_local = random.Random(100 + seed)
        order = list(range(N))
        rng_local.shuffle(order)
        S_rand = greedy_construct(order)
        S_rand = repair_4aps(S_rand)
        score_rand = count_3aps_bitmask(S_rand)
        if score_rand > best_score:
            best_score = score_rand
            S_best = S_rand

    # Phase 2: Full 2-opt local search on best solution
    if time.time() < TIME_LIMIT - 15:
        S_ls, score_ls = local_search_2opt(S_best, max_iters=300)
        if score_ls > best_score:
            best_score = score_ls
            S_best = S_ls

    # Phase 3: 2-swap local search for escaping local optima
    if time.time() < TIME_LIMIT - 10:
        S_2swap, score_2swap = local_search_2swap(S_best, max_iters=100)
        if score_2swap > best_score:
            best_score = score_2swap
            S_best = S_2swap

    # Phase 4: WalkSAT for further improvement
    deadline = start_time + TIME_LIMIT - 5
    if time.time() < deadline - 5:
        time_left = deadline - time.time()
        if time_left > 5:
            S_walk, score_walk = walksat(S_best, deadline - 2)
            if score_walk > best_score:
                best_score = score_walk
                S_best = S_walk

    # Final local search pass
    if time.time() < deadline - 2:
        S_final, score_final = local_search_2opt(S_best, max_iters=100)
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
