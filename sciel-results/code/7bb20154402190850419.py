"""Combined 4-AP-free set construction: incremental-score greedy + WalkSAT-style stochastic local search."""
import time
import random
import math

N = 256
TIME_LIMIT = 84.0  # safety margin below 90s


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

    # Precompute 3-AP weights (total number of 3-APs each element is in)
    elem_3ap_weight = [len(elem_to_3aps[x]) for x in range(N)]
    # Precompute 4-AP weights (total number of 4-APs each element is in)
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

    def local_search_2opt(S, max_iters=300):
        """2-opt local search with incremental scoring."""
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

            # Try swaps
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

            break

        return current_S, current_score

    def walksat_search(S, max_iters=2000):
        """WalkSAT-style stochastic local search for 4-AP-free set maximizing 3-APs.
        
        Adapted from MaxSAT: 
        - Hard clauses: each 4-AP must NOT be fully satisfied (forbidden)
        - Soft clauses: each 3-AP should be satisfied (rewarded)
        
        Uses a hybrid score: -penalty * (# violated 4-APs) + (# satisfied 3-APs)
        Flips variables (toggles elements) to minimize the combined objective.
        """
        current_S = S
        current_3aps = count_3aps_bitmask(current_S)
        current_4aps = count_4aps_bitmask(current_S)
        penalty = 100.0  # Heavy penalty for 4-AP violations
        
        best_S = current_S
        best_3aps = current_3aps
        best_4aps = current_4aps
        best_obj = -penalty * best_4aps + best_3aps

        for it in range(max_iters):
            if time.time() - start_time > TIME_LIMIT - 3:
                break

            # If we have a valid solution, track it
            if current_4aps == 0:
                if current_3aps > best_3aps:
                    best_S = current_S
                    best_3aps = current_3aps
                    best_4aps = 0
                    best_obj = current_3aps

            # Compute current objective
            current_obj = -penalty * current_4aps + current_3aps

            # Choose a move: either fix a 4-AP violation or improve 3-AP count
            if current_4aps > 0:
                # Pick a random violated 4-AP and try to break it
                # Find all violated 4-APs
                violated_aps = []
                for a, b, c, d in four_aps:
                    if (current_S >> a) & 1 and (current_S >> b) & 1 and (current_S >> c) & 1 and (current_S >> d) & 1:
                        violated_aps.append((a, b, c, d))
                
                if not violated_aps:
                    continue
                    
                ap = rng.choice(violated_aps)
                
                # Try removing one element from this 4-AP
                best_move = None
                best_move_score = float('inf')
                for x in ap:
                    S_new = current_S & ~(1 << x)
                    new_3aps = current_3aps - loss_3aps(current_S, x)
                    new_4aps = count_4aps_bitmask(S_new)
                    new_obj = -penalty * new_4aps + new_3aps
                    if new_obj < best_move_score:
                        best_move_score = new_obj
                        best_move = ('remove', x, new_3aps, new_4aps)
                
                if best_move:
                    _, x, new_3aps, new_4aps = best_move
                    current_S &= ~(1 << x)
                    current_3aps = new_3aps
                    current_4aps = new_4aps
            else:
                # No 4-AP violations: try to improve 3-AP count
                # With probability 0.7, do a greedy improvement; with 0.3, do a random perturbation
                if rng.random() < 0.7:
                    # Greedy: try adds and swaps
                    improved = False
                    members = [x for x in range(N) if (current_S >> x) & 1]
                    non_members = [x for x in range(N) if not ((current_S >> x) & 1)]
                    
                    # Try single adds
                    for x in non_members:
                        if not is_safe_add(current_S, x):
                            continue
                        new_score = current_3aps + gain_3aps(current_S, x)
                        if new_score > current_3aps:
                            current_S |= (1 << x)
                            current_3aps = new_score
                            improved = True
                            break
                    
                    if improved:
                        continue
                    
                    # Try swaps
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
                        current_3aps += best_gain
                        continue
                    
                    # No improvement; perturb
                # Random perturbation: toggle 1-3 random elements
                num_flips = rng.randint(1, 3)
                for _ in range(num_flips):
                    x = rng.randint(0, N - 1)
                    if (current_S >> x) & 1:
                        # Remove x
                        current_S &= ~(1 << x)
                        current_3aps -= loss_3aps(current_S | (1 << x), x)
                    else:
                        # Add x if safe
                        if is_safe_add(current_S, x):
                            current_S |= (1 << x)
                            current_3aps += gain_3aps(current_S & ~(1 << x), x)
                    current_4aps = count_4aps_bitmask(current_S)

        # Return best valid solution found
        if best_4aps == 0:
            return best_S, best_3aps
        else:
            # If no valid solution, repair the current one
            repaired = repair_4aps(current_S)
            score = count_3aps_bitmask(repaired)
            if score > best_3aps:
                return repaired, score
            return best_S, best_3aps

    def local_search_2opt_heavy(S, max_iters=150):
        """Enhanced 2-opt with 2-swap moves."""
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

            # Try swaps
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

            # Try 2-swap: remove two, add two
            if len(members) >= 2 and len(non_members) >= 2:
                sample_members = rng.sample(members, min(len(members), 25))
                sample_non_members = rng.sample(non_members, min(len(non_members), 25))

                best_2swap = None
                best_2swap_gain = 0
                for i, x1 in enumerate(sample_members):
                    S_after1 = current_S & ~(1 << x1)
                    loss1 = loss_3aps(current_S, x1)
                    for x2 in sample_members[i+1:]:
                        S_after2 = S_after1 & ~(1 << x2)
                        loss2 = loss_3aps(S_after1, x2)
                        total_loss = loss1 + loss2

                        for y1 in sample_non_members:
                            if not is_safe_add(S_after2, y1):
                                continue
                            S_after2_y1 = S_after2 | (1 << y1)
                            gain1 = gain_3aps(S_after2, y1)

                            for y2 in sample_non_members:
                                if y2 == y1:
                                    continue
                                if not is_safe_add(S_after2_y1, y2):
                                    continue
                                gain2 = gain_3aps(S_after2_y1, y2)
                                total_gain = gain1 + gain2
                                net = total_gain - total_loss
                                if net > best_2swap_gain:
                                    best_2swap_gain = net
                                    best_2swap = (x1, x2, y1, y2)

                if best_2swap is not None and best_2swap_gain > 0:
                    x1, x2, y1, y2 = best_2swap
                    current_S = (current_S & ~(1 << x1) & ~(1 << x2)) | (1 << y1) | (1 << y2)
                    current_score += best_2swap_gain
                    improved = True
                    continue

            if not improved:
                break

        return current_S, current_score

    # === PHASE 1: Multi-start greedy with structured orderings ===
    best_S = 0
    best_score = -1

    # Ordering 1: by 3-AP weight descending
    order1 = sorted(range(N), key=lambda x: (-elem_3ap_weight[x], x))
    S1 = greedy_construct(order1)
    S1 = repair_4aps(S1)
    score1 = count_3aps_bitmask(S1)
    if score1 > best_score:
        best_score = score1
        best_S = S1

    # Ordering 2: by 3-AP weight descending, break ties by 4-AP weight ascending
    order2 = sorted(range(N), key=lambda x: (-elem_3ap_weight[x], elem_4ap_weight[x], x))
    S2 = greedy_construct(order2)
    S2 = repair_4aps(S2)
    score2 = count_3aps_bitmask(S2)
    if score2 > best_score:
        best_score = score2
        best_S = S2

    # Ordering 3: residue class mod 3
    order3 = []
    for r in range(3):
        for x in range(r, N, 3):
            order3.append(x)
    S3 = greedy_construct(order3)
    S3 = repair_4aps(S3)
    score3 = count_3aps_bitmask(S3)
    if score3 > best_score:
        best_score = score3
        best_S = S3

    # Ordering 4: residue class mod 4
    order4 = []
    for r in range(4):
        for x in range(r, N, 4):
            order4.append(x)
    S4 = greedy_construct(order4)
    S4 = repair_4aps(S4)
    score4 = count_3aps_bitmask(S4)
    if score4 > best_score:
        best_score = score4
        best_S = S4

    # Ordering 5: random shuffle
    order5 = list(range(N))
    rng.shuffle(order5)
    S5 = greedy_construct(order5)
    S5 = repair_4aps(S5)
    score5 = count_3aps_bitmask(S5)
    if score5 > best_score:
        best_score = score5
        best_S = S5

    # === PHASE 2: 2-opt local search on best ===
    if time.time() - start_time < TIME_LIMIT - 15:
        S_ls, score_ls = local_search_2opt(best_S, max_iters=300)
        if score_ls > best_score:
            best_score = score_ls
            best_S = S_ls

    # === PHASE 3: WalkSAT-style stochastic search ===
    if time.time() - start_time < TIME_LIMIT - 10:
        S_ws, score_ws = walksat_search(best_S, max_iters=2000)
        if score_ws > best_score:
            best_score = score_ws
            best_S = S_ws

    # === PHASE 4: Heavy local search (2-swap moves) ===
    if time.time() - start_time < TIME_LIMIT - 5:
        S_hs, score_hs = local_search_2opt_heavy(best_S, max_iters=150)
        if score_hs > best_score:
            best_score = score_hs
            best_S = S_hs

    # === PHASE 5: More random restarts with WalkSAT ===
    deadline = start_time + TIME_LIMIT - 3
    for i in range(20):
        if time.time() > deadline:
            break

        order = list(range(N))
        rng.shuffle(order)
        S_init = greedy_construct(order)
        S_init = repair_4aps(S_init)
        score_init = count_3aps_bitmask(S_init)

        # Quick 2-opt
        S_ls2, score_ls2 = local_search_2opt(S_init, max_iters=50)
        if score_ls2 > best_score:
            best_score = score_ls2
            best_S = S_ls2

        # WalkSAT on the improved solution
        if time.time() < deadline - 2:
            S_ws2, score_ws2 = walksat_search(S_ls2, max_iters=500)
            if score_ws2 > best_score:
                best_score = score_ws2
                best_S = S_ws2

    # === PHASE 6: Final polish ===
    if time.time() - start_time < deadline - 1:
        S_final, score_final = local_search_2opt_heavy(best_S, max_iters=100)
        if score_final > best_score:
            best_score = score_final
            best_S = S_final

    # Verify no 4-APs
    assert count_4aps_bitmask(best_S) == 0, "4-AP violation found!"

    # Convert bitmask to sorted list (1-indexed)
    result = [x + 1 for x in range(N) if (best_S >> x) & 1]

    if not result:
        S_fb = 0
        for x in range(N):
            if is_safe_add(S_fb, x):
                S_fb |= (1 << x)
        result = [x + 1 for x in range(N) if (S_fb >> x) & 1]

    return result
