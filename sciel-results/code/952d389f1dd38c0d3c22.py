import time
import random
import numpy as np

def construct():
    start_time = time.time()
    n = 256
    deadline = start_time + 85.0

    # ---------- Precomputation ----------
    aps_3 = []
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            aps_3.append((a, a + d, a + 2 * d))
    m3 = len(aps_3)

    ap_index_by_elem = [[] for _ in range(n + 1)]
    for idx, (a, b, c) in enumerate(aps_3):
        ap_index_by_elem[a].append((idx, 0))
        ap_index_by_elem[b].append((idx, 1))
        ap_index_by_elem[c].append((idx, 2))

    four_aps = []
    for d in range(1, (n - 1) // 3 + 1):
        for s in range(1, n - 3 * d + 1):
            four_aps.append((s, s + d, s + 2 * d, s + 3 * d))
    m4 = len(four_aps)

    four_aps_by_elem = [[] for _ in range(n + 1)]
    for idx, ap in enumerate(four_aps):
        for pos, elem in enumerate(ap):
            four_aps_by_elem[elem].append((idx, pos))

    mask4 = [None] * (n + 1)
    for x in range(1, n + 1):
        lst = four_aps_by_elem[x]
        if lst:
            mask4[x] = []
            for ap_idx, pos in lst:
                ap = four_aps[ap_idx]
                m = 0
                for j in range(4):
                    if j != pos:
                        m |= (1 << (ap[j] - 1))
                mask4[x].append(m)
        else:
            mask4[x] = []

    gain_pairs = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        pairs = []
        for idx, pos in ap_index_by_elem[x]:
            a, b, c = aps_3[idx]
            if pos == 0:
                pairs.append((1 << (b-1), 1 << (c-1)))
            elif pos == 1:
                pairs.append((1 << (a-1), 1 << (c-1)))
            else:
                pairs.append((1 << (a-1), 1 << (b-1)))
        gain_pairs[x] = pairs

    # ---------- Core operations ----------
    def can_add(x, S):
        masks = mask4[x]
        for m in masks:
            if (S & m) == m:
                return False
        return True

    def gain_add(x, S):
        g = 0
        for bit1, bit2 in gain_pairs[x]:
            if (S & bit1) and (S & bit2):
                g += 1
        return g

    def loss_remove(x, S):
        S_wo_x = S & ~(1 << (x-1))
        g = 0
        for bit1, bit2 in gain_pairs[x]:
            if (S_wo_x & bit1) and (S_wo_x & bit2):
                g += 1
        return g

    def score(S):
        cnt = 0
        for a, b, c in aps_3:
            if (S & (1 << (a-1))) and (S & (1 << (b-1))) and (S & (1 << (c-1))):
                cnt += 1
        return cnt

    # ---------- Improved local search with 2-swap moves ----------
    def local_search_v2(S, current_score, max_iters=80, time_budget=None):
        S = int(S)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time_budget is not None and time.time() > time_budget:
                break
            improved = False
            iters += 1

            # Phase 1: Greedily add elements with positive gain
            for x in range(1, n + 1):
                if (S >> (x-1)) & 1:
                    continue
                if not can_add(x, S):
                    continue
                g = gain_add(x, S)
                if g > 0:
                    S |= (1 << (x-1))
                    current_score += g
                    improved = True
                    if time_budget is not None and time.time() > time_budget:
                        break

            if time_budget is not None and time.time() > time_budget:
                break

            # Phase 2: 1-swap moves (remove y, add best x)
            for y in range(1, n + 1):
                if time_budget is not None and time.time() > time_budget - 0.3:
                    break
                if not ((S >> (y-1)) & 1):
                    continue
                S_wo_y = S & ~(1 << (y-1))
                score_wo_y = current_score - loss_remove(y, S)
                best_x = -1
                best_gain = 0
                for x in range(1, n + 1):
                    if (S_wo_y >> (x-1)) & 1:
                        continue
                    if not can_add(x, S_wo_y):
                        continue
                    g = gain_add(x, S_wo_y)
                    if g > best_gain:
                        best_gain = g
                        best_x = x
                if best_x > 0 and score_wo_y + best_gain > current_score:
                    S = S_wo_y | (1 << (best_x - 1))
                    current_score = score_wo_y + best_gain
                    improved = True
                    if time_budget is not None and time.time() > time_budget:
                        break

            if time_budget is not None and time.time() > time_budget:
                break

            # Phase 3: 2-swap moves (remove y1, y2; add x1, x2)
            in_set = [x for x in range(1, n + 1) if (S >> (x-1)) & 1]
            for i in range(min(len(in_set), 40)):
                if time_budget is not None and time.time() > time_budget - 1.0:
                    break
                y1 = in_set[i]
                for j in range(i + 1, min(len(in_set), 60)):
                    y2 = in_set[j]
                    S_wo = S & ~(1 << (y1-1)) & ~(1 << (y2-1))
                    score_wo = current_score - loss_remove(y1, S) - loss_remove(y2, S & ~(1 << (y1-1)))
                    # Greedily add two best elements
                    candidates = []
                    for x in range(1, n + 1):
                        if (S_wo >> (x-1)) & 1:
                            continue
                        if not can_add(x, S_wo):
                            continue
                        g = gain_add(x, S_wo)
                        if g > 0:
                            candidates.append((g, x))
                    candidates.sort(reverse=True)
                    best_gain_total = 0
                    added = []
                    for g, x in candidates[:3]:
                        S_trial = S_wo
                        for ax in added:
                            S_trial |= (1 << (ax-1))
                        if can_add(x, S_trial):
                            S_trial |= (1 << (x-1))
                            best_gain_total += g
                            added.append(x)
                        if len(added) == 2:
                            break
                    if score_wo + best_gain_total > current_score:
                        S = S_wo
                        for ax in added:
                            S |= (1 << (ax-1))
                        current_score = score_wo + best_gain_total
                        improved = True
                        break
                if improved:
                    break

        return S, current_score

    # ---------- Simulated Annealing with 2-swap moves ----------
    def sa_local_search_v2(S, current_score, max_iters=300, time_budget=None):
        S = int(S)
        T = 4.0
        T_min = 0.05
        cooling = 0.985
        iters = 0
        while iters < max_iters:
            if time_budget is not None and time.time() > time_budget:
                break
            iters += 1

            move_type = random.random()
            if move_type < 0.35:
                candidates = [x for x in range(1, n+1) if not ((S >> (x-1)) & 1) and can_add(x, S)]
                if not candidates:
                    continue
                x = random.choice(candidates)
                S_new = S | (1 << (x-1))
                delta = gain_add(x, S)
            elif move_type < 0.60:
                in_set = [x for x in range(1, n+1) if (S >> (x-1)) & 1]
                if not in_set:
                    continue
                y = random.choice(in_set)
                S_new = S & ~(1 << (y-1))
                delta = -loss_remove(y, S)
            elif move_type < 0.85:
                in_set = [x for x in range(1, n+1) if (S >> (x-1)) & 1]
                not_in_set = [x for x in range(1, n+1) if not ((S >> (x-1)) & 1)]
                if not in_set or not not_in_set:
                    continue
                y = random.choice(in_set)
                S_wo_y = S & ~(1 << (y-1))
                valid_adds = [x for x in not_in_set if can_add(x, S_wo_y)]
                if not valid_adds:
                    continue
                x = random.choice(valid_adds)
                S_new = S_wo_y | (1 << (x-1))
                delta = gain_add(x, S_wo_y) - loss_remove(y, S)
            else:
                # 2-swap
                in_set = [x for x in range(1, n+1) if (S >> (x-1)) & 1]
                not_in_set = [x for x in range(1, n+1) if not ((S >> (x-1)) & 1)]
                if len(in_set) < 2 or len(not_in_set) < 2:
                    continue
                y1, y2 = random.sample(in_set, 2)
                S_wo = S & ~(1 << (y1-1)) & ~(1 << (y2-1))
                valid_adds = [x for x in not_in_set if can_add(x, S_wo)]
                if len(valid_adds) < 2:
                    continue
                x1, x2 = random.sample(valid_adds, 2)
                S_new = S_wo | (1 << (x1-1)) | (1 << (x2-1))
                delta = gain_add(x1, S_wo) + gain_add(x2, S_wo | (1 << (x1-1))) - loss_remove(y1, S) - loss_remove(y2, S & ~(1 << (y1-1)))

            if delta > 0 or random.random() < np.exp(delta / T):
                S = S_new
                current_score += delta

            T = max(T_min, T * cooling)

        # Final greedy polish
        improved = True
        while improved:
            if time_budget is not None and time.time() > time_budget:
                break
            improved = False
            for x in range(1, n + 1):
                if (S >> (x-1)) & 1:
                    continue
                if not can_add(x, S):
                    continue
                g = gain_add(x, S)
                if g > 0:
                    S |= (1 << (x-1))
                    current_score += g
                    improved = True
        return S, current_score

    # ---------- Construction strategies ----------
    def greedy_random(order, use_score=True, min_size=0):
        S = 0
        for x in order:
            if not can_add(x, S):
                continue
            if use_score:
                g = gain_add(x, S)
                if g > 0 or bin(S).count('1') < min_size:
                    S |= (1 << (x-1))
            else:
                S |= (1 << (x-1))
        return S

    def prune_from_full():
        S = (1 << n) - 1
        for ap in four_aps:
            if all((S >> (elem-1)) & 1 for elem in ap):
                min_loss = float('inf')
                min_elem = ap[0]
                for elem in ap:
                    l = loss_remove(elem, S)
                    if l < min_loss:
                        min_loss = l
                        min_elem = elem
                S &= ~(1 << (min_elem - 1))
        return S

    def greedy_score_order():
        """Build set by repeatedly adding the element with maximum gain."""
        S = 0
        remaining = list(range(1, n + 1))
        for _ in range(n):
            if not remaining:
                break
            best_x = None
            best_g = -1
            for x in remaining:
                if not can_add(x, S):
                    continue
                g = gain_add(x, S)
                if g > best_g:
                    best_g = g
                    best_x = x
            if best_x is None or best_g < 0:
                # No positive gain, break
                break
            S |= (1 << (best_x - 1))
            remaining.remove(best_x)
        return S

    # ---------- Main search ----------
    best_S = 0
    best_score = 0

    random.seed(12345)

    # Strategy 1: Score-ordered greedy
    if time.time() < deadline - 40:
        S1 = greedy_score_order()
        sc1 = score(S1)
        S1, sc1 = local_search_v2(S1, sc1, max_iters=50, time_budget=deadline - 35)
        if sc1 > best_score:
            best_score = sc1
            best_S = S1

    # Strategy 2: Prune from full
    if time.time() < deadline - 35:
        S_p = prune_from_full()
        sc_p = score(S_p)
        S_p, sc_p = local_search_v2(S_p, sc_p, max_iters=50, time_budget=deadline - 30)
        if sc_p > best_score:
            best_score = sc_p
            best_S = S_p

    # Strategy 3: Many random restarts with improved local search
    n_restarts = 40
    for r in range(n_restarts):
        if time.time() > deadline - 25:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_r = greedy_random(order, use_score=True, min_size=10)
        sc_r = score(S_r)
        S_r, sc_r = local_search_v2(S_r, sc_r, max_iters=30, time_budget=deadline - 20)
        if sc_r > best_score:
            best_score = sc_r
            best_S = S_r

    # Strategy 4: SA from best
    if time.time() < deadline - 20 and best_S:
        S_sa = int(best_S)
        sc_sa = best_score
        S_sa, sc_sa = sa_local_search_v2(S_sa, sc_sa, max_iters=400, time_budget=deadline - 12)
        if sc_sa > best_score:
            best_score = sc_sa
            best_S = S_sa

    # Strategy 5: SA from random starts
    for r in range(10):
        if time.time() > deadline - 12:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_r = greedy_random(order, use_score=True, min_size=10)
        sc_r = score(S_r)
        S_r, sc_r = sa_local_search_v2(S_r, sc_r, max_iters=250, time_budget=deadline - 8)
        if sc_r > best_score:
            best_score = sc_r
            best_S = S_r

    # Strategy 6: More random restarts with local search if time permits
    for r in range(20):
        if time.time() > deadline - 8:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_r = greedy_random(order, use_score=True, min_size=5)
        sc_r = score(S_r)
        S_r, sc_r = local_search_v2(S_r, sc_r, max_iters=25, time_budget=deadline - 5)
        if sc_r > best_score:
            best_score = sc_r
            best_S = S_r

    # Final local search on best
    if time.time() < deadline - 3 and best_S:
        best_S, best_score = local_search_v2(best_S, best_score, max_iters=60, time_budget=deadline - 1)

    # Final SA polish
    if time.time() < deadline - 2 and best_S:
        best_S, best_score = sa_local_search_v2(best_S, best_score, max_iters=150, time_budget=deadline - 0.5)

    # Convert bitmask to sorted list
    result = [x for x in range(1, n + 1) if (best_S >> (x-1)) & 1]
    if not result:
        result = [1]
    return result
