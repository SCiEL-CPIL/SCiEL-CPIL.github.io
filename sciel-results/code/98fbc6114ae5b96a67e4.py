import time
import random
import numpy as np

def construct():
    start_time = time.time()
    n = 256
    deadline = start_time + 82.0

    # ---------- Precomputation ----------
    # All 3-APs as (a,b,c)
    aps_3 = []
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            aps_3.append((a, a + d, a + 2 * d))
    m3 = len(aps_3)

    # For each element x, list of (ap_idx, pos) where x appears in ap
    ap_index_by_elem = [[] for _ in range(n + 1)]
    for idx, (a, b, c) in enumerate(aps_3):
        ap_index_by_elem[a].append((idx, 0))
        ap_index_by_elem[b].append((idx, 1))
        ap_index_by_elem[c].append((idx, 2))

    # All 4-APs
    four_aps = []
    for d in range(1, (n - 1) // 3 + 1):
        for s in range(1, n - 3 * d + 1):
            four_aps.append((s, s + d, s + 2 * d, s + 3 * d))
    m4 = len(four_aps)

    # For each x, list of (ap_idx, pos) for 4-APs containing x
    four_aps_by_elem = [[] for _ in range(n + 1)]
    for idx, ap in enumerate(four_aps):
        for pos, elem in enumerate(ap):
            four_aps_by_elem[elem].append((idx, pos))

    # Precompute mask4[x] = list of bitmasks of other 3 elements in each 4-AP containing x
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

    # gain_pairs[x] = list of (bit1, bit2) where both must be in S for adding x to complete a 3-AP
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

    def verify_no_4ap(S):
        for ap in four_aps:
            if all((S >> (elem-1)) & 1 for elem in ap):
                return False
        return True

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

    # ---------- 2-swap local search ----------
    def local_search_2swap(S, current_score, max_iters=30, time_budget=None):
        S = int(S)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time_budget is not None and time.time() > time_budget:
                break
            improved = False
            iters += 1
            in_set = [x for x in range(1, n + 1) if (S >> (x-1)) & 1]
            out_set = [x for x in range(1, n + 1) if not ((S >> (x-1)) & 1)]
            if not in_set or not out_set:
                break
            # For each element in S, try to remove and add back (single move)
            for y in in_set[:]:
                if time_budget is not None and time.time() > time_budget - 0.5:
                    break
                S_wo_y = S & ~(1 << (y-1))
                score_wo_y = current_score - loss_remove(y, S)
                best_x = -1
                best_gain = 0
                for x in out_set:
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
                    in_set = [x for x in range(1, n + 1) if (S >> (x-1)) & 1]
                    out_set = [x for x in range(1, n + 1) if not ((S >> (x-1)) & 1)]
                    improved = True

            # 2-swap moves: remove y1, y2; add x1, x2
            if time_budget is not None and time.time() > time_budget - 2.0:
                continue
            in_list = in_set
            out_list = out_set
            # Sample a subset of pairs for 2-swap to keep runtime bounded
            sample_size = min(15, len(in_list))
            random_indices = random.sample(range(len(in_list)), sample_size)
            out_sample = random.sample(range(len(out_list)), min(10, len(out_list)))
            for i in range(sample_size):
                if time_budget is not None and time.time() > time_budget - 1.0:
                    break
                y = in_list[random_indices[i]]
                for j in range(sample_size):
                    if j <= i:
                        continue
                    y2 = in_list[random_indices[j]]
                    if y2 == y:
                        continue
                    S_wo = S & ~(1 << (y-1)) & ~(1 << (y2-1))
                    score_wo = current_score - loss_remove(y, S) - (loss_remove(y2, S) - (1 if loss_remove(y, S) >= loss_remove(y2, S & ~(1 << (y-1))) and False else 0))
                    # More accurate: recompute
                    S_wo = S & ~(1 << (y-1)) & ~(1 << (y2-1))
                    # Recompute score from scratch for correctness
                    score_wo = 0
                    for a, b, c in aps_3:
                        if (S_wo & (1 << (a-1))) and (S_wo & (1 << (b-1))) and (S_wo & (1 << (c-1))):
                            score_wo += 1
                    # Try adding two elements from out_sample
                    best_delta = -10
                    best_pair = None
                    for x in out_sample:
                        if (S_wo >> (x-1)) & 1:
                            continue
                        if not can_add(x, S_wo):
                            continue
                        for x2 in out_sample:
                            if x2 == x:
                                continue
                            if (S_wo >> (x2-1)) & 1:
                                continue
                            S_trial = S_wo | (1 << (x-1)) | (1 << (x2-1))
                            if not can_add(x2, S_wo | (1 << (x-1))):
                                continue
                            # Check both can be added
                            S_tmp = S_wo
                            ok1 = True
                            ok2 = True
                            for m in mask4[x]:
                                if (S_tmp & m) == m:
                                    ok1 = False
                                    break
                            if ok1:
                                S_tmp |= (1 << (x-1))
                            for m in mask4[x2]:
                                if (S_tmp & m) == m:
                                    ok2 = False
                                    break
                            if not (ok1 and ok2):
                                continue
                            S_final = S_wo | (1 << (x-1)) | (1 << (x2-1))
                            if not verify_no_4ap(S_final):
                                continue
                            g_new = 0
                            for a, b, c in aps_3:
                                if (S_final & (1 << (a-1))) and (S_final & (1 << (b-1))) and (S_final & (1 << (c-1))):
                                    g_new += 1
                            delta = g_new - score_wo
                            if delta > best_delta:
                                best_delta = delta
                                best_pair = (x, x2, g_new)
                    if best_pair and best_delta > 0:
                        x, x2, g_new = best_pair
                        S = S_wo | (1 << (x-1)) | (1 << (x2-1))
                        current_score = g_new
                        in_set = [x for x in range(1, n + 1) if (S >> (x-1)) & 1]
                        out_set = [x for x in range(1, n + 1) if not ((S >> (x-1)) & 1)]
                        improved = True

        if not verify_no_4ap(S):
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
        return S, current_score

    # ---------- Simulated Annealing local search ----------
    def sa_local_search(S, current_score, max_iters=400, time_budget=None):
        S = int(S)
        T = 4.0
        T_min = 0.05
        cooling = 0.992
        iters = 0
        best_S = S
        best_sc = current_score
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
            elif move_type < 0.65:
                in_set = [x for x in range(1, n+1) if (S >> (x-1)) & 1]
                if not in_set:
                    continue
                y = random.choice(in_set)
                S_new = S & ~(1 << (y-1))
                delta = -loss_remove(y, S)
            else:
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
            if delta > 0 or random.random() < np.exp(delta / T):
                S = S_new
                current_score += delta
                if current_score > best_sc:
                    best_sc = current_score
                    best_S = S
            T = max(T_min, T * cooling)
        # Polish best
        S2 = int(best_S)
        sc2 = best_sc
        improved = True
        while improved and (time_budget is None or time.time() < time_budget):
            improved = False
            for x in range(1, n + 1):
                if (S2 >> (x-1)) & 1:
                    continue
                if not can_add(x, S2):
                    continue
                g = gain_add(x, S2)
                if g > 0:
                    S2 |= (1 << (x-1))
                    sc2 += g
                    improved = True
        if not verify_no_4ap(S2):
            for ap in four_aps:
                if all((S2 >> (elem-1)) & 1 for elem in ap):
                    min_loss = float('inf')
                    min_elem = ap[0]
                    for elem in ap:
                        l = loss_remove(elem, S2)
                        if l < min_loss:
                            min_loss = l
                            min_elem = elem
                    S2 &= ~(1 << (min_elem - 1))
                    sc2 = score(S2)
        return S2, sc2

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
        for _ in range(300):
            improved = False
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
                    improved = True
            if not improved:
                break
        return S

    # ---------- Main search ----------
    best_S = 0
    best_score = 0
    random.seed(12345)

    # Strategy 1: Natural order greedy
    if time.time() < deadline - 20:
        S1 = greedy_random(range(1, n + 1), use_score=False)
        sc1 = score(S1)
        S1, sc1 = local_search_2swap(S1, sc1, max_iters=20, time_budget=deadline - 15)
        if sc1 > best_score:
            best_score = sc1
            best_S = S1

    # Strategy 2: Random restarts with local search
    n_restarts = 40
    for r in range(n_restarts):
        if time.time() > deadline - 18:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_r = greedy_random(order, use_score=True, min_size=12)
        sc_r = score(S_r)
        S_r, sc_r = local_search_2swap(S_r, sc_r, max_iters=12, time_budget=deadline - 14)
        if sc_r > best_score:
            best_score = sc_r
            best_S = S_r

    # Strategy 3: Prune from full
    if time.time() < deadline - 16:
        S_p = prune_from_full()
        sc_p = score(S_p)
        S_p, sc_p = local_search_2swap(S_p, sc_p, max_iters=15, time_budget=deadline - 12)
        if sc_p > best_score:
            best_score = sc_p
            best_S = S_p

    # Strategy 4: SA from best
    if time.time() < deadline - 12 and best_S:
        S_sa = int(best_S)
        sc_sa = best_score
        S_sa, sc_sa = sa_local_search(S_sa, sc_sa, max_iters=500, time_budget=deadline - 8)
        if sc_sa > best_score:
            best_score = sc_sa
            best_S = S_sa

    # Strategy 5: SA from random start
    if time.time() < deadline - 12:
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_r = greedy_random(order, use_score=True, min_size=12)
        sc_r = score(S_r)
        S_r, sc_r = sa_local_search(S_r, sc_r, max_iters=400, time_budget=deadline - 6)
        if sc_r > best_score:
            best_score = sc_r
            best_S = S_r

    # Strategy 6: Final 2-swap polish
    if time.time() < deadline - 3 and best_S:
        best_S, best_score = local_search_2swap(best_S, best_score, max_iters=25, time_budget=deadline - 1)

    # Final verification
    if best_S and not verify_no_4ap(best_S):
        for ap in four_aps:
            if all((best_S >> (elem-1)) & 1 for elem in ap):
                min_loss = float('inf')
                min_elem = ap[0]
                for elem in ap:
                    l = loss_remove(elem, best_S)
                    if l < min_loss:
                        min_loss = l
                        min_elem = elem
                best_S &= ~(1 << (min_elem - 1))
        best_score = score(best_S)

    result = [x for x in range(1, n + 1) if (best_S >> (x-1)) & 1]
    if not result:
        result = [1]
    return result
