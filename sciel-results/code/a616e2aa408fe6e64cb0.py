import time
import random
import numpy as np

def construct():
    start_time = time.time()
    n = 256
    deadline = start_time + 85.0  # safety margin before 90s kill

    # ---------- Precomputation ----------
    # All 3-APs
    aps_3 = []
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            aps_3.append((a, a + d, a + 2 * d))
    m3 = len(aps_3)

    # For each element, list of 3-AP indices it appears in
    ap_index_by_elem = [[] for _ in range(n + 1)]
    for idx, (a, b, c) in enumerate(aps_3):
        ap_index_by_elem[a].append(idx)
        ap_index_by_elem[b].append(idx)
        ap_index_by_elem[c].append(idx)

    # All 4-APs
    four_aps = []
    for d in range(1, (n - 1) // 3 + 1):
        for s in range(1, n - 3 * d + 1):
            four_aps.append((s, s + d, s + 2 * d, s + 3 * d))
    m4 = len(four_aps)

    # For each element, list of 4-AP indices it appears in
    four_ap_by_elem = [[] for _ in range(n + 1)]
    for idx, ap in enumerate(four_aps):
        for e in ap:
            four_ap_by_elem[e].append(idx)

    # For each 4-AP, the other 3 elements as a bitmask
    four_ap_others = []
    for ap in four_aps:
        for i in range(4):
            x = ap[i]
            others = 0
            for j in range(4):
                if j != i:
                    others |= (1 << (ap[j] - 1))
            four_ap_others.append((x, others))

    # mask4[x]: list of bitmasks of OTHER 3 elements for each 4-AP containing x
    mask4 = [[] for _ in range(n + 1)]
    for x, others in four_ap_others:
        mask4[x].append(others)

    # gain_pairs[x]: for each 3-AP containing x, the pair of other elements as bitmasks
    gain_pairs = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        for idx in ap_index_by_elem[x]:
            a, b, c = aps_3[idx]
            others = [y for y in (a, b, c) if y != x]
            gain_pairs[x].append((1 << (others[0] - 1), 1 << (others[1] - 1)))

    # ---------- Core operations ----------
    def can_add(x, S):
        for m in mask4[x]:
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
        S_wo = S & ~(1 << (x - 1))
        g = 0
        for bit1, bit2 in gain_pairs[x]:
            if (S_wo & bit1) and (S_wo & bit2):
                g += 1
        return g

    def score_full(S):
        cnt = 0
        for a, b, c in aps_3:
            if (S >> (a - 1)) & 1 and (S >> (b - 1)) & 1 and (S >> (c - 1)) & 1:
                cnt += 1
        return cnt

    def is_valid(S):
        for ap in four_aps:
            if all((S >> (e - 1)) & 1 for e in ap):
                return False
        return True

    # ---------- Local search: greedy add, swap, remove ----------
    def local_search(S, sc, max_rounds=50, time_budget=None):
        S = int(S)
        for rnd in range(max_rounds):
            if time_budget is not None and time.time() > time_budget:
                break
            improved = False

            # Greedy add pass
            for x in range(1, n + 1):
                if time_budget is not None and time.time() > time_budget:
                    break
                if (S >> (x - 1)) & 1:
                    continue
                if not can_add(x, S):
                    continue
                g = gain_add(x, S)
                if g > 0:
                    S |= (1 << (x - 1))
                    sc += g
                    improved = True

            if not improved:
                break

            # Try 1-swap moves
            for y in range(1, n + 1):
                if time_budget is not None and time.time() > time_budget:
                    break
                if not ((S >> (y - 1)) & 1):
                    continue
                S_wo = S & ~(1 << (y - 1))
                sc_wo = sc - loss_remove(y, S)

                best_x = -1
                best_g = 0
                for x in range(1, n + 1):
                    if (S_wo >> (x - 1)) & 1:
                        continue
                    if not can_add(x, S_wo):
                        continue
                    g = gain_add(x, S_wo)
                    if g > best_g:
                        best_g = g
                        best_x = x
                if best_x > 0 and sc_wo + best_g > sc:
                    S = S_wo | (1 << (best_x - 1))
                    sc = sc_wo + best_g
                    improved = True

        return S, sc

    # ---------- Tabu Search ----------
    def tabu_search(S, sc, max_iters=500, tabu_len=30, time_budget=None):
        S = int(S)
        tabu = [0] * (n + 1)  # iteration at which element becomes untabu
        tabu_iter = 0
        best_S = S
        best_sc = sc

        for it in range(max_iters):
            if time_budget is not None and time.time() > time_budget:
                break
            tabu_iter += 1

            best_move = None
            best_delta = 0

            # Try all add moves
            for x in range(1, n + 1):
                if (S >> (x - 1)) & 1:
                    continue
                if not can_add(x, S):
                    continue
                if tabu[x] > tabu_iter:
                    continue
                delta = gain_add(x, S)
                if delta > best_delta:
                    best_delta = delta
                    best_move = ('add', x, delta)

            # Try all remove moves
            for x in range(1, n + 1):
                if not ((S >> (x - 1)) & 1):
                    continue
                if tabu[x] > tabu_iter:
                    continue
                delta = -loss_remove(x, S)
                if delta > best_delta:
                    best_delta = delta
                    best_move = ('remove', x, delta)

            # Try all swap moves (remove y, add x)
            for y in range(1, n + 1):
                if not ((S >> (y - 1)) & 1):
                    continue
                if tabu[y] > tabu_iter:
                    continue
                S_wo = S & ~(1 << (y - 1))
                loss_y = loss_remove(y, S)
                for x in range(1, n + 1):
                    if (S_wo >> (x - 1)) & 1:
                        continue
                    if not can_add(x, S_wo):
                        continue
                    if tabu[x] > tabu_iter:
                        continue
                    gain_x = gain_add(x, S_wo)
                    delta = gain_x - loss_y
                    if delta > best_delta:
                        best_delta = delta
                        best_move = ('swap', x, y, delta)

            if best_move is None:
                break

            if best_move[0] == 'add':
                x = best_move[1]
                S |= (1 << (x - 1))
                tabu[x] = tabu_iter + tabu_len
            elif best_move[0] == 'remove':
                x = best_move[1]
                S &= ~(1 << (x - 1))
                tabu[x] = tabu_iter + tabu_len
            else:  # swap
                x, y = best_move[1], best_move[2]
                S &= ~(1 << (y - 1))
                S |= (1 << (x - 1))
                tabu[x] = tabu_iter + tabu_len
                tabu[y] = tabu_iter + tabu_len

            sc += best_move[-1]

            if sc > best_sc:
                best_sc = sc
                best_S = S

        # Final greedy polish
        for x in range(1, n + 1):
            if time_budget is not None and time.time() > time_budget:
                break
            if (S >> (x - 1)) & 1:
                continue
            if not can_add(x, S):
                continue
            g = gain_add(x, S)
            if g > 0:
                S |= (1 << (x - 1))
                sc += g

        return best_S, best_sc

    # ---------- SA local search ----------
    def sa_search(S, sc, max_iters=300, time_budget=None):
        S = int(S)
        T = 4.0
        T_min = 0.1
        cool = 0.98
        for _ in range(max_iters):
            if time_budget is not None and time.time() > time_budget:
                break
            r = random.random()
            if r < 0.40:
                # add
                cands = [x for x in range(1, n + 1) if not ((S >> (x - 1)) & 1) and can_add(x, S)]
                if not cands:
                    continue
                x = random.choice(cands)
                S_new = S | (1 << (x - 1))
                delta = gain_add(x, S)
            elif r < 0.65:
                # remove
                in_s = [x for x in range(1, n + 1) if (S >> (x - 1)) & 1]
                if not in_s:
                    continue
                y = random.choice(in_s)
                S_new = S & ~(1 << (y - 1))
                delta = -loss_remove(y, S)
            elif r < 0.85:
                # swap
                in_s = [x for x in range(1, n + 1) if (S >> (x - 1)) & 1]
                not_s = [x for x in range(1, n + 1) if not ((S >> (x - 1)) & 1)]
                if not in_s or not not_s:
                    continue
                y = random.choice(in_s)
                S_wo = S & ~(1 << (y - 1))
                valid = [x for x in not_s if can_add(x, S_wo)]
                if not valid:
                    continue
                x = random.choice(valid)
                S_new = S_wo | (1 << (x - 1))
                delta = gain_add(x, S_wo) - loss_remove(y, S)
            else:
                # 2-swap: remove 2, add 2
                in_s = [x for x in range(1, n + 1) if (S >> (x - 1)) & 1]
                not_s = [x for x in range(1, n + 1) if not ((S >> (x - 1)) & 1)]
                if len(in_s) < 2 or len(not_s) < 2:
                    continue
                y1, y2 = random.sample(in_s, 2)
                S_wo = S & ~(1 << (y1 - 1)) & ~(1 << (y2 - 1))
                valid = [x for x in not_s if can_add(x, S_wo)]
                if len(valid) < 2:
                    continue
                x1, x2 = random.sample(valid, 2)
                S_new = S_wo | (1 << (x1 - 1)) | (1 << (x2 - 1))
                delta = gain_add(x1, S_wo) + gain_add(x2, S_wo) - loss_remove(y1, S) - loss_remove(y2, S)

            if delta > 0 or random.random() < np.exp(delta / T):
                S = S_new
                sc += delta
            T = max(T_min, T * cool)

        # Final greedy polish
        for x in range(1, n + 1):
            if time_budget is not None and time.time() > time_budget:
                break
            if (S >> (x - 1)) & 1:
                continue
            if not can_add(x, S):
                continue
            g = gain_add(x, S)
            if g > 0:
                S |= (1 << (x - 1))
                sc += g
        return S, sc

    # ---------- Construction: greedy with random order ----------
    def greedy_build(order, min_g=0):
        S = 0
        for x in order:
            if not can_add(x, S):
                continue
            g = gain_add(x, S)
            if g >= min_g:
                S |= (1 << (x - 1))
        return S

    # ---------- Prune from full set ----------
    def prune_full():
        S = (1 << n) - 1
        for ap in four_aps:
            if all((S >> (e - 1)) & 1 for e in ap):
                best_loss = float('inf')
                best_elem = ap[0]
                for e in ap:
                    l = loss_remove(e, S)
                    if l < best_loss:
                        best_loss = l
                        best_elem = e
                S &= ~(1 << (best_elem - 1))
        return S

    # ---------- Random initial set (density-controlled) ----------
    def random_density(p):
        S = 0
        for x in range(1, n + 1):
            if random.random() < p:
                S |= (1 << (x - 1))
        return S

    def repair(S):
        # Remove elements to break 4-APs
        for ap in four_aps:
            if all((S >> (e - 1)) & 1 for e in ap):
                # remove element with lowest loss
                best_loss = float('inf')
                best_e = ap[0]
                for e in ap:
                    l = loss_remove(e, S)
                    if l < best_loss:
                        best_loss = l
                        best_e = e
                S &= ~(1 << (best_e - 1))
        return S

    # ---------- Main search ----------
    random.seed(12345)
    best_S = 0
    best_score = 0

    def update_best(S, sc):
        nonlocal best_S, best_score
        if sc > best_score:
            best_S = S
            best_score = sc
            return True
        return False

    # Strategy A: Prune from full + local search
    if time.time() < deadline - 30:
        S = prune_full()
        sc = score_full(S)
        S, sc = local_search(S, sc, max_rounds=40, time_budget=deadline - 25)
        update_best(S, sc)

    # Strategy B: Random restarts with greedy build + local search
    n_restarts = 80
    for r in range(n_restarts):
        if time.time() > deadline - 20:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S = greedy_build(order, min_g=0)
        sc = score_full(S)
        S, sc = local_search(S, sc, max_rounds=20, time_budget=deadline - 15)
        update_best(S, sc)

    # Strategy C: Random density + repair + local search
    for p in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        if time.time() > deadline - 20:
            break
        S = random_density(p)
        S = repair(S)
        sc = score_full(S)
        S, sc = local_search(S, sc, max_rounds=20, time_budget=deadline - 15)
        update_best(S, sc)

    # Strategy D: SA from best
    if time.time() < deadline - 15 and best_S:
        S = int(best_S)
        sc = best_score
        S, sc = sa_search(S, sc, max_iters=600, time_budget=deadline - 10)
        update_best(S, sc)

    # Strategy E: More random restarts with SA
    for r in range(40):
        if time.time() > deadline - 12:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S = greedy_build(order, min_g=0)
        sc = score_full(S)
        S, sc = sa_search(S, sc, max_iters=200, time_budget=deadline - 8)
        update_best(S, sc)

    # Strategy F: SA from random density
    for p in [0.35, 0.45, 0.55, 0.65]:
        if time.time() > deadline - 12:
            break
        S = random_density(p)
        S = repair(S)
        sc = score_full(S)
        S, sc = sa_search(S, sc, max_iters=250, time_budget=deadline - 8)
        update_best(S, sc)

    # Strategy G: Tabu search from best
    if time.time() < deadline - 10 and best_S:
        S = int(best_S)
        sc = best_score
        S, sc = tabu_search(S, sc, max_iters=400, tabu_len=30, time_budget=deadline - 5)
        update_best(S, sc)

    # Strategy H: More random restarts if time permits
    for r in range(40):
        if time.time() > deadline - 6:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S = greedy_build(order, min_g=0)
        sc = score_full(S)
        S, sc = local_search(S, sc, max_rounds=25, time_budget=deadline - 4)
        update_best(S, sc)

    # Final polish
    if time.time() < deadline - 2 and best_S:
        best_S, best_score = local_search(best_S, best_score, max_rounds=50, time_budget=deadline - 0.5)

    # Verify and return
    result = [x for x in range(1, n + 1) if (best_S >> (x - 1)) & 1]
    if not result:
        result = [1]

    # Final validity check
    if not is_valid(best_S):
        # Fallback: greedy from 1 upward
        S = 0
        for x in range(1, n + 1):
            if can_add(x, S):
                S |= (1 << (x - 1))
        result = [x for x in range(1, n + 1) if (S >> (x - 1)) & 1]

    return result
