import time
import random
import numpy as np

def construct():
    start_time = time.time()
    n = 256
    deadline = start_time + 82.0

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

    # ---------- New: Conflict-based repair ----------
    def find_conflicts(S):
        """Return list of 4-APs fully contained in S."""
        conflicts = []
        for ap in four_aps:
            if all((S >> (elem-1)) & 1 for elem in ap):
                conflicts.append(ap)
        return conflicts

    def repair(S):
        """Remove minimal elements to break all 4-AP conflicts."""
        S = int(S)
        conflicts = find_conflicts(S)
        while conflicts:
            # Find element appearing in most conflicts
            from collections import Counter
            elem_count = Counter()
            for ap in conflicts:
                for e in ap:
                    elem_count[e] += 1
            # Remove element with highest conflict count, tie-break by lowest loss
            best_elem = None
            best_key = None
            for e, cnt in elem_count.most_common():
                if not ((S >> (e-1)) & 1):
                    continue
                loss = loss_remove(e, S)
                key = (cnt, -loss)
                if best_key is None or key > best_key:
                    best_key = key
                    best_elem = e
            if best_elem is None:
                break
            S &= ~(1 << (best_elem - 1))
            conflicts = find_conflicts(S)
        return S

    # ---------- New: Weighted greedy with look-ahead ----------
    def weighted_greedy(order, look_ahead=2):
        S = 0
        for i, x in enumerate(order):
            if not can_add(x, S):
                continue
            g = gain_add(x, S)
            if g <= 0:
                # Check if adding x enables future gains
                if look_ahead > 0:
                    S_trial = S | (1 << (x-1))
                    future_gain = 0
                    for j in range(i+1, min(i+look_ahead+5, len(order))):
                        y = order[j]
                        if not ((S_trial >> (y-1)) & 1) and can_add(y, S_trial):
                            future_gain += gain_add(y, S_trial)
                    if future_gain <= 0:
                        continue
            S |= (1 << (x-1))
        return S

    # ---------- New: Modular structure-based initialization ----------
    def modular_init(modulus, residues):
        """Build set using modular arithmetic to avoid 4-APs."""
        S = 0
        for x in range(1, n + 1):
            if x % modulus in residues:
                if can_add(x, S):
                    S |= (1 << (x-1))
        return S

    # ---------- Fast local search (single pass) ----------
    def fast_local_search(S, current_score, max_passes=5, time_budget=None):
        S = int(S)
        for p in range(max_passes):
            if time_budget is not None and time.time() > time_budget:
                break
            improved = False
            # Add phase
            for x in range(1, n + 1):
                if time_budget is not None and time.time() > time_budget - 0.1:
                    break
                if (S >> (x-1)) & 1:
                    continue
                if not can_add(x, S):
                    continue
                g = gain_add(x, S)
                if g > 0:
                    S |= (1 << (x-1))
                    current_score += g
                    improved = True
            # Swap phase
            if not improved:
                in_set = [x for x in range(1, n+1) if (S >> (x-1)) & 1]
                not_in_set = [x for x in range(1, n+1) if not ((S >> (x-1)) & 1)]
                for y in in_set:
                    if time_budget is not None and time.time() > time_budget - 0.1:
                        break
                    S_wo = S & ~(1 << (y-1))
                    sw = current_score - loss_remove(y, S)
                    best_x = -1
                    best_g = 0
                    for x in not_in_set:
                        if not can_add(x, S_wo):
                            continue
                        g = gain_add(x, S_wo)
                        if g > best_g:
                            best_g = g
                            best_x = x
                    if best_x > 0 and sw + best_g > current_score:
                        S = S_wo | (1 << (best_x - 1))
                        current_score = sw + best_g
                        improved = True
                        break
            if not improved:
                break
        return S, current_score

    # ---------- Main search ----------
    best_S = 0
    best_score = 0

    # Strategy 1: Multiple modular initializations
    random.seed(42)
    modular_configs = [
        (3, {0, 1}),
        (4, {0, 1, 3}),
        (5, {0, 1, 3, 4}),
        (7, {0, 1, 2, 4, 5, 6}),
        (8, {0, 1, 3, 5, 7}),
        (9, {0, 1, 2, 4, 5, 7, 8}),
        (11, {0, 1, 2, 3, 5, 6, 8, 9, 10}),
    ]
    for mod, res in modular_configs:
        if time.time() > deadline - 30:
            break
        S_m = modular_init(mod, res)
        sc_m = score(S_m)
        S_m, sc_m = fast_local_search(S_m, sc_m, max_passes=8, time_budget=deadline - 25)
        if sc_m > best_score:
            best_score = sc_m
            best_S = S_m

    # Strategy 2: Random order weighted greedy + repair + local search
    for r in range(40):
        if time.time() > deadline - 20:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_r = weighted_greedy(order, look_ahead=2)
        S_r = repair(S_r)
        sc_r = score(S_r)
        S_r, sc_r = fast_local_search(S_r, sc_r, max_passes=10, time_budget=deadline - 15)
        if sc_r > best_score:
            best_score = sc_r
            best_S = S_r

    # Strategy 3: Start from best, try diverse perturbations
    if best_S and time.time() < deadline - 15:
        for pert in range(15):
            if time.time() > deadline - 10:
                break
            S_p = int(best_S)
            # Randomly remove k elements and repair
            k = random.randint(3, 10)
            in_set = [x for x in range(1, n+1) if (S_p >> (x-1)) & 1]
            to_remove = random.sample(in_set, min(k, len(in_set)))
            for x in to_remove:
                S_p &= ~(1 << (x-1))
            S_p = repair(S_p)
            sc_p = score(S_p)
            S_p, sc_p = fast_local_search(S_p, sc_p, max_passes=8, time_budget=deadline - 8)
            if sc_p > best_score:
                best_score = sc_p
                best_S = S_p

    # Strategy 4: Final intensive local search
    if best_S and time.time() < deadline - 5:
        best_S, best_score = fast_local_search(best_S, best_score, max_passes=20, time_budget=deadline - 2)

    # Strategy 5: Try to add any remaining beneficial elements
    if best_S and time.time() < deadline - 2:
        for x in range(1, n + 1):
            if time.time() > deadline - 1:
                break
            if (best_S >> (x-1)) & 1:
                continue
            if can_add(x, best_S):
                g = gain_add(x, best_S)
                if g > 0:
                    best_S |= (1 << (x-1))
                    best_score += g

    result = [x for x in range(1, n + 1) if (best_S >> (x-1)) & 1]
    if not result:
        result = [1]
    return result
