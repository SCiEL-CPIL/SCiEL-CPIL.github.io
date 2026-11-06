import time
import random
import numpy as np

def construct():
    start_time = time.time()
    n = 256
    deadline = start_time + 82.0  # safety margin before 90s kill

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

    # Precompute the "other three" as a bitmask for each (x, 4-ap-idx)
    # mask4[x][k] = bitmask of the other 3 elements in the k-th 4-AP containing x
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

    # For each x, the (ap_idx, pos) lists for 3-APs
    # Precompute for fast gain: for each x, the list of (other1, other2) pairs
    # where x is a: need b,c in S
    # where x is b: need a,c in S
    # where x is c: need a,b in S
    # Represent as bitmask of required pairs? Simpler: list of (bit1, bit2)
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

    # ---------- Core operations on bitmask S (uint64, bit x-1 set if x in S) ----------
    def is_valid(S):
        """Check no 4-AP fully contained in S."""
        for ap_idx, ap in enumerate(four_aps):
            if (S >> (ap[0]-1)) & 1 and (S >> (ap[1]-1)) & 1 and (S >> (ap[2]-1)) & 1 and (S >> (ap[3]-1)) & 1:
                return False
        return True

    def can_add(x, S):
        """Check if adding x to S creates a 4-AP."""
        masks = mask4[x]
        if len(masks) == 0:
            return True
        # For each 4-AP containing x, check if other 3 are in S
        for m in masks:
            if (S & m) == m:
                return False
        return True

    def score(S):
        """Count 3-APs in S."""
        cnt = 0
        for a, b, c in aps_3:
            if (S >> (a-1)) & 1 and (S >> (b-1)) & 1 and (S >> (c-1)) & 1:
                cnt += 1
        return cnt

    def gain_add(x, S):
        """Number of new 3-APs formed by adding x to S."""
        g = 0
        for bit1, bit2 in gain_pairs[x]:
            if (S & bit1) and (S & bit2):
                g += 1
        return g

    def loss_remove(x, S):
        """Number of 3-APs lost by removing x from S."""
        g = 0
        S_wo_x = S & ~(1 << (x-1))
        for bit1, bit2 in gain_pairs[x]:
            if (S_wo_x & bit1) and (S_wo_x & bit2):
                g += 1
        return g

    # ---------- Local search ----------
    def local_search(S, current_score, max_iters=30, time_budget=None):
        """Local search with add, remove, 1-swap moves."""
        S = int(S)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time_budget is not None and time.time() > time_budget:
                break
            improved = False
            iters += 1

            # Phase 1: Try adding elements with positive gain
            added_any = False
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
                    added_any = True
                    if time_budget is not None and time.time() > time_budget:
                        break
            if time_budget is not None and time.time() > time_budget:
                break

            # Phase 2: Try removing elements to enable better additions
            # For each y in S, try removing it, then greedily add back
            if iters % 2 == 0:
                for y in range(1, n + 1):
                    if time_budget is not None and time.time() > time_budget - 1:
                        break
                    if not ((S >> (y-1)) & 1):
                        continue
                    S_trial = S & ~(1 << (y-1))
                    score_trial = current_score - loss_remove(y, S)
                    # Greedily add back
                    for x in range(1, n + 1):
                        if (S_trial >> (x-1)) & 1:
                            continue
                        if not can_add(x, S_trial):
                            continue
                        g = gain_add(x, S_trial)
                        if g > 0:
                            S_trial |= (1 << (x-1))
                            score_trial += g
                    if score_trial > current_score:
                        S = S_trial
                        current_score = score_trial
                        improved = True
                        if time_budget is not None and time.time() > time_budget:
                            break

            # Phase 3: Try 1-swap: remove y, add x (x not in S, y in S)
            if iters % 3 == 0:
                for y in range(1, n + 1):
                    if time_budget is not None and time.time() > time_budget - 1:
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

        return S, current_score

    # ---------- Construction strategies ----------
    def greedy_random(order, use_score=True):
        S = 0
        for x in order:
            if not can_add(x, S):
                continue
            if use_score:
                g = gain_add(x, S)
                if g > 0 or bin(S).count('1') < 20:
                    S |= (1 << (x-1))
            else:
                S |= (1 << (x-1))
        return S

    def prune_from_full():
        S = (1 << n) - 1  # all elements
        # Greedily remove to break 4-APs
        # For each 4-AP, if all present, remove the element with least 3-AP contribution
        for ap in four_aps:
            if all((S >> (elem-1)) & 1 for elem in ap):
                # Remove the element with lowest loss_remove
                min_loss = float('inf')
                min_elem = ap[0]
                for elem in ap:
                    l = loss_remove(elem, S)
                    if l < min_loss:
                        min_loss = l
                        min_elem = elem
                S &= ~(1 << (min_elem - 1))
        return S

    # ---------- Main search ----------
    best_S = 0
    best_score = 0

    # Strategy 1: Natural order greedy
    if time.time() < deadline - 20:
        S1 = greedy_random(range(1, n + 1), use_score=False)
        sc1 = score(S1)
        S1, sc1 = local_search(S1, sc1, max_iters=20, time_budget=deadline - 10)
        if sc1 > best_score:
            best_score = sc1
            best_S = S1

    # Strategy 2: Many random restarts
    random.seed(12345)
    n_restarts = 40
    for r in range(n_restarts):
        if time.time() > deadline - 15:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_r = greedy_random(order, use_score=True)
        sc_r = score(S_r)
        S_r, sc_r = local_search(S_r, sc_r, max_iters=15, time_budget=deadline - 10)
        if sc_r > best_score:
            best_score = sc_r
            best_S = S_r

    # Strategy 3: Prune from full
    if time.time() < deadline - 15:
        S_p = prune_from_full()
        sc_p = score(S_p)
        S_p, sc_p = local_search(S_p, sc_p, max_iters=25, time_budget=deadline - 8)
        if sc_p > best_score:
            best_score = sc_p
            best_S = S_p

    # Strategy 4: More random restarts if time permits
    for r in range(30):
        if time.time() > deadline - 5:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_r = greedy_random(order, use_score=True)
        sc_r = score(S_r)
        S_r, sc_r = local_search(S_r, sc_r, max_iters=20, time_budget=deadline - 3)
        if sc_r > best_score:
            best_score = sc_r
            best_S = S_r

    # Final local search on best
    if time.time() < deadline - 2 and best_S:
        best_S, best_score = local_search(best_S, best_score, max_iters=30, time_budget=deadline - 1)

    # Convert bitmask to sorted list
    result = [x for x in range(1, n + 1) if (best_S >> (x-1)) & 1]
    if not result:
        result = [1]
    return result

# EVOLVE-BLOCK-END
