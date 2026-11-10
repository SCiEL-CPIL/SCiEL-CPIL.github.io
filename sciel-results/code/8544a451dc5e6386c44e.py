"""Optimized solver for large reciprocal weight under lcm separation."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 82.0  # safety margin before 90s wall-clock

    N = 256
    nums = list(range(2, 257))
    n = len(nums)  # 255

    # ---- conflict graph: conflict[i][j] = 1 iff lcm(nums[i], nums[j]) <= 256 ----
    conflict = [bytearray(n) for _ in range(n)]
    for i in range(n):
        a = nums[i]
        for j in range(i + 1, n):
            b = nums[j]
            g = gcd(a, b)
            if a * b // g <= N:
                conflict[i][j] = 1
                conflict[j][i] = 1

    # ---- weights as exact integers (LCM 1..256) so bounds are exact ----
    L = 1
    for x in nums:
        L = L * x // gcd(L, x)
    weights = [L // x for x in nums]  # weight[i] = L / nums[i]  (exact)

    # ---- order: descending weight (small numbers first); tie-break by degree ----
    degree = [sum(conflict[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: (-weights[i], degree[i]))
    # reverse-indices: for a node at position p, its candidate set is order[p:]
    rev = [0] * n
    for p, idx in enumerate(order):
        rev[idx] = p
    # precompute cumulative suffix of weights (in order) for the fractional bound
    suffix = [0] * (n + 1)
    for p in range(n - 1, -1, -1):
        suffix[p] = suffix[p + 1] + weights[order[p]]

    def frac_bound(start, cand_mask):
        """Fractional knapsack upper bound over remaining candidates (start..n-1)."""
        total = 0
        p = start
        while p < n:
            idx = order[p]
            if (cand_mask >> idx) & 1:
                total += weights[idx]
            p += 1
        return total

    # ---- branch-and-bound (iterative) ----
    best_w = [0]
    best_sel = [0]
    stack = [(0, 0, (1 << n) - 1)]  # (position, weight, candidate_mask)
    t0 = time.time()
    nodes = 0
    while stack:
        if time.time() - t0 > (DEADLINE - 10.0):
            break
        pos, w, cm = stack.pop()
        nodes += 1
        # fractional upper bound
        ub = w + frac_bound(pos, cm)
        if ub <= best_w[0]:
            continue
        # try to pick candidates at position pos (in descending weight order)
        picked_any = False
        p = pos
        while p < n:
            idx = order[p]
            if not ((cm >> idx) & 1):
                p += 1
                continue
            # include idx
            nw = w + weights[idx]
            if nw > best_w[0]:
                best_w[0] = nw
                best_sel[0] = idx
            # remaining candidates: those after p that don't conflict with idx
            new_cm = cm & ~conflict[idx] & ~((1 << idx) - 1) & ~((1 << idx))
            # restrict to positions > p: clear bits for order[0..p]
            for q in range(p + 1):
                new_cm &= ~((1 << order[q]))
            if new_cm:
                stack.append((p + 1, nw, new_cm))
            # exclude idx: clear its bit and continue
            cm &= ~((1 << idx))
            picked_any = True
            p += 1
            if not cm:
                break
    bbs_time = time.time() - t0

    # ---- exact local search polish (add / 1-1 / 2-1 / 1-2 moves) ----
    # reconstruct best selection as a set
    sel = set()
    m = best_sel[0]
    while m:
        lsb = m & (-m)
        sel.add(lsb.bit_length() - 1)
        m ^= lsb

    def can_add(mask, idx):
        return (mask & conflict[idx]) == 0

    cur_w = best_w[0]
    cur_mask = 0
    for idx in sel:
        cur_mask |= (1 << idx)
    all_idx = list(range(n))

    def try_moves(mask, w, selset, budget):
        improved = True
        while improved and (time.time() - start_time) < budget:
            improved = False
            # 1) add
            best_add = -1
            for idx in all_idx:
                if (mask >> idx) & 1:
                    continue
                if can_add(mask, idx):
                    best_add = idx
                    break
            if best_add >= 0:
                mask |= (1 << best_add)
                w += weights[best_add]
                selset.add(best_add)
                improved = True
                continue
            # 2) 1-for-1 swap
            best_swap = None
            for i in list(selset):
                nm = mask & ~(1 << i)
                for j in all_idx:
                    if (mask >> j) & 1:
                        continue
                    if can_add(nm, j):
                        g = weights[j] - weights[i]
                        if best_swap is None or g > best_swap[2]:
                            best_swap = (i, j, g)
            if best_swap and best_swap[2] > 0:
                i, j, g = best_swap
                mask = (mask & ~(1 << i)) | (1 << j)
                selset.remove(i)
                selset.add(j)
                w += g
                improved = True
                continue
            # 3) 2-for-1 swap
            if (time.time() - start_time) < budget - 2:
                best21 = None
                sl = list(selset)
                for a in range(len(sl)):
                    i = sl[a]
                    for b in range(a + 1, len(sl)):
                        j = sl[b]
                        nm = mask & ~(1 << i) & ~(1 << j)
                        for k in all_idx:
                            if (mask >> k) & 1:
                                continue
                            if can_add(nm, k):
                                g = weights[k] - weights[i] - weights[j]
                                if best21 is None or g > best21[3]:
                                    best21 = (i, j, k, g)
                if best21 and best21[3] > 0:
                    i, j, k, g = best21
                    mask = (mask & ~(1 << i) & ~(1 << j)) | (1 << k)
                    selset.remove(i)
                    selset.remove(j)
                    selset.add(k)
                    w += g
                    improved = True
                    continue
            # 4) 1-for-2 swap
            if (time.time() - start_time) < budget - 1:
                found = False
                for i in list(selset):
                    nm = mask & ~(1 << i)
                    bestpair = None
                    for j in all_idx:
                        if (mask >> j) & 1:
                            continue
                        if not can_add(nm, j):
                            continue
                        nmj = nm | (1 << j)
                        for k in all_idx:
                            if (mask >> k) & 1 or k == j:
                                continue
                            if can_add(nmj, k):
                                g = weights[j] + weights[k] - weights[i]
                                if bestpair is None or g > bestpair[2]:
                                    bestpair = (j, k, g)
                    if bestpair and bestpair[2] > 0:
                        j, k, g = bestpair
                        mask = (mask & ~(1 << i)) | (1 << j) | (1 << k)
                        selset.remove(i)
                        selset.add(j)
                        selset.add(k)
                        w += g
                        improved = True
                        found = True
                        break
                if found:
                    continue
        return mask, w, selset

    cur_mask, cur_w, sel = try_moves(cur_mask, cur_w, sel, DEADLINE - 1.0)

    # ---- convert to values ----
    result = []
    m = cur_mask
    while m:
        lsb = m & (-m)
        idx = lsb.bit_length() - 1
        result.append(nums[idx])
        m ^= lsb
    if not result:
        result = [2]
    return sorted(result)
# EVOLVE-BLOCK-END
