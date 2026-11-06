import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    base = 51975

    # All odd divisors > 1 of 51975, sorted ascending
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    divisors.sort()
    n_div = len(divisors)

    # Precompute residue-class index arrays (contiguous slices via arange)
    # class_idx[m] is a list of m arrays, each of length base//m
    class_idx = {}
    for m in divisors:
        step = m
        class_idx[m] = [np.arange(r, base, step) for r in range(m)]

    def elapsed():
        return time.time() - start_time

    def time_left():
        return deadline - time.time()

    def count_uncovered(cover_count):
        return int(np.count_nonzero(cover_count == 0))

    def greedy_build(order, seed=0):
        """Greedy: process moduli in `order`, pick residue maximizing newly-covered."""
        if time_left() < 2:
            return None
        rng = random.Random(seed)
        covered = np.zeros(base, dtype=np.uint8)
        classes = []
        for m in order:
            if time_left() < 3:
                break
            best_r = 0
            best_cnt = -1
            for r in range(m):
                idxs = class_idx[m][r]
                cnt = int(np.count_nonzero(covered[idxs] == 0))
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_r = r
            classes.append([m, best_r])
            covered[class_idx[m][best_r]] = 1
        return classes

    def greedy_build_descending(seed=0, max_classes=None):
        """Greedy with descending moduli order (large first), optionally limited to max_classes."""
        if time_left() < 2:
            return None
        rng = random.Random(seed)
        covered = np.zeros(base, dtype=np.uint8)
        classes = []
        order = sorted(divisors, reverse=True)
        if max_classes is not None:
            order = order[:max_classes]
        for m in order:
            if time_left() < 3:
                break
            best_r = 0
            best_cnt = -1
            for r in range(m):
                idxs = class_idx[m][r]
                cnt = int(np.count_nonzero(covered[idxs] == 0))
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_r = r
            classes.append([m, best_r])
            covered[class_idx[m][best_r]] = 1
        return classes

    # ---- Generate several greedy initializations ----
    candidates = []

    # 1. Increasing order (small moduli first) — best for covering density
    c = greedy_build(divisors, seed=0)
    if c: candidates.append(c)

    # 2. Decreasing order
    c = greedy_build(sorted(divisors, reverse=True), seed=1)
    if c: candidates.append(c)

    # 3. Order by (base//m) descending = small m first, but with random tie-break
    order3 = sorted(divisors, key=lambda m: (-base // m, random.Random(2).random()))
    c = greedy_build(order3, seed=2)
    if c: candidates.append(c)

    # 4. Random orders
    for s in range(8):
        if time_left() < 5:
            break
        rng = random.Random(100 + s)
        order = divisors[:]
        rng.shuffle(order)
        c = greedy_build(order, seed=200 + s)
        if c: candidates.append(c)

    # 5. "Gap-aware" order: start with small moduli (high coverage), then fill
    #    We'll just try a few curated orders.
    # Order: all divisors sorted by m ascending (same as 1) but we already have that.

    # NEW: Descending greedy with limited classes (sparse baseline)
    for max_c in [5, 10, 15, 20]:
        if time_left() < 5:
            break
        c = greedy_build_descending(seed=300 + max_c, max_classes=max_c)
        if c: candidates.append(c)

    # NEW: Mixed order - large moduli first, then small
    for s in range(4):
        if time_left() < 5:
            break
        rng = random.Random(400 + s)
        large_first = sorted(divisors, reverse=True)[:10]
        remaining = [m for m in divisors if m not in large_first]
        rng.shuffle(remaining)
        order = large_first + remaining
        c = greedy_build(order, seed=500 + s)
        if c: candidates.append(c)

    if not candidates:
        # Fallback: trivial single class
        return [[3, 0]]

    # Evaluate each candidate, keep the best
    def evaluate(classes):
        cover_count = np.zeros(base, dtype=np.int32)
        for m, r in classes:
            cover_count[class_idx[m][r]] += 1
        return count_uncovered(cover_count)

    best_classes = None
    best_unc = base
    for c in candidates:
        unc = evaluate(c)
        if unc < best_unc:
            best_unc = unc
            best_classes = [x[:] for x in c]

    if best_classes is None:
        return [[3, 0]]

    # ---- Local search ----
    def local_search(classes, max_rounds=200):
        classes = [x[:] for x in classes]
        cover_count = np.zeros(base, dtype=np.int32)
        for m, r in classes:
            cover_count[class_idx[m][r]] += 1
        uncovered = count_uncovered(cover_count)

        used_moduli = set(m for m, r in classes)
        available_moduli = [m for m in divisors if m not in used_moduli]

        improved = True
        round_num = 0
        while improved and time_left() > 3 and round_num < max_rounds:
            improved = False
            round_num += 1

            uncovered_mask = (cover_count == 0)
            exactly_one_mask = (cover_count == 1)

            # Move 1: change residue of an existing class
            for idx in range(len(classes)):
                if time_left() < 4:
                    break
                m, r = classes[idx]
                old_idxs = class_idx[m][r]
                # For each new_r != r, since classes are disjoint:
                #   loss = count of i in old_idxs with cover_count==1
                #   gain = count of i in new_idxs with cover_count==0
                #   delta = loss - gain  (we want delta < 0)
                loss = int(np.count_nonzero(exactly_one_mask[old_idxs]))
                best_delta = 0
                best_new_r = None
                for new_r in range(m):
                    if new_r == r:
                        continue
                    new_idxs = class_idx[m][new_r]
                    gain = int(np.count_nonzero(uncovered_mask[new_idxs]))
                    delta = loss - gain
                    if delta < best_delta:
                        best_delta = delta
                        best_new_r = new_r
                if best_new_r is not None:
                    new_idxs = class_idx[m][best_new_r]
                    cover_count[old_idxs] -= 1
                    cover_count[new_idxs] += 1
                    classes[idx][1] = best_new_r
                    uncovered += best_delta
                    improved = True

            # Move 2: add a new modulus
            if time_left() < 4:
                pass
            else:
                uncovered_mask = (cover_count == 0)
                best_add_gain = 0
                best_add = None
                for m in available_moduli:
                    if time_left() < 4:
                        break
                    best_r_for_m = None
                    best_gain_for_m = 0
                    for r in range(m):
                        idxs = class_idx[m][r]
                        gain = int(np.count_nonzero(uncovered_mask[idxs]))
                        if gain > best_gain_for_m:
                            best_gain_for_m = gain
                            best_r_for_m = r
                    if best_gain_for_m > best_add_gain:
                        best_add_gain = best_gain_for_m
                        best_add = (m, best_r_for_m)
                if best_add is not None:
                    m, r = best_add
                    classes.append([m, r])
                    idxs = class_idx[m][r]
                    cover_count[idxs] += 1
                    uncovered -= best_add_gain
                    used_moduli.add(m)
                    available_moduli.remove(m)
                    improved = True

            # Move 3: remove a redundant class (one that covers no uniquely-covered residue)
            if time_left() < 4:
                pass
            else:
                for idx in range(len(classes)):
                    if time_left() < 4:
                        break
                    m, r = classes[idx]
                    idxs = class_idx[m][r]
                    unique = int(np.count_nonzero(cover_count[idxs] == 1))
                    if unique == 0:
                        cover_count[idxs] -= 1
                        classes.pop(idx)
                        used_moduli.discard(m)
                        if m not in available_moduli:
                            available_moduli.append(m)
                        improved = True
                        break

        return classes, uncovered

    best_classes, best_unc = local_search(best_classes)

    # ---- Additional restarts if time permits ----
    # Try a few more greedy initializations + local search
    for s in range(20):
        if time_left() < 10:
            break
        rng = random.Random(500 + s)
        order = divisors[:]
        # Bias toward small moduli: sort with noise
        order.sort(key=lambda m: (m + rng.random() * 50, rng.random()))
        c = greedy_build(order, seed=600 + s)
        if not c:
            continue
        unc0 = evaluate(c)
        if unc0 >= best_unc:
            continue
        c2, unc2 = local_search(c, max_rounds=50)
        if unc2 < best_unc:
            best_unc = unc2
            best_classes = c2

    # NEW: Additional restarts with descending/mixed initializations
    for s in range(15):
        if time_left() < 10:
            break
        # Alternate between different initialization strategies
        if s % 3 == 0:
            c = greedy_build_descending(seed=700 + s, max_classes=None)
        elif s % 3 == 1:
            rng = random.Random(800 + s)
            large_first = sorted(divisors, reverse=True)[:8]
            remaining = [m for m in divisors if m not in large_first]
            rng.shuffle(remaining)
            order = large_first + remaining
            c = greedy_build(order, seed=900 + s)
        else:
            rng = random.Random(1000 + s)
            order = divisors[:]
            order.sort(key=lambda m: (m + rng.random() * 30, rng.random()))
            c = greedy_build(order, seed=1100 + s)
        if not c:
            continue
        unc0 = evaluate(c)
        if unc0 >= best_unc:
            continue
        c2, unc2 = local_search(c, max_rounds=50)
        if unc2 < best_unc:
            best_unc = unc2
            best_classes = c2

    # Final: ensure validity and return
    # Verify distinct moduli and valid residues
    seen = set()
    final = []
    for m, r in best_classes:
        if m in seen:
            continue
        if m <= 1 or m > base or base % m != 0:
            continue
        if r < 0 or r >= m:
            continue
        seen.add(m)
        final.append([m, r])

    if not final:
        return [[3, 0]]

    return final
