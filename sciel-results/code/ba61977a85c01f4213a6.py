import random
import time

N = 256

def construct():
    start = time.time()
    deadline = start + 50.0

    def is_valid(sel):
        seen = set()
        for i in range(len(sel)):
            for j in range(i, len(sel)):
                s = sel[i] + sel[j]
                if s in seen:
                    return False
                seen.add(s)
        return True

    def maximal_deficit(sel):
        """Return list of missing elements that could still be added (empty => maximal)."""
        sums = set()
        for i in range(len(sel)):
            for j in range(i, len(sel)):
                sums.add(sel[i] + sel[j])
        in_set = set(sel)
        bad = []
        for x in range(1, N + 1):
            if x in in_set:
                continue
            if any((x + a) in sums for a in sel):
                continue
            if (2 * x) in sums:
                continue
            bad.append(x)
        return bad

    def greedy_block(order, sel0=None, sums0=None, pairs0=None):
        """Greedy: repeatedly add a candidate whose new sums block the most remaining elements."""
        sel = list(sel0) if sel0 else []
        sums = set(sums0) if sums0 else set()
        # sum -> list of (i,j) with i<=j producing it
        pairs = {}
        if pairs0:
            for s, lst in pairs0.items():
                pairs[s] = list(lst)
        else:
            for i in range(len(sel)):
                for j in range(i, len(sel)):
                    s = sel[i] + sel[j]
                    pairs.setdefault(s, []).append((i, j))
        remaining = [x for x in order if x not in set(sel)]
        remaining_set = set(remaining)

        while remaining:
            best_x = None
            best_block = -1
            best_sums = None
            for x in remaining:
                new_sums = [x + a for a in sel] + [2 * x]
                if len(set(new_sums)) != len(new_sums):
                    continue
                if any(s in sums for s in new_sums):
                    continue
                # how many remaining elements get blocked by the new sums
                ns = set(new_sums)
                block = 0
                for y in remaining:
                    if y == x:
                        continue
                    blocked = False
                    for a in sel:
                        if (x + y) == (a + a) or (x + y) in ns or (y + a) in ns:
                            blocked = True
                            break
                    if not blocked and (2 * x) in ns:
                        # already covered by new_sums distinctness
                        pass
                    if blocked:
                        block += 1
                if block > best_block:
                    best_block = block
                    best_x = x
                    best_sums = new_sums
            if best_x is None:
                break
            # commit
            sel.append(best_x)
            idx = len(sel) - 1
            for a in sel[:-1]:
                s = best_x + a
                sums.add(s)
                pairs.setdefault(s, []).append((idx, sel.index(a)))
            sums.add(2 * best_x)
            pairs.setdefault(2 * best_x, []).append((idx, idx))
            remaining.remove(best_x)
            if time.time() > deadline:
                break
        return sel

    def repair(sel):
        """Add elements until maximal (or no more can be added)."""
        sel = list(sel)
        while True:
            deficit = maximal_deficit(sel)
            if not deficit:
                return sel
            # add the first deficit element that keeps validity
            added = False
            for x in deficit:
                new = sel + [x]
                if is_valid(new):
                    sel = new
                    added = True
                    break
            if not added:
                # shouldn't happen if deficit was computed correctly
                return sel

    def local_search(sel):
        """Try swapping a member for a smaller-deficit alternative to shrink size."""
        sel = list(sel)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(sel)
            for i in range(n):
                cur = sel[i]
                base = sel[:i] + sel[i + 1:]
                for x in range(1, N + 1):
                    if x in sel:
                        continue
                    cand = base + [x]
                    if is_valid(cand):
                        # prefer if it reduces or keeps deficit and size
                        if len(cand) <= n:
                            # accept swap that keeps valid (size equal)
                            # only accept if it helps future reduction: keep if deficit not larger
                            if len(maximal_deficit(cand)) <= len(maximal_deficit(sel)):
                                sel = cand
                                improved = True
                                break
                if improved:
                    break
        return sel

    best = None
    best_size = float('inf')

    # Baseline: ascending greedy
    def simple_greedy(order):
        sel = []
        sums = set()
        for x in order:
            ns = [x + a for a in sel] + [2 * x]
            if len(set(ns)) == len(ns) and not any(s in sums for s in ns):
                sel.append(x)
                sums.update(ns)
        return sel

    base = simple_greedy(range(1, N + 1))
    base = repair(base)
    if is_valid(base):
        best = base
        best_size = len(base)

    # Randomized block-greedy over many seeds
    for seed in range(4000):
        if time.time() > deadline:
            break
        random.seed(seed)
        order = list(range(1, N + 1))
        random.shuffle(order)
        sel = greedy_block(order)
        if not sel:
            continue
        if not is_valid(sel):
            continue
        if len(sel) >= best_size:
            # still might repair to something; skip heavy work
            continue
        sel = repair(sel)
        if not is_valid(sel):
            continue
        sel = local_search(sel)
        if is_valid(sel) and len(sel) < best_size:
            best = sel
            best_size = len(sel)

    if best is None:
        best = base
    return best
