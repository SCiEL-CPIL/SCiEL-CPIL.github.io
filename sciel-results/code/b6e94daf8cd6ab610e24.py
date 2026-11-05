# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 110.0  # safety margin under 150s wall-clock

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def largest_half_sumfree(vals):
        n = len(vals)
        ordered = sorted(vals, key=lambda x: -abs(x))
        half = ordered[: (n + 1) // 2]
        if not is_sum_free(half):
            chosen = []
            chosen_set = set()
            for x in ordered:
                ok = True
                for y in chosen:
                    if x + y in chosen_set:
                        ok = False
                        break
                if 2 * x in chosen_set:
                    ok = False
                if ok:
                    chosen.append(x)
                    chosen_set.add(x)
            return chosen
        return half

    def greedy_fill(vals, base):
        base_set = set(base)
        candidates = sorted([v for v in vals if v not in base_set], key=lambda x: -abs(x))
        chosen = list(base)
        chosen_set = set(base)
        for x in candidates:
            ok = True
            for y in chosen:
                if x + y in chosen_set:
                    ok = False
                    break
            if 2 * x in chosen_set:
                ok = False
            if ok:
                chosen.append(x)
                chosen_set.add(x)
        return chosen

    def conflict_elements(x, sset):
        """Return set of elements in sset involved in a sum-free violation with x."""
        els = set()
        for a in sset:
            b = x - a
            if b in sset:
                els.add(a)
                els.add(b)
        if 2 * x in sset:
            els.add(x)
        return els

    def local_search(subset, vals, max_rounds=500):
        cur_set = set(subset)
        best = len(cur_set)
        candidates = [v for v in vals if v not in cur_set]
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            # Try pure adds
            for x in candidates:
                if x in cur_set:
                    continue
                if not conflict_elements(x, cur_set):
                    cur_set.add(x)
                    best = len(cur_set)
                    improved = True
            if improved:
                continue
            # Try remove+add
            cand_sorted = sorted(candidates, key=lambda x: -abs(x))
            for x in cand_sorted:
                if x in cur_set:
                    continue
                conf = conflict_elements(x, cur_set)
                if not conf:
                    continue
                for r in conf:
                    new_set = cur_set - {r}
                    new_set.add(x)
                    # verify
                    ok = True
                    lst = list(new_set)
                    for a in lst:
                        for b in lst:
                            if a + b in new_set:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        cur_set = new_set
                        best = len(cur_set)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return list(cur_set)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        base = largest_half_sumfree(vals)
        filled = greedy_fill(vals, base)
        improved = local_search(filled, vals)
        results.append(improved)
    return results
# EVOLVE-BLOCK-END
