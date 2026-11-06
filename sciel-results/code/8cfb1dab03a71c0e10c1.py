def construct():
    import time
    from random import Random

    t0 = time.time()
    cutoff = t0 + 112

    def gen_set(n, s, lo=-10000, hi=10000):
        r = Random(s)
        vals = set()
        while len(vals) < n:
            v = r.randint(lo, hi)
            if v != 0:
                vals.add(v)
        return sorted(vals)

    def verify(sf):
        ss = set(sf)
        for a in sf:
            for b in sf:
                if a + b in ss:
                    return False
        return True

    def conflicts_with(x, ss):
        """Return set of elements in ss that conflict with x."""
        conf = set()
        for a in ss:
            if a + x in ss:
                conf.add(a)
            if x - a in ss:
                conf.add(x - a)
        if 2 * x in ss:
            conf.add(x)
        return conf

    def build_from_seed(seed_elems, full_set, rng, max_adds=200):
        """Grow a seed set by greedily adding non-conflicting elements."""
        cur = list(seed_elems)
        cur_set = set(cur)
        remaining = [x for x in full_set if x not in cur_set]
        rng.shuffle(remaining)
        for x in remaining:
            if len(cur) - len(cur_set) != 0:
                continue
            if 2 * x in cur_set:
                continue
            bad = False
            for a in cur_set:
                if a + x in cur_set or x - a in cur_set:
                    bad = True
                    break
            if not bad:
                cur.append(x)
                cur_set.add(x)
        return cur

    def perturb_and_regrow(cur, full_set, rng, n_perturb=3):
        """Remove some random elements and try to re-add better ones."""
        cur = list(cur)
        cur_set = set(cur)
        # Remove n_perturb random elements
        to_remove = rng.sample(cur, min(n_perturb, len(cur)))
        for x in to_remove:
            cur.remove(x)
            cur_set.discard(x)
        # Try to add back removed elements or new ones
        candidates = [x for x in full_set if x not in cur_set]
        rng.shuffle(candidates)
        for x in candidates:
            if 2 * x in cur_set:
                continue
            ok = True
            for a in cur_set:
                if a + x in cur_set or x - a in cur_set:
                    ok = False
                    break
            if ok:
                cur.append(x)
                cur_set.add(x)
        return cur

    def solve_one(n, seed):
        full = gen_set(n, seed)
        rng = Random(seed * 7 + 42)
        n_elems = len(full)

        # Strategy 1: All elements with |x| >= threshold of one sign
        # Strategy 2: Greedy from scratch with multiple random orderings
        # Strategy 3: Perturbation local search

        best = []

        # Greedy from scratch - try multiple random orderings
        for trial in range(8):
            cand = list(full)
            rng.shuffle(cand)
            cur = []
            cur_set = set()
            for x in cand:
                if 2 * x in cur_set:
                    continue
                ok = True
                for a in cur_set:
                    if a + x in cur_set or x - a in cur_set:
                        ok = False
                        break
                if ok:
                    cur.append(x)
                    cur_set.add(x)
            if len(cur) > len(best):
                best = list(cur)

        # Try seed-based: take largest magnitude elements
        by_mag = sorted(full, key=lambda x: -abs(x))
        # Try taking top k by magnitude
        for k in [n_elems // 2, n_elems // 3, n_elems // 4]:
            seed_elems = by_mag[:k]
            grown = build_from_seed(seed_elems, full, rng)
            if len(grown) > len(best):
                best = list(grown)

        # Try odd/even split
        odds = [x for x in full if x % 2 != 0]
        evens = [x for x in full if x % 2 == 0]
        if len(odds) > len(best):
            best = list(odds)
        if len(evens) > len(best):
            best = list(evens)

        # Local search: perturb and regrow
        for iteration in range(30):
            if time.time() > cutoff:
                break
            perturbed = perturb_and_regrow(best, full, rng, n_perturb=2)
            if len(perturbed) > len(best):
                best = perturbed

        # Final add pass
        cur_set = set(best)
        remaining = [x for x in full if x not in cur_set]
        rng.shuffle(remaining)
        for x in remaining:
            if 2 * x in cur_set:
                continue
            ok = True
            for a in cur_set:
                if a + x in cur_set or x - a in cur_set:
                    ok = False
                    break
            if ok:
                best.append(x)
                cur_set.add(x)

        if not verify(best):
            # Fallback: take all odd
            best = [x for x in full if x % 2 != 0]

        return best

    out = []
    for sz, sd in [(60, 801), (120, 802), (240, 803)]:
        if time.time() > cutoff:
            f = gen_set(sz, sd)
            out.append([x for x in f if x % 2 != 0])
            continue
        out.append(solve_one(sz, sd))

    return out
