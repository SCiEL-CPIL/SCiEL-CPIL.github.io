import time
import random

_START = None

def _elapsed():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START

def _validate(colors):
    n = len(colors)
    # Check every nonconstant 4-term AP has >= 3 distinct colors
    for a in range(n):
        for d in range(1, (n - a - 1) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e >= n:
                break
            if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
                return False
    return True

def _count_distinct(colors):
    return len(set(colors))

def _try_mod_k(k, offset=0):
    """Color by (x+offset) mod k. Valid iff no nonconstant 4-AP uses <3 residues."""
    n = 128
    colors = [(x + offset) % k for x in range(n)]
    # quick validity check
    for a in range(n):
        for d in range(1, (n - a - 1) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e >= n:
                break
            if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
                return None
    return colors

def construct():
    random.seed(0)
    deadline = 60.0  # safety margin under 90s
    # Deadline guard helper
    def timeup():
        return _elapsed() > deadline

    # 1) Try small structured colorings first (fast, deterministic)
    for k in (3, 4, 5, 6, 7, 8):
        for off in range(k):
            c = _try_mod_k(k, off)
            if c is not None:
                return c

    # 2) Two-factor construction: color = (x mod r) * s + (floor(x/r) mod t)
    #    Explore small r,t to get <=4 colors with validity.
    for r in (2, 3, 4):
        for t in (2, 3, 4):
            if r * t > 6:
                continue
            n = 128
            colors = [((x % r) * t + (x // r) % t) for x in range(n)]
            if _validate(colors):
                return colors

    # 3) Local search / simulated annealing to minimize distinct colors
    #    Start from a known valid 5-coloring (mod 5) and try to merge colors.
    base = _try_mod_k(5, 0)
    if base is None:
        # Fallback: build greedy valid coloring
        colors = []
        for x in range(1, 129):
            for color in range(128):
                ok = True
                # 4-AP constraints where x is the last term
                for d in range(1, (x - 1) // 3 + 1):
                    idxs = [x - 1 - j * d for j in (1, 2, 3)]
                    if len({color} | {colors[i] for i in idxs}) < 3:
                        ok = False
                        break
                # x as 3rd term: with x+d, x+2d we need >=3 distinct among 4 incl later (handled later)
                # But also ensure not all first three same with a later equal -> handled when later added.
                if ok:
                    colors.append(color)
                    break
        return colors

    # We have a valid 5-coloring. Try to reduce number of used colors by recoloring.
    best = list(base)
    best_d = _count_distinct(best)

    n = 128
    # Precompute all 4-APs (a, b, c, e)
    aps = []
    for a in range(n):
        for d in range(1, (n - a - 1) // 3 + 1):
            e = a + 3 * d
            if e >= n:
                break
            aps.append((a, a + d, a + 2 * d, e))

    def violation_count(col):
        v = 0
        for (a, b, c, e) in aps:
            if len({col[a], col[b], col[c], col[e]}) < 3:
                v += 1
        return v

    cur = list(best)
    cur_v = violation_count(cur)
    cur_d = _count_distinct(cur)

    # Target: reduce distinct colors. We'll do a local search that prefers lower distinct
    # and lower violations.
    temperature = 2.0
    cooling = 0.995
    steps = 0
    max_steps = 200000
    color_pool = [0, 1, 2, 3, 4]  # allow 5 colors; we aim to end using <=4 or fewer

    while steps < max_steps and not timeup():
        i = random.randrange(n)
        old = cur[i]
        if old in color_pool:
            new = random.choice(color_pool)
        else:
            new = random.choice(color_pool)
        if new == old:
            steps += 1
            continue

        # Compute delta in violations by updating only APs containing i
        # Precompute APs containing each index for speed.
        # (recompute lazily)
        new_v = cur_v
        # We'll recompute violations for APs containing i
        # Build reverse map
        # To keep it simple but fast, maintain per-index list of ap-indices.
        pass

        # Instead, do a cheap full recompute every few steps, else accept heuristically.
        # For speed, maintain per-index AP list.
        # We'll build it once outside; but here rebuild if not present.
        steps += 1
        # Simple: full recompute is O(#APs) ~ maybe 2k-5k; with 200k steps too slow.
        # Break out and do a different strategy below.
        break

    # 4) Deterministic color-merging: try to map two colors of the valid 5-coloring
    #    onto one and repair via targeted recoloring with backtracking on conflicts.
    #    Try all pairs of colors to merge (5 choose 2 = 10).
    for c1 in range(5):
        for c2 in range(c1 + 1, 5):
            if timeup():
                break
            # Map c2 -> c1
            cand = [c1 if x == c2 else x for x in base]
            # Repair: recolor violating positions one by one with a color that fixes all
            # their APs, preferring to keep total distinct low.
            changed = True
            iters = 0
            while changed and iters < 200 and not timeup():
                changed = False
                iters += 1
                for (a, b, c, e) in aps:
                    if len({cand[a], cand[b], cand[c], cand[e]}) >= 3:
                        continue
                    # violation: pick a position to recolor
                    # try each of the 4 positions, each candidate color in {0..4}
                    fixed = False
                    for pos in (a, b, c, e):
                        for nc in (0, 1, 2, 3, 4):
                            if nc == cand[pos]:
                                continue
                            old = cand[pos]
                            cand[pos] = nc
                            # check if this AP is now ok AND no new violations introduced
                            ok = True
                            for (p1, p2, p3, p4) in aps:
                                if len({cand[p1], cand[p2], cand[p3], cand[p4]}) < 3:
                                    # only worry if this AP involves pos
                                    if pos not in (p1, p2, p3, p4):
                                        continue
                                    ok = False
                                    break
                            if ok:
                                fixed = True
                                changed = True
                                break
                            cand[pos] = old
                        if fixed:
                            break
                    if not fixed:
                        # try a different violating AP order; just continue
                        pass
            d = _count_distinct(cand)
            if d < best_d and _validate(cand):
                best = list(cand)
                best_d = d
                if best_d <= 3:
                    return best

    if _validate(best):
        return best

    # 5) Fallback: return the valid 5-coloring (mod 5) which we know is valid.
    return base
