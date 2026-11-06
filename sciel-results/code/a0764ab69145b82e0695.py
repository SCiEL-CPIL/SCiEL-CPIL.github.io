import time
import random

MAXV = 256

def _is_valid_seq(vals):
    """Validate: strictly increasing in [1,256], all consecutive-block sums distinct."""
    n = len(vals)
    if n == 0:
        return True
    prev = 0
    for v in vals:
        if v < 1 or v > MAXV or v <= prev:
            return False
        prev = v
    # block sums distinct
    seen = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += vals[j]
            if s in seen:
                return False
            seen.add(s)
    return True


def _greedy(vals, diffs, start_val, deadline, rng, prefer_large):
    """Fast greedy using prefix-difference set. Returns (vals, diffs)."""
    while time.time() < deadline:
        best = None
        order = range(start_val, MAXV + 1)
        if prefer_large:
            order = range(MAXV, start_val - 1, -1)
        else:
            cands = list(range(start_val, MAXV + 1))
            rng.shuffle(cands)
            order = cands
        for x in order:
            # new block sums = x + suffix sums of vals
            total = x
            ok = True
            for s in total, :
                pass
            # compute new sums
            newsums = []
            t = x
            for a in reversed(vals):
                t += a
                newsums.append(t)
            for s in newsums:
                if s in diffs:
                    ok = False
                    break
            if ok:
                for s in newsums:
                    diffs.add(s)
                vals.append(x)
                start_val = x + 1
                best = x
                break
        if best is None:
            break
    return vals, diffs


def _bnb(vals, diffs, start_val, deadline, rng, max_depth, branch_factor,
         prefer_large):
    """Branch-and-bound over prefix-sum Sidon construction.

    vals: current sequence (strictly increasing).
    diffs: set of all positive differences p[j]-p[i] (block sums).
    Returns best (longest) sequence found within deadline.
    """
    best = list(vals)
    best_len = len(vals)
    # current last prefix value = sum(vals)
    cur_last = sum(vals) if vals else 0

    def rec(vals, diffs, start_val, depth):
        nonlocal best, best_len
        if time.time() > deadline:
            return
        n = len(vals)
        # admissible upper bound on final length:
        # we can add at most (MAXV - last) more elements each >=1, but they must
        # be strictly increasing so at most (MAXV - last) of them.
        cur_last = sum(vals) if vals else 0
        remaining_slots = MAXV - (vals[-1] if vals else 0)
        ub = n + remaining_slots
        if ub <= best_len:
            return

        if n >= best_len:
            # record
            if n > best_len:
                best = list(vals)
                best_len = n

        if depth >= max_depth:
            return

        # candidate ordering
        if prefer_large:
            order = list(range(MAXV, vals[-1] if vals else 0, -1))
        else:
            order = list(range(vals[-1] + 1 if vals else 1, MAXV + 1))
            rng.shuffle(order)

        tried = 0
        for x in order:
            if time.time() > deadline:
                return
            if x <= (vals[-1] if vals else 0):
                continue
            # compute new block sums
            newsums = []
            t = x
            for a in reversed(vals):
                t += a
                newsums.append(t)
            conflict = False
            for s in newsums:
                if s in diffs:
                    conflict = True
                    break
            if conflict:
                continue
            # commit
            for s in newsums:
                diffs.add(s)
            vals.append(x)

            rec(vals, diffs, x + 1, depth + 1)

            vals.pop()
            for s in newsums:
                diffs.discard(s)

            tried += 1
            if tried >= branch_factor:
                break

    rec(vals, diffs, start_val, 0)
    return best


def construct():
    t0 = time.time()
    deadline = t0 + 82.0  # safety margin under 90s

    best = []
    best_len = 0

    # ---- Phase 1: fast greedy warm starts (several variants) ----
    rng = random.Random(12345)
    for attempt in range(40):
        if time.time() > deadline - 20.0:
            break
        rng = random.Random(attempt * 7919 + 31)
        vals = []
        diffs = set()
        prefer_large = (attempt % 2 == 0)
        vals, diffs = _greedy(vals, diffs, 1, deadline - 20.0, rng, prefer_large)
        if len(vals) > best_len:
            best = vals
            best_len = len(vals)

    # ---- Phase 2: branch-and-bound from warm starts ----
    # Try a few different branch-and-bound configurations.
    configs = [
        dict(max_depth=6, branch_factor=2, prefer_large=True),
        dict(max_depth=6, branch_factor=3, prefer_large=True),
        dict(max_depth=5, branch_factor=4, prefer_large=False),
        dict(max_depth=7, branch_factor=2, prefer_large=True),
    ]
    for ci, cfg in enumerate(configs):
        if time.time() > deadline - 3.0:
            break
        rng = random.Random(1000 + ci)
        # Use the current best as seed, plus a couple of fresh greedy seeds.
        seeds = [list(best)]
        for g in range(3):
            if time.time() > deadline - 3.0:
                break
            rng2 = random.Random(ci * 100 + g * 7 + 1)
            v2, d2 = _greedy([], set(), 1, deadline - 3.0, rng2, True)
            seeds.append(v2)

        for seed in seeds:
            if time.time() > deadline - 2.0:
                break
            seed = list(seed)
            diffs = set()
            # rebuild diffs for seed
            n = len(seed)
            for i in range(n):
                s = 0
                for j in range(i, n):
                    s += seed[j]
                    diffs.add(s)
            res = _bnb(seed, diffs, seed[-1] + 1 if seed else 1,
                       deadline - 2.0, random.Random(ci * 17 + len(seed)),
                       cfg['max_depth'], cfg['branch_factor'],
                       cfg['prefer_large'])
            if len(res) > best_len:
                best = res
                best_len = len(res)

    # ---- Final validation ----
    if not _is_valid_seq(best):
        # fallback: deterministic safe sequence
        best = [1, 2, 4, 8, 16, 32, 64, 128]
        # ensure validity
        if not _is_valid_seq(best):
            best = [1, 2, 4, 8, 16, 32, 64]
    return best
