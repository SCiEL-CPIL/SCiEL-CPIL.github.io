import time
import random

# ---------------------------------------------------------------------------
# Fast incremental validity check for a strictly-increasing sequence.
# Only consecutive (contiguous) block sums must be distinct.
# Returns (is_valid, sums_set).
# ---------------------------------------------------------------------------
def _check(seq):
    n = len(seq)
    if n == 0:
        return True, set()
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False, sums
            sums.add(s)
    return True, sums

# ---------------------------------------------------------------------------
# Greedy builders (append at the end, strictly increasing).
# These are fast and give good diverse starting points.
# ---------------------------------------------------------------------------
def _greedy_smallest():
    values = []
    sums = set()
    for x in range(1, 257):
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _greedy_largest():
    values = []
    sums = set()
    last = 0
    while True:
        found_c = None
        found_tails = None
        for c in range(256, last, -1):
            tails = [c]
            total = c
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                found_c = c
                found_tails = tails
                break
        if found_c is None:
            break
        values.append(found_c)
        sums.update(found_tails)
        last = found_c
    return values

def _greedy_random(rng, bias=0.5):
    values = []
    sums = set()
    last = 0
    while True:
        valid = []
        for c in range(last + 1, 257):
            tails = [c]
            total = c
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                valid.append((c, tails))
        if not valid:
            break
        k = len(valid)
        start = int(bias * (k - 1))
        start = max(0, min(start, k - 1))
        idx = rng.randint(start, k - 1)
        c, tails = valid[idx]
        values.append(c)
        sums.update(tails)
        last = c
    return values

def _greedy_shuffled(rng):
    # Insert elements in a random order, maintaining sorted order and validity.
    order = list(range(1, 257))
    rng.shuffle(order)
    selected = []
    for x in order:
        if x in selected:
            continue
        cand = selected + [x]
        cand.sort()
        ok, _ = _check(cand)
        if ok:
            selected = cand
    return selected

# ---------------------------------------------------------------------------
# Local search refiner. Works on a sorted list. Moves:
#   * add    : insert an unused value anywhere (keeps sorted)
#   * swap   : remove one, insert a different unused value
#   * remove : drop one (escape local maxima)
#   * polish : greedy sweep over remaining values in ascending order
# ---------------------------------------------------------------------------
def _local_search(start_seq, deadline, rng):
    seq = sorted(start_seq)
    used = set(seq)
    best = seq[:]
    best_len = len(seq)
    stall = 0
    last_imp = time.time()
    universe = set(range(1, 257))

    def try_add(x):
        nonlocal seq, used, best, best_len, stall, last_imp
        if x in used:
            return
        cand = seq + [x]
        cand.sort()
        ok, _ = _check(cand)
        if ok:
            seq = cand
            used.add(x)
            if len(seq) > best_len:
                best_len = len(seq)
                best = seq[:]
                last_imp = time.time()
                stall = 0
            else:
                stall += 1
        else:
            stall += 1

    def try_swap(a, b):
        nonlocal seq, used, best, best_len, stall, last_imp
        if a == b or a not in used or b in used:
            return
        cand = [v for v in seq if v != a]
        cand.append(b)
        cand.sort()
        ok, _ = _check(cand)
        if ok:
            seq = cand
            used.remove(a)
            used.add(b)
            if len(seq) > best_len:
                best_len = len(seq)
                best = seq[:]
                last_imp = time.time()
                stall = 0
            else:
                stall += 1
        else:
            stall += 1

    while time.time() < deadline:
        r = rng.random()
        avail = universe - used
        if r < 0.62:
            if avail:
                try_add(rng.choice(sorted(avail)))
        elif r < 0.80:
            if avail and seq:
                try_swap(rng.choice(seq), rng.choice(sorted(avail)))
        else:
            # remove to escape
            if seq:
                x = rng.choice(seq)
                seq.remove(x)
                used.remove(x)
                stall += 1

        # Periodic greedy polish: fill remaining gaps in ascending order.
        if stall > 40 or (time.time() - last_imp > 2.0 and stall > 8):
            test = seq[:]
            tset = set(test)
            for x in sorted(universe - tset):
                cand = test + [x]
                cand.sort()
                ok, _ = _check(cand)
                if ok:
                    test = cand
                    tset.add(x)
            if len(test) > len(seq):
                seq = test
                used = tset
                if len(seq) > best_len:
                    best_len = len(seq)
                    best = seq[:]
                    last_imp = time.time()
                stall = 0
            else:
                if stall > 160:
                    # forced perturbation: drop a few random elements
                    if len(seq) > 3:
                        for x in rng.sample(seq, 3):
                            seq.remove(x)
                            used.remove(x)
                    stall = 0
    return best

# ---------------------------------------------------------------------------
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin before the 60s eval / 90s kill

    rng = random.Random(20240517)

    # 1) Build a diverse pool of starting sequences.
    starts = []
    starts.append(_greedy_smallest())
    if time.time() > deadline:
        return max(starts, key=len)
    starts.append(_greedy_largest())
    if time.time() > deadline:
        return max(starts, key=len)

    # several randomized greedy starts (different seeds / biases)
    for seed, bias in [(42, 0.75), (123, 0.6), (7, 0.85), (99, 0.5), (314, 0.9)]:
        if time.time() > deadline:
            break
        r = random.Random(seed)
        starts.append(_greedy_random(r, bias))
    # a couple of shuffled-insertion starts
    for seed in [(5, 0.0), (11, 0.0)]:
        if time.time() > deadline:
            break
        r = random.Random(seed[0])
        starts.append(_greedy_shuffled(r))

    # 2) Refine each start with local search, tracking the global best.
    best = max(starts, key=len)
    # sort starts by length descending so the strongest starts get the most time
    starts.sort(key=len, reverse=True)

    for s in starts:
        if time.time() > deadline:
            break
        # give each start a sub-deadline but let the loop respect the global one
        refined = _local_search(s, deadline, rng)
        if len(refined) > len(best):
            best = refined

    # 3) Final greedy polish on the overall best to squeeze any remaining room.
    if time.time() < deadline:
        test = best[:]
        tset = set(test)
        for x in sorted(set(range(1, 257)) - tset):
            if time.time() > deadline:
                break
            cand = test + [x]
            cand.sort()
            ok, _ = _check(cand)
            if ok:
                test = cand
                tset.add(x)
        if len(test) > len(best):
            best = test

    return best

if __name__ == "__main__":
    res = construct()
    # print(len(res), res)
    pass
