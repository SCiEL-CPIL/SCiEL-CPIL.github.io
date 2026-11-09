import time
import random
import numpy as np

N = 256
MAX_EACH = 64
DEADLINE = 85.0  # safety margin before 90s kill

# Precompute pair-symmetric indices (i <= j) for the sum-uniqueness check
_PAIR_IDX = np.array([(i, j) for i in range(N) for j in range(i, N)], dtype=np.int32)
_I = _PAIR_IDX[:, 0]
_J = _PAIR_IDX[:, 1]

def _sums_ok(mask):
    """True if all pair sums (including a+a) within the mask are unique."""
    elems = np.flatnonzero(mask)
    k = len(elems)
    if k <= 1:
        return True
    s = elems[_I] + elems[_J]
    return np.unique(s).size == s.size

def _diffs_mask(mask):
    """Bitmask (length 2*N-1) of signed nonzero differences; index = diff + (N-1)."""
    elems = np.flatnonzero(mask)
    k = len(elems)
    m = np.zeros(2 * N - 1, dtype=bool)
    if k <= 1:
        return m
    d = elems[:, None] - elems[None, :]
    # d is k x k; nonzero off-diagonal
    flat = d.ravel()
    flat = flat[flat != 0]
    if flat.size == 0:
        return m
    idx = flat + (N - 1)
    m[idx] = True
    return m

def _can_add(mask, x, sum_ok, diff_mask, other_diff_mask):
    """Check if adding element x (0-based) to this mask keeps constraints.
    sum_ok: precomputed bool whether current mask has unique sums (we need to check new sums vs existing).
    Actually we recompute: cheaper to just test the new sums against existing sums.
    For speed in SA we use a simpler full recheck on acceptance (see below)."""
    pass

def _full_valid(maskA, maskB):
    """Full validity check: both Sidon, disjoint nonzero diffs, sizes <= MAX_EACH, nonempty."""
    if maskA.sum() == 0 or maskB.sum() == 0:
        return False
    if maskA.sum() > MAX_EACH or maskB.sum() > MAX_EACH:
        return False
    if not _sums_ok(maskA):
        return False
    if not _sums_ok(maskB):
        return False
    dA = _diffs_mask(maskA)
    dB = _diffs_mask(maskB)
    if dA & dB:
        return False
    return True

def _score(maskA, maskB):
    """Objective: |A| + |B|, but only if valid; else heavily penalized."""
    sa = maskA.sum()
    sb = maskB.sum()
    if sa > MAX_EACH or sb > MAX_EACH:
        return -(sa + sb) * 10
    if not _sums_ok(maskA) or not _sums_ok(maskB):
        return -(sa + sb) * 5
    dA = _diffs_mask(maskA)
    dB = _diffs_mask(maskB)
    overlap = (dA & dB).sum()
    if overlap > 0:
        return -(sa + sb) * 5 - overlap
    if sa == 0 or sb == 0:
        return 0
    return sa + sb

def _initial_random():
    """Random feasible-ish start: pick random small sets, then repair by removal."""
    maskA = np.zeros(N, dtype=bool)
    maskB = np.zeros(N, dtype=bool)
    # Seed with a few random elements
    for _ in range(10):
        x = random.randint(0, N - 1)
        maskA[x] = True
    for _ in range(10):
        x = random.randint(0, N - 1)
        maskB[x] = True
    # Repair: remove elements until valid
    for _ in range(200):
        if _full_valid(maskA, maskB):
            break
        # Remove a random element from the larger or invalid set
        if not _sums_ok(maskA):
            elems = np.flatnonzero(maskA)
            if len(elems) > 0:
                maskA[elems[random.randint(0, len(elems) - 1)]] = False
        elif not _sums_ok(maskB):
            elems = np.flatnonzero(maskB)
            if len(elems) > 0:
                maskB[elems[random.randint(0, len(elems) - 1)]] = False
        else:
            dA = _diffs_mask(maskA)
            dB = _diffs_mask(maskB)
            if dA & dB:
                # Remove from A or B randomly
                if random.random() < 0.5:
                    elems = np.flatnonzero(maskA)
                    if len(elems) > 0:
                        maskA[elems[random.randint(0, len(elems) - 1)]] = False
                else:
                    elems = np.flatnonzero(maskB)
                    if len(elems) > 0:
                        maskB[elems[random.randint(0, len(elems) - 1)]] = False
    return maskA, maskB

def _greedy_seed():
    """Fast greedy to get a decent starting point."""
    maskA = np.zeros(N, dtype=bool)
    maskB = np.zeros(N, dtype=bool)
    sumA = set()
    sumB = set()
    diffA = set()
    diffB = set()

    def can_add(x, elems, sums, diffs, other_diffs):
        # sums is set of pair sums for elems
        for a in elems:
            s = x + a
            if s in sums:
                return False
        s2 = x + x
        if s2 in sums:
            return False
        new_diffs = set()
        for a in elems:
            d = x - a
            new_diffs.add(d)
            new_diffs.add(-d)
        if new_diffs & diffs:
            return False
        if new_diffs & other_diffs:
            return False
        return True

    elemsA = []
    elemsB = []
    for x in range(N):
        if len(elemsA) < MAX_EACH and can_add(x, elemsA, sumA, diffA, diffB):
            for a in elemsA:
                sumA.add(x + a)
            sumA.add(x + x)
            for a in elemsA:
                d = x - a
                diffA.add(d)
                diffA.add(-d)
            elemsA.append(x)
            maskA[x] = True
        if len(elemsB) < MAX_EACH and can_add(x, elemsB, sumB, diffB, diffA):
            for b in elemsB:
                sumB.add(x + b)
            sumB.add(x + x)
            for b in elemsB:
                d = x - b
                diffB.add(d)
                diffB.add(-d)
            elemsB.append(x)
            maskB[x] = True
    return maskA, maskB

def _sa_run(maskA, maskB, deadline):
    """Simulated annealing on (maskA, maskB)."""
    t0 = time.time()
    curA = maskA.copy()
    curB = maskB.copy()
    cur_score = _score(curA, curB)
    bestA = curA.copy()
    bestB = curB.copy()
    best_score = cur_score

    T = 10.0
    T_min = 0.01
    # Estimate steps: each step is cheap (full recheck of sums/diffs is O(k^2) with k<=64)
    # We'll just loop until deadline.
    no_improve = 0
    max_no_improve = 2000

    while time.time() < deadline:
        # Choose move type
        r = random.random()
        newA = curA.copy()
        newB = curB.copy()

        if r < 0.4:
            # Add to A
            if curA.sum() < MAX_EACH:
                # Pick a random x not in A
                candidates = np.flatnonzero(~curA)
                if len(candidates) > 0:
                    x = candidates[random.randint(0, len(candidates) - 1)]
                    newA[x] = True
        elif r < 0.7:
            # Add to B
            if curB.sum() < MAX_EACH:
                candidates = np.flatnonzero(~curB)
                if len(candidates) > 0:
                    x = candidates[random.randint(0, len(candidates) - 1)]
                    newB[x] = True
        elif r < 0.85:
            # Remove from A
            if curA.sum() > 1:
                elems = np.flatnonzero(curA)
                x = elems[random.randint(0, len(elems) - 1)]
                newA[x] = False
        else:
            # Remove from B
            if curB.sum() > 1:
                elems = np.flatnonzero(curB)
                x = elems[random.randint(0, len(elems) - 1)]
                newB[x] = False

        new_score = _score(newA, newB)
        delta = new_score - cur_score
        if delta >= 0 or random.random() < np.exp(delta / T):
            curA, curB = newA, newB
            cur_score = new_score
            if cur_score > best_score:
                best_score = cur_score
                bestA = curA.copy()
                bestB = curB.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        # Cool down
        T = max(T_min, T * 0.9995)

        if no_improve > max_no_improve:
            # Restart cooling
            T = 5.0
            no_improve = 0

    return bestA, bestB, best_score

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE

    # Phase 1: Greedy seed
    maskA, maskB = _greedy_seed()
    bestA, bestB = maskA.copy(), maskB.copy()
    best_score = _score(bestA, bestB)

    # Phase 2: SA from greedy seed
    if time.time() < deadline - 2.0:
        a, b, s = _sa_run(maskA, maskB, deadline - 2.0)
        if s > best_score:
            bestA, bestB, best_score = a, b, s

    # Phase 3: Multiple SA restarts from random seeds
    while time.time() < deadline - 1.0:
        maskA, maskB = _initial_random()
        # Quick greedy fill from random seed
        # (optional: just run SA)
        a, b, s = _sa_run(maskA, maskB, min(deadline - 1.0, time.time() + 5.0))
        if s > best_score:
            bestA, bestB, best_score = a, b, s

    # Convert to 1-based lists
    A = [int(x) + 1 for x in np.flatnonzero(bestA)]
    B = [int(x) + 1 for x in np.flatnonzero(bestB)]

    # Ensure nonempty
    if not A:
        A = [1]
    if not B:
        B = [2]

    return [A, B]
