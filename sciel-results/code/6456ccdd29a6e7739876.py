import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 84  # safety margin before 90s kill

    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1  # 129
    MAX_VAL = 128

    def calc_reps(A):
        """Unordered-pair representation counts for sums N_LOW..N_HIGH via convolution."""
        ind = np.zeros(MAX_VAL + 1, dtype=np.int32)
        for a in A:
            ind[a] = 1
        conv = np.convolve(ind, ind, mode='full')  # ordered-pair counts per sum
        reps = np.empty(RANGE_SIZE, dtype=np.int32)
        for idx, s in enumerate(range(N_LOW, N_HIGH + 1)):
            ordered = conv[s]
            diag = 1 if (s % 2 == 0 and ind[s // 2] == 1) else 0
            reps[idx] = (ordered + diag) // 2
        return reps

    def calc_reps_delta(A, reps, x_add, x_rem):
        """Apply add of x_add and/or removal of x_rem to existing reps (in-place).
        x_add / x_rem may be None."""
        ind = np.zeros(MAX_VAL + 1, dtype=np.int32)
        for a in A:
            ind[a] = 1
        # removal first
        if x_rem is not None:
            xr = x_rem
            for s in range(N_LOW, N_HIGH + 1):
                other = s - xr
                if 0 <= other <= MAX_VAL:
                    if other == xr:
                        delta = 1
                    else:
                        delta = 2 * int(ind[other])
                    if delta:
                        reps[s - N_LOW] -= delta
        # add second
        if x_add is not None:
            xa = x_add
            for s in range(N_LOW, N_HIGH + 1):
                other = s - xa
                if 0 <= other <= MAX_VAL:
                    if other == xa:
                        delta = 1
                    else:
                        delta = 2 * int(ind[other])
                    if delta:
                        reps[s - N_LOW] += delta
        return reps

    def is_covered(reps):
        return bool(np.all(reps > 0))

    def get_peak(reps):
        return int(np.max(reps))

    def sumsq(reps):
        return int(np.dot(reps, reps))

    def greedy_build(seed_rng, cap=80):
        """Greedy add elements to cover [N_LOW,N_HIGH]; prefer balancing reps."""
        A = {0, 128}
        reps = calc_reps(A)
        uncovered = set(range(N_LOW, N_HIGH + 1))
        for s in range(N_LOW, N_HIGH + 1):
            if reps[s - N_LOW] > 0:
                uncovered.discard(s)
        while len(uncovered) > 0 and len(A) < cap:
            best_x = None
            best_key = None
            candidates = list(range(0, MAX_VAL + 1))
            seed_rng.shuffle(candidates)
            for x in candidates:
                if x in A:
                    continue
                new_reps = calc_reps_delta(A, reps.copy(), x, None)
                # newly covered count
                new_cov = int(np.sum((reps == 0) & (new_reps > 0)))
                if is_covered(new_reps):
                    key = (10**9, -sumsq(new_reps), -get_peak(new_reps))
                else:
                    key = (new_cov, -sumsq(new_reps), -get_peak(new_reps))
                if best_key is None or key > best_key:
                    best_key = key
                    best_x = x
                    best_reps = new_reps
            if best_x is None:
                break
            A.add(best_x)
            reps = best_reps
            uncovered = {s for s in range(N_LOW, N_HIGH + 1) if reps[s - N_LOW] == 0}
        return A, reps

    # ---------- build diverse valid seeds ----------
    seeds = []
    # deterministic greedy
    A0, r0 = greedy_build(random.Random(1234))
    if is_covered(r0):
        seeds.append((A0, r0.copy()))
    # a few randomised greedy seeds
    for sd in (7, 2024, 99, 31337, 555):
        A, r = greedy_build(random.Random(sd))
        if is_covered(r):
            seeds.append((A, r.copy()))
    # fallback guaranteed-valid seed
    if not seeds:
        A = set(range(64, 129))
        seeds.append((A, calc_reps(A)))

    best_A = None
    best_peak = float('inf')
    best_sumsq = float('inf')

    def consider(A, reps):
        nonlocal best_A, best_peak, best_sumsq
        if not is_covered(reps):
            return
        pk = get_peak(reps)
        sq = sumsq(reps)
        if pk < best_peak or (pk == best_peak and sq < best_sumsq):
            best_peak = pk
            best_sumsq = sq
            best_A = list(A)

    for A, r in seeds:
        consider(A, r)

    # ---------- per-seed local search: repair + sum-of-squares minimisation ----------
    for A0, r0 in seeds:
        if time.time() > deadline - 6:
            break
        A = set(A0)
        reps = r0.copy()
        cur_peak = get_peak(reps)
        cur_sq = sumsq(reps)
        rng = random.Random(int(time.time() * 1000) % (2**31))

        # repair pass: ensure coverage, then remove extras
        def repair():
            nonlocal A, reps, cur_peak, cur_sq
            while time.time() < deadline - 6:
                uncovered = [s for s in range(N_LOW, N_HIGH + 1) if reps[s - N_LOW] == 0]
                if not uncovered:
                    break
                # find a single add that covers the most uncovered
                best_x = None
                best_cov = -1
                best_r = None
                cands = list(range(0, MAX_VAL + 1))
                rng.shuffle(cands)
                for x in cands:
                    if x in A:
                        continue
                    nr = calc_reps_delta(A, reps.copy(), x, None)
                    cov = int(np.sum((reps == 0) & (nr > 0)))
                    if cov > best_cov:
                        best_cov = cov
                        best_x = x
                        best_r = nr
                if best_x is None or best_cov <= 0:
                    break
                A.add(best_x)
                reps = best_r
                cur_peak = get_peak(reps)
                cur_sq = sumsq(reps)
            # removal pass
            changed = True
            while changed and time.time() < deadline - 6:
                changed = False
                for x in list(A):
                    if len(A) <= 1:
                        break
                    nr = calc_reps_delta(A, reps.copy(), None, x)
                    if is_covered(nr):
                        A.discard(x)
                        reps = nr
                        cur_peak = get_peak(reps)
                        cur_sq = sumsq(reps)
                        changed = True
                        break
            consider(A, reps)

        repair()

        # main loop: minimise sum-of-squares via add/remove/swap moves
        while time.time() < deadline - 6:
            move = rng.random()
            if move < 0.45:
                # swap: remove r, add a
                r = rng.choice(list(A))
                a = rng.randint(0, MAX_VAL)
                if a == r:
                    continue
                nr = calc_reps_delta(A, reps.copy(), a, r)
                if is_covered(nr):
                    nsq = sumsq(nr)
                    npeak = get_peak(nr)
                    # accept if improves sumsq, or equal peak and better sumsq, or small peak risk
                    if nsq < cur_sq or (npeak <= cur_peak and nsq < cur_sq):
                        A.discard(r); A.add(a)
                        reps = nr; cur_sq = nsq; cur_peak = npeak
                        consider(A, reps)
            elif move < 0.70:
                # remove
                r = rng.choice(list(A))
                if len(A) <= 1:
                    continue
                nr = calc_reps_delta(A, reps.copy(), None, r)
                if is_covered(nr):
                    nsq = sumsq(nr)
                    npeak = get_peak(nr)
                    if nsq < cur_sq or (npeak <= cur_peak and nsq < cur_sq):
                        A.discard(r)
                        reps = nr; cur_sq = nsq; cur_peak = npeak
                        consider(A, reps)
            else:
                # add
                a = rng.randint(0, MAX_VAL)
                if a in A:
                    continue
                nr = calc_reps_delta(A, reps.copy(), a, None)
                if is_covered(nr):
                    nsq = sumsq(nr)
                    npeak = get_peak(nr)
                    if nsq < cur_sq or (npeak <= cur_peak and nsq < cur_sq):
                        A.add(a)
                        reps = nr; cur_sq = nsq; cur_peak = npeak
                        consider(A, reps)

        # targeted peak-breaking: attack the argmax sums
        if time.time() < deadline - 6:
            for _ in range(30):
                if time.time() > deadline - 6:
                    break
                if not is_covered(reps):
                    repair()
                    if not is_covered(reps):
                        break
                peak = get_peak(reps)
                # find sums at the peak
                peak_sums = [N_LOW + i for i in range(RANGE_SIZE) if reps[i] == peak]
                # try to reduce one peak sum by removing an element that contributes to it,
                # or swapping an element to a less crowded spot
                s = rng.choice(peak_sums)
                # elements x in A such that x and s-x in A (contribute to sum s)
                contrib = [x for x in A if (s - x) in A]
                if not contrib:
                    continue
                improved = False
                for x in rng.sample(contrib, k=min(len(contrib), 8)):
                    # try removing x
                    nr = calc_reps_delta(A, reps.copy(), None, x)
                    if is_covered(nr) and get_peak(nr) < peak:
                        A.discard(x); reps = nr
                        cur_peak = get_peak(reps); cur_sq = sumsq(reps)
                        improved = True
                        break
                    # try swapping x to a random other value
                    y = rng.randint(0, MAX_VAL)
                    if y == x or y in A:
                        continue
                    nr = calc_reps_delta(A, reps.copy(), y, x)
                    if is_covered(nr) and get_peak(nr) < peak:
                        A.discard(x); A.add(y); reps = nr
                        cur_peak = get_peak(reps); cur_sq = sumsq(reps)
                        improved = True
                        break
                consider(A, reps)
                if not improved:
                    # nudge: random swap that keeps coverage and lowers sumsq
                    x = rng.choice(list(A))
                    y = rng.randint(0, MAX_VAL)
                    if y == x or y in A:
                        continue
                    nr = calc_reps_delta(A, reps.copy(), y, x)
                    if is_covered(nr) and sumsq(nr) < cur_sq:
                        A.discard(x); A.add(y); reps = nr
                        cur_peak = get_peak(reps); cur_sq = sumsq(reps)
                        consider(A, reps)

    if best_A is None:
        A = set(range(64, 129))
        return list(A)
    return best_A
