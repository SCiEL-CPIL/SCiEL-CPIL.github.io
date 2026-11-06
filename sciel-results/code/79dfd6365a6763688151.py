# EVOLVE-BLOCK-START
import time
import random

N = 256
DEADLINE = 78.0  # safety margin under the 90s kill


def _build_aps():
    aps = []
    for start in range(1, N + 1):
        for step in range(1, N + 1):
            L = (N - start) // step + 1
            idxs = tuple(start - 1 + k * step for k in range(L))
            aps.append((idxs, start, step))
    return aps


APS = _build_aps()
AP_CONTAIN = [[] for _ in range(N)]
for _i, (_idxs, _s, _st) in enumerate(APS):
    for _x in _idxs:
        AP_CONTAIN[_x].append(_i)
N_APS = len(APS)


def _init_state(signs):
    pref = [0] * (N + 1)
    for k in range(N):
        pref[k + 1] = pref[k] + signs[k]
    sums = [0] * N_APS
    for i, (idxs, _s, _st) in enumerate(APS):
        sums[i] = pref[idxs[-1] + 1] - pref[idxs[0]]
    # frequency of absolute sums
    freq = {}
    for v in sums:
        a = v if v >= 0 else -v
        freq[a] = freq.get(a, 0) + 1
    # global max
    m = 1
    for a in freq:
        if a > m:
            m = a
    return sums, freq, m


def _thue_morse():
    seq = []
    for i in range(N):
        c = bin(i).count("1") & 1
        seq.append(1 if c == 0 else -1)
    return seq


def _alternating():
    return [1 if (i % 2 == 0) else -1 for i in range(N)]


def _block_pattern(period):
    base = [1 if (i % (2 * period) < period) else -1 for i in range(period * 2)]
    return [base[i % (2 * period)] for i in range(N)]


def _random_seq(rng):
    return [rng.choice((-1, 1)) for _ in range(N)]


def _local_search(signs, sums, freq, cur_max, rng, t_end):
    """Incremental hill-climb with exact O(1) max tracking. Returns improved state."""
    cur_signs = signs[:]
    cur_sums = sums[:]
    cur_freq = dict(freq)
    best_signs = cur_signs[:]
    best_sums = cur_sums[:]
    best_freq = dict(cur_freq)
    best_max = cur_max

    while time.time() < t_end:
        pos = rng.randrange(N)
        old = cur_signs[pos]
        delta = -2 * old
        # For each affected AP, update freq and sums
        affected = AP_CONTAIN[pos]
        new_max = cur_max
        # We'll tentatively apply; if rejected, revert.
        for i in affected:
            oldv = cur_sums[i]
            oa = oldv if oldv >= 0 else -oldv
            cur_freq[oa] -= 1
            if cur_freq[oa] == 0:
                del cur_freq[oa]
            nv = oldv + delta
            cur_sums[i] = nv
            na = nv if nv >= 0 else -nv
            cur_freq[na] = cur_freq.get(na, 0) + 1
            if na > new_max:
                new_max = na
        cur_signs[pos] = -old
        # Determine the true current max: it's the largest key with freq>0,
        # but if the old max key was removed, we may need to lower it.
        # Since we only removed `oa` values and added `na`, the new max is
        # max(new_max, (old max if still present)).
        # Find new global max efficiently:
        # If cur_max still in cur_freq, new_max = max(new_max, cur_max)
        # else we need the max key in cur_freq.
        if cur_max in cur_freq:
            if cur_max > new_max:
                new_max = cur_max
        else:
            # find max key
            # freq keys are small ints; scanning keys is fine (bounded by N)
            # but could be many; use max over keys
            mk = 1
            for k in cur_freq:
                if k > mk:
                    mk = k
            if mk > new_max:
                new_max = mk
        cur_max = new_max

        # Acceptance: keep if improved, else revert (pure hill-climb) with
        # occasional random restart to best to escape plateaus.
        if cur_max < best_max:
            best_max = cur_max
            best_signs = cur_signs[:]
            best_sums = cur_sums[:]
            best_freq = dict(cur_freq)
        elif rng.random() < 0.03:
            # revert to best
            cur_signs = best_signs[:]
            cur_sums = best_sums[:]
            cur_freq = dict(best_freq)
            cur_max = best_max
        else:
            # revert this flip
            for i in affected:
                nv = cur_sums[i]
                na = nv if nv >= 0 else -nv
                cur_freq[na] -= 1
                if cur_freq[na] == 0:
                    del cur_freq[na]
                ov = nv - delta
                cur_sums[i] = ov
                oa = ov if ov >= 0 else -ov
                cur_freq[oa] = cur_freq.get(oa, 0) + 1
            cur_signs[pos] = old
            # recompute cur_max
            if cur_max in cur_freq:
                pass
            else:
                mk = 1
                for k in cur_freq:
                    if k > mk:
                        mk = k
                cur_max = mk
    return best_signs, best_sums, best_freq, best_max


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    rng = random.Random(777)

    seeds = [_thue_morse(), _alternating()]
    for p in (2, 3, 4, 5, 8, 16, 32, 64):
        seeds.append(_block_pattern(p))
    for _ in range(8):
        seeds.append(_random_seq(rng))

    best_signs = None
    best_max = 10**9

    n = len(seeds)
    per_seed = (DEADLINE - 3.0) / n

    for s in seeds:
        if time.time() > deadline - 2.5:
            break
        signs = s[:]
        sums, freq, m = _init_state(signs)
        t_slice = min(deadline - 2.0, time.time() + per_seed)
        signs, sums, freq, m = _local_search(signs, sums, freq, m, rng, t_slice)
        if m < best_max:
            best_max = m
            best_signs = signs[:]

    # Final polish
    if time.time() < deadline - 1.0 and best_signs is not None:
        signs = best_signs[:]
        sums, freq, m = _init_state(signs)
        signs, sums, freq, m = _local_search(signs, sums, freq, m,
                                             random.Random(4242), deadline - 0.5)
        if m < best_max:
            best_max = m
            best_signs = signs[:]

    if best_signs is None:
        best_signs = _thue_morse()
    return [1 if v >= 0 else -1 for v in best_signs]
# EVOLVE-BLOCK-END
