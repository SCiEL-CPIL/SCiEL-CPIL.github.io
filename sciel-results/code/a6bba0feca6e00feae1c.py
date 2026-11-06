"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
_MAX = 512
_N = 32
_DEADLINE = 145.0  # safety margin before 150s kill

def _score_set(arr):
    """Count distinct d>0 such that some a,a+d,a+2d all in set. arr is a set of ints."""
    s = arr
    # precompute membership as a boolean list for speed
    present = [False] * (_MAX + 1)
    for x in s:
        present[x] = True
    count = 0
    # d max: need a+2d <= 512 and a>=1, but also must be in set; bound d by (max-min)/2
    lo = min(s)
    hi = max(s)
    maxd = (hi - lo) // 2
    for d in range(1, maxd + 1):
        # check existence of a in s with a+d in s and a+2d in s
        # iterate over elements as potential a
        for a in s:
            b = a + d
            if b > hi:
                continue
            if present[b]:
                c = b + d
                if c <= hi and present[c]:
                    count += 1
                    break
    return count

def _score_mask(mask):
    """mask is a bool numpy array of length _MAX+1 (index 0 unused)."""
    present = mask
    # elements present
    elems = np.nonzero(present[1:])[0] + 1
    if elems.size < 3:
        return 0
    lo = int(elems[0])
    hi = int(elems[-1])
    maxd = (hi - lo) // 2
    count = 0
    elist = elems.tolist()
    for d in range(1, maxd + 1):
        found = False
        for a in elist:
            b = a + d
            if b > hi:
                continue
            if present[b]:
                c = b + d
                if c <= hi and present[c]:
                    found = True
                    break
        if found:
            count += 1
    return count

def _random_set(rng):
    return set(rng.sample(range(1, _MAX + 1), _N))

def _local_search(start, rng, deadline):
    """Greedy + local search on a set. Returns best set and score."""
    cur = set(start)
    cur_score = _score_set(cur)
    best = set(cur)
    best_score = cur_score
    # present mask for speed
    present = [False] * (_MAX + 1)
    for x in cur:
        present[x] = True

    # candidate universe
    universe = list(range(1, _MAX + 1))
    in_set = set(cur)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # try single-swap moves: remove one, add one
        # to limit cost, sample a random subset of elements to remove and random candidates to add
        elems = list(in_set)
        rng.shuffle(elems)
        # candidates not in set
        out_set = [x for x in universe if x not in in_set]
        rng.shuffle(out_set)
        # limit number of attempts per pass
        attempts = 0
        max_attempts = 2000
        for rem in elems:
            if time.time() >= deadline or attempts >= max_attempts:
                break
            for add in out_set:
                if time.time() >= deadline or attempts >= max_attempts:
                    break
                attempts += 1
                # tentatively swap
                present[rem] = False
                present[add] = True
                in_set.remove(rem)
                in_set.add(add)
                # recompute score with mask
                sc = _score_mask(np.array(present, dtype=bool))
                if sc > cur_score:
                    cur_score = sc
                    cur = set(in_set)
                    if sc > best_score:
                        best_score = sc
                        best = set(in_set)
                    improved = True
                    # restart candidate lists
                    out_set = [x for x in universe if x not in in_set]
                    rng.shuffle(out_set)
                    elems = list(in_set)
                    rng.shuffle(elems)
                    attempts = 0
                    break
                else:
                    # revert
                    present[add] = False
                    present[rem] = True
                    in_set.remove(add)
                    in_set.add(rem)
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    rng = random.Random(12345)

    # Start from a structured seed: consecutive block is weak (score low).
    # Use a few diverse initial seeds and take the best after local search.
    seeds = []
    # seed 1: consecutive
    seeds.append(set(range(1, _N + 1)))
    # seed 2: random
    seeds.append(_random_set(rng))
    # seed 3: random
    seeds.append(_random_set(rng))
    # seed 4: low-density structured (every other)
    seeds.append(set(range(1, 2 * _N + 1, 2)))

    best = set(range(1, _N + 1))
    best_score = _score_set(best)

    for s in seeds:
        if time.time() >= deadline:
            break
        b, sc = _local_search(s, rng, deadline)
        if sc > best_score:
            best_score = sc
            best = b

    # If time remains, do extra random restarts
    while time.time() < deadline - 5:
        s = _random_set(rng)
        b, sc = _local_search(s, rng, deadline)
        if sc > best_score:
            best_score = sc
            best = b

    return sorted(best)
# EVOLVE-BLOCK-END

