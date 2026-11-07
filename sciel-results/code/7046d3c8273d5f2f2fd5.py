"""Rewritten program: randomized greedy search for divisibility-incomparable subset sums."""
import random
import time

# EVOLVE-BLOCK-START

def _is_valid(nums):
    """Exact check: all nonempty subset sums are pairwise non-dividing."""
    n = len(nums)
    sums = [0]
    for x in nums:
        sums += [s + x for s in sums]
    # drop the empty sum 0
    ss = [s for s in sums if s != 0]
    ss = sorted(set(ss))
    # no two distinct values may divide one another
    for i in range(len(ss)):
        a = ss[i]
        for j in range(i + 1, len(ss)):
            b = ss[j]
            if b % a == 0:
                return False
    return True

def _greedy(seed, deadline, max_size=30):
    """Build a set by greedily adding elements that keep validity."""
    rng = random.Random(seed)
    candidates = list(range(1, 257))
    rng.shuffle(candidates)

    chosen = []
    sums = [0]  # all subset sums of chosen (includes empty)
    for x in candidates:
        if len(chosen) >= max_size:
            break
        if time.time() > deadline:
            break
        new_sums = [s + x for s in sums]
        # candidate multiset of sums = sums (old) + new_sums (new), drop the 0
        all_sums = [s for s in sums if s != 0] + new_sums
        all_sums = sorted(set(all_sums))
        ok = True
        for i in range(len(all_sums)):
            a = all_sums[i]
            for j in range(i + 1, len(all_sums)):
                b = all_sums[j]
                if b % a == 0:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            chosen.append(x)
            sums = all_sums  # reuse
    return chosen

def construct():
    start = time.time()
    deadline = start + 50.0  # safety margin before 60s
    best = [31, 32, 34]
    best_len = len(best)
    seed = 0
    while time.time() < deadline:
        seed += 1
        cur = _greedy(seed, deadline)
        if len(cur) > best_len:
            # verify exact validity before accepting
            if _is_valid(cur):
                best = cur
                best_len = len(cur)
    return list(best)

# EVOLVE-BLOCK-END
