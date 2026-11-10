import time
import random

def _check_divisibility(sums_list):
    """Check if any two distinct sums have a divisibility relation."""
    sorted_sums = sorted(sums_list)
    n = len(sorted_sums)
    for i in range(n):
        a = sorted_sums[i]
        for j in range(i + 1, n):
            b = sorted_sums[j]
            if b % a == 0:
                return False
    return True

def _get_sums(elements):
    """Get all non-empty subset sums as a set."""
    n = len(elements)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sums.add(s)
    return sums

def _is_valid(elements):
    """Check validity."""
    n = len(elements)
    if n == 0:
        return True
    sums = _get_sums(elements)
    if len(sums) < (1 << n) - 1:
        return False
    return _check_divisibility(list(sums))

def _can_add_incremental(elements, new_elem, existing_sums):
    """Check if adding new_elem keeps validity."""
    x = new_elem
    if x in existing_sums:
        return False, set()
    for s1 in existing_sums:
        if s1 - x in existing_sums:
            return False, set()
    new_sums = set(existing_sums)
    new_sums.add(x)
    for s in existing_sums:
        new_sums.add(s + x)
    if not _check_divisibility(list(new_sums)):
        return False, set()
    return True, new_sums

def construct():
    start_time = time.time()
    deadline = start_time + 78
    all_candidates = list(range(1, 257))
    best = []
    seeds = [
        [31, 32, 34],
        [29, 31, 34],
        [30, 32, 35],
        [28, 31, 34],
        [31, 33, 35],
        [27, 30, 33],
        [26, 29, 32],
    ]
    for seed in seeds:
        if time.time() > deadline:
            break
        if not _is_valid(seed):
            continue
        if len(seed) > len(best):
            best = list(seed)
        existing_sums = _get_sums(seed)
        current = list(seed)
        for trial in range(150):
            if time.time() > deadline:
                break
            remaining = [c for c in all_candidates if c not in current]
            random.shuffle(remaining)
            cur = list(current)
            cur_sums = set(existing_sums)
            for c in remaining:
                if time.time() > deadline:
                    break
                ok, new_sums = _can_add_incremental(cur, c, cur_sums)
                if ok:
                    cur.append(c)
                    cur_sums = new_sums
            if len(cur) > len(best):
                best = cur
    for trial in range(800):
        if time.time() > deadline:
            break
        size = random.randint(2, 5)
        init = random.sample(all_candidates, size)
        if not _is_valid(init):
            continue
        if len(init) > len(best):
            best = list(init)
        existing_sums = _get_sums(init)
        cur = list(init)
        cur_sums = set(existing_sums)
        remaining = [c for c in all_candidates if c not in cur]
        random.shuffle(remaining)
        for c in remaining:
            if time.time() > deadline:
                break
            ok, new_sums = _can_add_incremental(cur, c, cur_sums)
            if ok:
                cur.append(c)
                cur_sums = new_sums
        if len(cur) > len(best):
            best = cur
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
