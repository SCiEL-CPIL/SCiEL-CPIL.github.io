import time
import random

# LCM of 1..22
_LCM = 232792560

def _is_valid(subset):
    """Check if all subset sums of reciprocals are distinct (via LCM scaling)."""
    if not subset:
        return True
    weights = [_LCM // i for i in subset]
    sums = {0}
    for w in weights:
        new_sums = {s + w for s in sums}
        if new_sums & sums:
            return False
        sums |= new_sums
    return True

def _is_valid_with_sums(subset):
    """Return (valid, sums_set) for incremental use."""
    weights = [_LCM // i for i in subset]
    sums = {0}
    for w in weights:
        new_sums = {s + w for s in sums}
        if new_sums & sums:
            return False, sums
        sums |= new_sums
    return True, sums

def _find_conflict(subset):
    """
    Find an element whose removal helps, by detecting a collision.
    Returns (i, j) indices or None. We detect: does there exist w such that
    adding it creates overlap? Instead, find any pair of distinct subsets with
    same sum. We do this by building sums and tracking which elements.
    For repair, we just need to know validity; we remove a random element if invalid.
    """
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin (evaluator kills at 90s)

    best_subset = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_subset)

    all_elems = list(range(1, 23))
    weights = {i: _LCM // i for i in all_elems}

    def time_ok():
        return time.time() < deadline

    # ---- Strategy A: Greedy with random restarts (incremental) ----
    for _ in range(5000):
        if not time_ok():
            break
        candidates = list(all_elems)
        random.shuffle(candidates)
        current = []
        sums = {0}
        for c in candidates:
            w = weights[c]
            new_sums = {s + w for s in sums}
            if new_sums & sums:
                continue
            sums |= new_sums
            current.append(c)
        if len(current) > best_size:
            best_size = len(current)
            best_subset = list(current)

    # ---- Strategy B: Repair from a large random set ----
    for _ in range(3000):
        if not time_ok():
            break
        perm = list(all_elems)
        random.shuffle(perm)
        # Start with a random subset of size ~18
        current = perm[:18]
        # Repair: while invalid, remove a random element
        for _ in range(30):
            if _is_valid(current):
                break
            if len(current) <= best_size:
                break
            # Remove a random element
            idx = random.randrange(len(current))
            current.pop(idx)
        if len(current) > best_size:
            best_size = len(current)
            best_subset = list(current)

    # ---- Strategy C: Local improvement (add missing, swap) ----
    def try_improve(subset):
        if not time_ok():
            return subset
        cur = list(subset)
        cur_set = set(cur)
        improved = True
        while improved and time_ok():
            improved = False
            # Try adding each missing element
            for c in all_elems:
                if c in cur_set:
                    continue
                if _is_valid(cur + [c]):
                    cur.append(c)
                    cur_set.add(c)
                    improved = True
                    break
            # Try swaps: remove one, add one
            if not improved:
                for i in range(len(cur)):
                    if not time_ok():
                        break
                    for c in all_elems:
                        if c in cur_set:
                            continue
                        candidate = cur[:i] + cur[i+1:] + [c]
                        if _is_valid(candidate):
                            cur = candidate
                            cur_set = set(cur)
                            improved = True
                            break
                    if improved:
                        break
        return cur

    # Improve the best found
    improved_best = try_improve(best_subset)
    if len(improved_best) > best_size:
        best_size = len(improved_best)
        best_subset = improved_best

    # ---- Strategy D: More greedy restarts if time remains ----
    for _ in range(5000):
        if not time_ok():
            break
        candidates = list(all_elems)
        random.shuffle(candidates)
        current = []
        sums = {0}
        for c in candidates:
            w = weights[c]
            new_sums = {s + w for s in sums}
            if new_sums & sums:
                continue
            sums |= new_sums
            current.append(c)
        if len(current) > best_size:
            best_size = len(current)
            best_subset = list(current)
            # Try to improve
            improved = try_improve(best_subset)
            if len(improved) > best_size:
                best_size = len(improved)
                best_subset = improved

    # Final polish: try to improve best one more time
    if time_ok():
        improved = try_improve(best_subset)
        if len(improved) > best_size:
            best_subset = improved

    # Ensure nonempty
    if not best_subset:
        best_subset = [1]

    return best_subset
