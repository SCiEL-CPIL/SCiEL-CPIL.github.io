"""Construct large 4-AP-free subsets of [1,512] via local search with conflict checks."""
import time
import random

_N = 512


def _has_4ap_with_new_element(x, present):
    """Check if adding x to `present` creates a nonconstant 4-AP."""
    for d in range(1, _N + 1):
        # x as 1st term: x, x+d, x+2d, x+3d
        if x + 3 * d <= _N:
            if (x + d) in present and (x + 2 * d) in present and (x + 3 * d) in present:
                return True
        # x as 2nd term: x-d, x, x+d, x+2d
        if x - d >= 1 and x + 2 * d <= _N:
            if (x - d) in present and (x + d) in present and (x + 2 * d) in present:
                return True
        # x as 3rd term: x-2d, x-d, x, x+d
        if x - 2 * d >= 1 and x + d <= _N:
            if (x - 2 * d) in present and (x - d) in present and (x + d) in present:
                return True
        # x as 4th term: x-3d, x-2d, x-d, x
        if x - 3 * d >= 1:
            if (x - 3 * d) in present and (x - 2 * d) in present and (x - d) in present:
                return True
    return False


def _valid_set(present):
    for x in present:
        for d in range(1, _N + 1):
            if x + 3 * d <= _N:
                if (x + d) in present and (x + 2 * d) in present and (x + 3 * d) in present:
                    return False
            if x - d >= 1 and x + 2 * d <= _N:
                if (x - d) in present and (x + d) in present and (x + 2 * d) in present:
                    return False
            if x - 2 * d >= 1 and x + d <= _N:
                if (x - 2 * d) in present and (x - d) in present and (x + d) in present:
                    return False
            if x - 3 * d >= 1:
                if (x - 3 * d) in present and (x - 2 * d) in present and (x - d) in present:
                    return False
    return True


def _greedy_init(rng):
    order = list(range(1, _N + 1))
    rng.shuffle(order)
    present = set()
    for x in order:
        if not _has_4ap_with_new_element(x, present):
            present.add(x)
    return present


def _local_search(present, rng, deadline):
    n = len(present)
    all_elems = set(range(1, _N + 1))
    temp = 1.0

    while time.time() < deadline:
        absent = all_elems - present
        if not absent:
            break

        # Try to add an element
        candidates = list(absent)
        rng.shuffle(candidates)
        added = False
        for y in candidates:
            if not _has_4ap_with_new_element(y, present):
                present.add(y)
                n += 1
                added = True
                break

        if added:
            continue

        # Can't add, try swaps
        if n == 0:
            break

        present_list = sorted(present)
        swap_done = False
        for _ in range(20):
            idx = rng.randrange(n)
            x = present_list[idx]
            y = rng.choice(tuple(absent))
            new_present = present - {x} | {y}
            if not _has_4ap_with_new_element(y, new_present):
                # Check if we can add from new state
                new_absent = all_elems - new_present
                if new_absent:
                    nc = list(new_absent)
                    rng.shuffle(nc)
                    for z in nc:
                        if not _has_4ap_with_new_element(z, new_present):
                            new_present.add(z)
                            present = new_present
                            n = len(present)
                            swap_done = True
                            break
                if swap_done:
                    break
                if rng.random() < temp:
                    present = new_present
                    n = len(present)
                    swap_done = True
                    break

        temp = max(0.01, temp * 0.999)

    return present


def construct():
    start_time = time.time()
    deadline = start_time + 90 * 0.85

    best = set()
    best_size = 0

    restart = 0
    while time.time() < deadline:
        restart += 1
        rng = random.Random(1000 + restart * 13)

        present = _greedy_init(rng)
        if len(present) > best_size:
            best = set(present)
            best_size = len(present)

        present = _local_search(present, rng, deadline)
        if len(present) > best_size:
            best = set(present)
            best_size = len(present)

    if best and _valid_set(best):
        return sorted(best)
    return [1]


if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
