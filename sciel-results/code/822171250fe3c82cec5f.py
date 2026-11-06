"""Improved B_3[1] construction: greedy with multiple random restarts."""
# EVOLVE-BLOCK-START
import time
import random

def _greedy_build(order, deadline):
    """Build a B_3[1] set by greedily adding elements in the given order."""
    selected = []
    existing_sums = set()

    def try_add(x):
        nonlocal selected, existing_sums
        new_sums = set()

        # (x, x, x)
        s = 3 * x
        if s in existing_sums or s in new_sums:
            return False
        new_sums.add(s)

        # (x, x, y) for y in selected
        for y in selected:
            s = 2 * x + y
            if s in existing_sums or s in new_sums:
                return False
            new_sums.add(s)

        # (x, y, y) for y in selected
        for y in selected:
            s = x + 2 * y
            if s in existing_sums or s in new_sums:
                return False
            new_sums.add(s)

        # (x, y, z) for y <= z in selected
        for i, y in enumerate(selected):
            for z in selected[i:]:
                s = x + y + z
                if s in existing_sums or s in new_sums:
                    return False
                new_sums.add(s)

        existing_sums.update(new_sums)
        selected.append(x)
        return True

    for x in order:
        if time.time() > deadline:
            break
        if len(selected) >= 80:
            break
        try_add(x)

    return selected

def construct():
    deadline = time.time() + 55.0
    best = []

    # Strategy 1: increasing order
    result = _greedy_build(range(1, 4097), deadline)
    if len(result) > len(best):
        best = result

    # Strategy 2: multiple random restarts
    candidates = list(range(1, 4097))
    num_restarts = 50
    for _ in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
        random.shuffle(candidates)
        result = _greedy_build(candidates, deadline)
        if len(result) > len(best):
            best = result

    return best
# EVOLVE-BLOCK-END
