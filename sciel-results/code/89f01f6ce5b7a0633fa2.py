"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import time


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _is_isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3


def _greedy_maximal(order, chosen, deadline):
    """Greedy maximal construction given an ordering of points."""
    for p in order:
        if time.time() > deadline:
            return None
        # Check if p can be added: no isosceles with any pair in chosen
        ok = True
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                if _is_isosceles(p, chosen[i], chosen[j]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            chosen.append(p)
    return chosen


def construct():
    deadline = time.time() + 280  # safety margin before 330s kill
    all_points = list(itertools.product(range(N), repeat=2))
    
    best = None
    best_size = float('inf')
    
    # First: deterministic row-major greedy
    row_major = list(itertools.product(range(N), repeat=2))
    result = _greedy_maximal(row_major, [], deadline)
    if result is not None:
        best = result
        best_size = len(result)
    
    # Randomized restarts with different orderings
    rng = random.Random(42)
    for trial in range(2000):
        if time.time() > deadline:
            break
        # Shuffle the order
        order = all_points[:]
        rng.shuffle(order)
        chosen = []
        result = _greedy_maximal(order, chosen, deadline)
        if result is None:
            break
        if len(result) < best_size:
            best = result
            best_size = len(result)
            # If we've found a very small set, we can be more aggressive
            if best_size <= 10:
                break
    
    return best if best is not None else []
# EVOLVE-BLOCK-END
