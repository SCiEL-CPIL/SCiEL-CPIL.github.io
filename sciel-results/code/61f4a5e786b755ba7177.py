"""Rewritten program: fast bitset subset-sum + greedy + add/swap local search."""
# EVOLVE-BLOCK-START
import time
import random

TARGET = 200
MASK = (1 << (TARGET + 1)) - 1  # bits 0..TARGET


def _creates_target(reachable, x):
    """True if adding x makes TARGET reachable (i.e. TARGET-x already reachable)."""
    if x > TARGET:
        return False
    return (reachable >> (TARGET - x)) & 1


def _add(reachable, x):
    return reachable | ((reachable << x) & MASK)


def _is_200_free(selected):
    reachable = 1  # bit 0
    for x in selected:
        if _creates_target(reachable, x):
            return False
        reachable = _add(reachable, x)
    return True


def _reachable_of(selected):
    reachable = 1
    for x in selected:
        reachable = _add(reachable, x)
    return reachable


def _greedy(order):
    selected = []
    reachable = 1
    for x in order:
        if not _creates_target(reachable, x):
            selected.append(x)
            reachable = _add(reachable, x)
    return selected, reachable


def _add_local(selected, reachable, all_elements):
    """Repeatedly add any missing element that is safe."""
    selected_set = set(selected)
    changed = True
    while changed:
        changed = False
        for x in all_elements:
            if x not in selected_set and not _creates_target(reachable, x):
                selected.append(x)
                selected_set.add(x)
                reachable = _add(reachable, x)
                changed = True
    return selected, reachable


def _swap_improve(selected, all_elements):
    """Try removing one element then greedily re-adding; keep if larger."""
    best = list(selected)
    best_size = len(best)
    for a in list(selected):
        sel = [x for x in selected if x != a]
        reach = _reachable_of(sel)
        sel, reach = _add_local(sel, reach, all_elements)
        if len(sel) > best_size:
            best, best_size = sel, len(sel)
    return best


def _two_swap_improve(selected, all_elements, deadline):
    """Try removing two elements then greedily re-adding; keep if larger."""
    best = list(selected)
    best_size = len(best)
    sel_list = list(selected)
    n = len(sel_list)
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            if time.time() > deadline:
                break
            sel = [x for k, x in enumerate(sel_list) if k != i and k != j]
            reach = _reachable_of(sel)
            sel, reach = _add_local(sel, reach, all_elements)
            if len(sel) > best_size:
                best, best_size = sel, len(sel)
    return best


def construct():
    start = time.time()
    deadline = start + 55.0
    all_elements = list(range(1, 101))
    rng = random.Random(20240517)

    best = []
    best_size = 0

    def consider(sel):
        nonlocal best, best_size
        if len(sel) > best_size:
            best, best_size = list(sel), len(sel)

    # Diverse initial orders
    orders = [
        sorted(all_elements, reverse=True),
        sorted(all_elements),
        sorted(all_elements, key=lambda x: x % 7),
        sorted(all_elements, key=lambda x: -x % 13),
    ]
    for _ in range(40):
        o = all_elements[:]
        rng.shuffle(o)
        orders.append(o)

    for o in orders:
        sel, reach = _greedy(o)
        sel, reach = _add_local(sel, reach, all_elements)
        consider(sel)

    # Swap-based improvement on the current best (and a few good starts)
    if time.time() < deadline:
        improved = _swap_improve(best, all_elements)
        consider(improved)
        if time.time() < deadline - 5.0 and len(best) < 100:
            improved2 = _two_swap_improve(best, all_elements, deadline - 2.0)
            consider(improved2)

    # Random restarts: greedy + add-local + occasional swap
    iters = 0
    while time.time() < deadline - 3.0:
        o = all_elements[:]
        rng.shuffle(o)
        sel, reach = _greedy(o)
        sel, reach = _add_local(sel, reach, all_elements)
        iters += 1
        if iters % 25 == 0 and time.time() < deadline - 5.0:
            sel = _swap_improve(sel, all_elements)
            if len(sel) < 100 and time.time() < deadline - 3.0:
                sel = _two_swap_improve(sel, all_elements, deadline - 2.0)
        consider(sel)
        if best_size >= 100:
            break

    # Final polish: add-local then swap on the best
    if time.time() < deadline - 1.0 and best:
        sel, reach = _add_local(list(best), _reachable_of(best), all_elements)
        consider(sel)
        if time.time() < deadline - 3.0:
            sel = _swap_improve(best, all_elements)
            consider(sel)
            if len(sel) < 100 and time.time() < deadline - 1.0:
                sel = _two_swap_improve(sel, all_elements, deadline - 0.5)
                consider(sel)

    # Safety: ensure nonempty and valid; fall back to a known-good 35-set
    if not best or not _is_200_free(best):
        best = list(range(66, 101))
    return best
# EVOLVE-BLOCK-END
