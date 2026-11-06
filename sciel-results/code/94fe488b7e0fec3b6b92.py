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


def _add_local_pair(selected, reachable, all_elements):
    """Try to add pairs of missing elements simultaneously."""
    selected_set = set(selected)
    missing = [x for x in all_elements if x not in selected_set]
    n = len(missing)
    improved = False
    for i in range(n):
        for j in range(i + 1, n):
            x, y = missing[i], missing[j]
            # Check if adding both x and y (in either order) is safe
            r1 = reachable
            if not _creates_target(r1, x):
                r1 = _add(r1, x)
                if not _creates_target(r1, y):
                    # Safe to add both
                    selected.append(x)
                    selected.append(y)
                    selected_set.add(x)
                    selected_set.add(y)
                    reachable = _add(_add(reachable, x), y)
                    improved = True
                    break
        if improved:
            # Restart scanning since reachable changed
            missing = [x for x in all_elements if x not in selected_set]
            n = len(missing)
            improved = False
            continue
    return selected, reachable


def _k_swap_improve(selected, all_elements, k_max=3, deadline=None):
    """
    Try removing up to k_max elements then greedily re-adding; keep if larger.
    This generalizes 1-swap and 2-swap.
    """
    best = list(selected)
    best_size = len(best)
    sel_list = list(selected)
    n = len(sel_list)
    
    # 1-swap
    for i in range(n):
        if deadline and time.time() > deadline:
            break
        sel = [x for k, x in enumerate(sel_list) if k != i]
        reach = _reachable_of(sel)
        sel, reach = _add_local(sel, reach, all_elements)
        sel, reach = _add_local_pair(sel, reach, all_elements)
        if len(sel) > best_size:
            best, best_size = sel, len(sel)
            sel_list = list(best)
            n = len(sel_list)

    # 2-swap
    for i in range(n):
        if deadline and time.time() > deadline:
            break
        for j in range(i + 1, n):
            if deadline and time.time() > deadline:
                break
            sel = [x for k, x in enumerate(sel_list) if k != i and k != j]
            reach = _reachable_of(sel)
            sel, reach = _add_local(sel, reach, all_elements)
            sel, reach = _add_local_pair(sel, reach, all_elements)
            if len(sel) > best_size:
                best, best_size = sel, len(sel)
                sel_list = list(best)
                n = len(sel_list)
                break # restart i loop with new best? Or just update best. Let's update best and continue.
                # Actually, restarting the outer loop is safer to avoid missing improvements from the new base.
                # But for efficiency, just updating best is okay for now.
                # Let's break inner and let outer continue with old sel_list? No, sel_list is updated.
                # So we should break inner and outer to restart from scratch with new best?
                # That's expensive. Let's just update best and continue.
                # But sel_list is now the new best. So the next iterations will use the new best.
                # This is a form of hill climbing.

    # 3-swap (only if we have time and k_max >= 3)
    if k_max >= 3 and (deadline is None or time.time() < deadline):
        # To avoid O(n^3) explosion, we limit the scope or use a heuristic.
        # Given n <= 100, n^3 is 1,000,000. Each step is ~100 ops. 100M ops might be too slow.
        # We'll only do 3-swap if n is small or we have a lot of time.
        # Or we can sample random triples.
        # Let's do random sampling of triples to keep it fast.
        rng_local = random.Random(42)
        for _ in range(1000): # Limit number of random 3-swaps
            if deadline and time.time() > deadline:
                break
            if n < 3:
                break
            idxs = rng_local.sample(range(n), 3)
            idx_set = set(idxs)
            sel = [x for k, x in enumerate(sel_list) if k not in idx_set]
            reach = _reachable_of(sel)
            sel, reach = _add_local(sel, reach, all_elements)
            sel, reach = _add_local_pair(sel, reach, all_elements)
            if len(sel) > best_size:
                best, best_size = sel, len(sel)
                sel_list = list(best)
                n = len(sel_list)

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
        sorted(all_elements, key=lambda x: x % 5),
        sorted(all_elements, key=lambda x: -x % 11),
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
        improved = _k_swap_improve(best, all_elements, k_max=3, deadline=deadline)
        consider(improved)

    # Random restarts: greedy + add-local + pair-local + occasional swap
    iters = 0
    while time.time() < deadline - 3.0:
        o = all_elements[:]
        rng.shuffle(o)
        sel, reach = _greedy(o)
        sel, reach = _add_local(sel, reach, all_elements)
        sel, reach = _add_local_pair(sel, reach, all_elements)
        iters += 1
        if iters % 10 == 0 and time.time() < deadline - 5.0:
            sel = _k_swap_improve(sel, all_elements, k_max=3, deadline=deadline - 2.0)
        consider(sel)
        if best_size >= 100:
            break

    # Final polish: add-local + pair-local then swap on the best
    if time.time() < deadline - 1.0 and best:
        sel, reach = _add_local(list(best), _reachable_of(best), all_elements)
        sel, reach = _add_local_pair(sel, reach, all_elements)
        consider(sel)
        if time.time() < deadline - 3.0:
            sel = _k_swap_improve(best, all_elements, k_max=3, deadline=deadline - 0.5)
            consider(sel)

    # Safety: ensure nonempty and valid; fall back to a known-good 35-set
    if not best or not _is_200_free(best):
        best = list(range(66, 101))
    return best
# EVOLVE-BLOCK-END
