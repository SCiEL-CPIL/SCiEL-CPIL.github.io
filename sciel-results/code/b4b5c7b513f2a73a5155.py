"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
import time

_TARGET = 200
_UNIVERSE = list(range(1, 101))
_MASK = (1 << (_TARGET + 1)) - 1  # Mask to keep bits 0..200


def _greedy_descending_bitset():
    """Greedy: process from largest to smallest, keep if it does not create a sum of 200."""
    selected = []
    reachable = 1  # bit 0 is set (sum 0)
    for x in range(100, 0, -1):
        if ((reachable << x) & (1 << _TARGET)) == 0:
            selected.append(x)
            reachable |= (reachable << x)
            reachable &= _MASK
    return selected


def _local_improve_bitset(selected):
    """Try to add back excluded elements one at a time if safe. Repeat until no change."""
    selected_set = set(selected)
    excluded = [x for x in _UNIVERSE if x not in selected_set]

    reachable = 1
    for x in selected:
        reachable |= (reachable << x)
        reachable &= _MASK

    improved = True
    while improved:
        improved = False
        for x in excluded:
            if x in selected_set:
                continue
            new_reachable = reachable | (reachable << x)
            if (new_reachable & (1 << _TARGET)) == 0:
                selected_set.add(x)
                reachable = new_reachable & _MASK
                improved = True
    return list(selected_set)


def _try_orders_bitset():
    """Try multiple construction orders and keep the best (largest) valid set."""
    best = None
    orders = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
    ]
    evens = [x for x in range(100, 0, -2)]
    odds = [x for x in range(99, 0, -2)]
    orders.append(evens + odds)
    orders.append(odds + evens)

    for order in orders:
        selected = []
        reachable = 1
        for x in order:
            if ((reachable << x) & (1 << _TARGET)) == 0:
                selected.append(x)
                reachable |= (reachable << x)
                reachable &= _MASK
        sel = _local_improve_bitset(selected)
        if best is None or len(sel) > len(best):
            best = sel
    return best


def _has_sum_bitset(selected, target=_TARGET):
    """Check validity using bitset."""
    reachable = 1
    for x in selected:
        reachable |= (reachable << x)
        reachable &= (1 << (target + 1)) - 1
    return bool(reachable & (1 << target))


def _greedy_with_weights_bitset(weight_func, start=100, end=0, step=-1):
    """Greedy with a weight function: at each step, pick the available element with
    highest weight that is safe to add. This is more flexible than fixed order."""
    available = set(range(start, end, step))
    selected = []
    reachable = 1
    while available:
        # Find the best (highest weight) safe element
        best_x = None
        best_w = -1
        for x in available:
            if ((reachable << x) & (1 << _TARGET)) == 0:
                w = weight_func(x, len(selected))
                if w > best_w:
                    best_w = w
                    best_x = x
        if best_x is None:
            break
        available.remove(best_x)
        selected.append(best_x)
        reachable |= (reachable << best_x)
        reachable &= _MASK
    return selected


def _random_restart_greedy(seed, time_limit):
    """Randomize the order and do greedy + local improve."""
    import random
    rng = random.Random(seed)
    order = list(range(1, 101))
    rng.shuffle(order)
    selected = []
    reachable = 1
    for x in order:
        if ((reachable << x) & (1 << _TARGET)) == 0:
            selected.append(x)
            reachable |= (reachable << x)
            reachable &= _MASK
    return _local_improve_bitset(selected)


def _pairwise_swap_improve(selected, time_limit):
    """Try to improve by swapping one excluded element with one included element.
    If excluding an element allows adding two excluded elements (net +1), do it."""
    selected_set = set(selected)
    excluded = [x for x in _UNIVERSE if x not in selected_set]
    if not excluded:
        return selected

    best = list(selected)
    best_len = len(best)
    best_reachable = 1
    for x in best:
        best_reachable |= (best_reachable << x)
        best_reachable &= _MASK

    improved = True
    while improved:
        improved = False
        for out_x in best:
            # Remove out_x
            # Rebuild reachable without out_x
            new_reachable = 1
            for x in best:
                if x == out_x:
                    continue
                new_reachable |= (new_reachable << x)
                new_reachable &= _MASK
            # Now try to add excluded elements
            new_selected = [x for x in best if x != out_x]
            new_sel_set = set(new_selected)
            added = 0
            for in_x in excluded:
                if in_x in new_sel_set:
                    continue
                if ((new_reachable << in_x) & (1 << _TARGET)) == 0:
                    new_sel_set.add(in_x)
                    new_reachable |= (new_reachable << in_x)
                    new_reachable &= _MASK
                    added += 1
            if added > 1 and len(new_selected) + added > best_len:
                best = list(new_sel_set)
                best_len = len(best)
                best_reachable = new_reachable
                improved = True
        # Also try single swap: remove one, add one different excluded
        for out_x in best:
            new_reachable = 1
            for x in best:
                if x == out_x:
                    continue
                new_reachable |= (new_reachable << x)
                new_reachable &= _MASK
            new_sel_set = set(x for x in best if x != out_x)
            for in_x in excluded:
                if in_x in new_sel_set:
                    continue
                if ((new_reachable << in_x) & (1 << _TARGET)) == 0:
                    new_sel_set.add(in_x)
                    if len(new_sel_set) > best_len:
                        best = list(new_sel_set)
                        best_len = len(best)
                        improved = True
    return best


def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill

    # 1. Greedy descending
    best = _greedy_descending_bitset()
    best = _local_improve_bitset(best)

    # 2. Try other orders
    if time.time() < deadline:
        cand = _try_orders_bitset()
        if cand is not None and len(cand) > len(best):
            best = cand

    # 3. Weighted greedy with different weight functions
    if time.time() < deadline:
        # Weight: prefer larger elements (same as descending but dynamic)
        cand = _greedy_with_weights_bitset(lambda x, cnt: x)
        cand = _local_improve_bitset(cand)
        if len(cand) > len(best):
            best = cand

    # 4. Random restarts
    if time.time() < deadline:
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            cand = _random_restart_greedy(seed, deadline)
            if cand is not None and len(cand) > len(best):
                best = cand
                # Try pairwise swap on the new best
                best = _pairwise_swap_improve(best, deadline)

    # 5. Pairwise swap improvement on current best
    if time.time() < deadline - 10:
        best = _pairwise_swap_improve(best, deadline)

    # Final safety check
    if _has_sum_bitset(best):
        best = _greedy_descending_bitset()
        best = _local_improve_bitset(best)

    if not best:
        best = [100]

    return best
# EVOLVE-BLOCK-END
