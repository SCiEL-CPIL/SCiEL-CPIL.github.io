"""Combined B2[2] set maximization: fast greedy seed + structured local search with deadline guard."""
import time
import random

DEADLINE = 140.0  # safety margin before 150s kill


def _try_add(S, counts, x):
    """Return new counts if adding x keeps all sums <=2, else None."""
    if x in S:
        return None
    new_counts = dict(counts)
    for y in S:
        s = x + y
        c = new_counts.get(s, 0) + 1
        if c > 2:
            return None
        new_counts[s] = c
    s2 = 2 * x
    c2 = new_counts.get(s2, 0) + 1
    if c2 > 2:
        return None
    new_counts[s2] = c2
    return new_counts


def _remove(S, counts, x):
    """Remove x from S, return new counts."""
    new_counts = dict(counts)
    for y in S:
        if y == x:
            continue
        s = x + y
        new_counts[s] -= 1
        if new_counts[s] == 0:
            del new_counts[s]
    s2 = 2 * x
    new_counts[s2] -= 1
    if new_counts[s2] == 0:
        del new_counts[s2]
    return new_counts


def _greedy_seed():
    """Fast greedy: add x if it doesn't violate the B2[2] constraint."""
    chosen = []
    counts = {}
    for x in range(1, 1001):
        ok = True
        for y in chosen:
            s = x + y
            if counts.get(s, 0) + 1 > 2:
                ok = False
                break
        if ok:
            s2 = 2 * x
            if counts.get(s2, 0) + 1 > 2:
                ok = False
        if ok:
            for y in chosen:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            counts[2 * x] = counts.get(2 * x, 0) + 1
            chosen.append(x)
    return chosen, counts


def _randomized_greedy(rng):
    """Randomized greedy: shuffle order, add if feasible."""
    S = []
    counts = {}
    order = list(range(1, 1001))
    rng.shuffle(order)
    for x in order:
        ok = True
        for y in S:
            s = x + y
            if counts.get(s, 0) + 1 > 2:
                ok = False
                break
        if ok:
            s2 = 2 * x
            if counts.get(s2, 0) + 1 > 2:
                ok = False
        if ok:
            for y in S:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            counts[2 * x] = counts.get(2 * x, 0) + 1
            S.append(x)
    return S, counts


def _local_search(S, counts, time_limit, rng):
    """Improve feasible set via add/remove/swap moves with time budget."""
    S = list(S)
    counts = dict(counts)
    best_S = list(S)
    best_counts = dict(counts)
    best_size = len(S)
    end = time.time() + time_limit
    all_elems = set(range(1, 1001))

    # Precompute set for fast membership
    S_set = set(S)

    while time.time() < end:
        # Decide move type with bias toward adds when small, swaps when large
        r = rng.random()
        if r < 0.4:
            move_type = 'add'
        elif r < 0.7:
            move_type = 'swap'
        else:
            move_type = 'remove'

        if move_type == 'add':
            comp = all_elems - S_set
            if not comp:
                continue
            # Try a few random candidates per iteration
            for _ in range(3):
                if time.time() >= end:
                    break
                x = rng.choice(list(comp))
                new_counts = _try_add(S, counts, x)
                if new_counts is not None:
                    S.append(x)
                    S_set.add(x)
                    counts = new_counts
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = list(S)
                        best_counts = dict(counts)
                        comp = all_elems - S_set
                    if not comp:
                        break
        elif move_type == 'remove':
            if not S:
                continue
            x = rng.choice(S)
            counts = _remove(S, counts, x)
            S.remove(x)
            S_set.discard(x)
        else:  # swap
            if not S:
                continue
            x = rng.choice(S)
            comp = all_elems - S_set
            if not comp:
                continue
            y = rng.choice(list(comp))
            # Remove x then try add y
            S_tmp = [z for z in S if z != x]
            S_set_tmp = S_set - {x}
            counts_after_rem = _remove(S, counts, x)
            new_counts = _try_add(S_tmp, counts_after_rem, y)
            if new_counts is not None:
                S = S_tmp + [y]
                S_set = S_set_tmp | {y}
                counts = new_counts
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = list(S)
                    best_counts = dict(counts)
    return best_S, best_counts


def construct():
    start = time.time()
    rng = random.Random(42)

    # Phase 1: Greedy seed
    S, counts = _greedy_seed()
    best_S = list(S)
    best_counts = dict(counts)
    best_size = len(S)

    # Phase 2: Local search on greedy seed
    remaining = DEADLINE - (time.time() - start)
    if remaining > 5:
        S, counts = _local_search(S, counts, time_limit=min(15, remaining - 5), rng=rng)
        if len(S) > best_size:
            best_S, best_counts, best_size = list(S), dict(counts), len(S)

    # Phase 3: Randomized restarts
    restart_budget = DEADLINE - (time.time() - start) - 10
    restart_count = 0
    while time.time() - start < DEADLINE - 15 and restart_count < 50:
        S_r, counts_r = _randomized_greedy(rng)
        restart_count += 1

        # Local search on this restart
        ls_time = min(10, DEADLINE - (time.time() - start) - 10)
        if ls_time < 2:
            break
        S_l, counts_l = _local_search(S_r, counts_r, time_limit=ls_time, rng=rng)
        if len(S_l) > best_size:
            best_S, best_counts, best_size = list(S_l), dict(counts_l), len(S_l)

    # Phase 4: Final local search on best found
    final_time = DEADLINE - (time.time() - start) - 2
    if final_time > 3:
        S_f, counts_f = _local_search(best_S, best_counts, time_limit=final_time, rng=rng)
        if len(S_f) > best_size:
            best_S, best_counts, best_size = list(S_f), dict(counts_f), len(S_f)

    return best_S
