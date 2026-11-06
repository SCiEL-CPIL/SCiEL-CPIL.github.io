import random
import time
# EVOLVE-BLOCK-START

_LCM = 232792560
_INV = {x: _LCM // x for x in range(1, 23)}
_ALL = list(range(1, 23))

def _rebuild_sums(chosen):
    """Rebuild the set of all subset sums for a given chosen set."""
    sums = {0}
    for x in chosen:
        inv = _INV[x]
        shifted = {s + inv for s in sums}
        sums = sums | shifted
    return sums

def _greedy_order(order, deadline):
    """
    Given an ordering of candidates, greedily add elements maintaining dissociation.
    Returns (set_of_elements, set_of_subset_sums).
    Uses fast incremental collision check.
    """
    sums = {0}
    chosen = set()
    for x in order:
        if x in chosen:
            continue
        inv = _INV[x]
        # collision check: does shifting sums by inv hit sums?
        # Use set comprehension and isdisjoint for speed (C implementation)
        shifted = {s + inv for s in sums}
        if not sums.isdisjoint(shifted):
            continue
        # accept
        # update sums: add s+inv for all s
        # sums = sums | shifted is faster than loop
        sums = sums | shifted
        chosen.add(x)
        if time.time() > deadline:
            break
    return chosen, sums

def _complete(chosen, sums, deadline):
    """Try to add more elements to chosen in random order until no more fit."""
    remaining = [x for x in _ALL if x not in chosen]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(remaining)
        still_remaining = []
        for x in remaining:
            if x in chosen:
                continue
            inv = _INV[x]
            shifted = {s + inv for s in sums}
            if sums.isdisjoint(shifted):
                sums = sums | shifted
                chosen.add(x)
                improved = True
            else:
                still_remaining.append(x)
        remaining = still_remaining
    return chosen, sums

def _local_search(chosen, sums, deadline):
    """Try swap-based local search: remove one element, try to add one or two others.
    Uses a soft penalty approach: prefer swaps that maintain dissociation, but allow
    temporary size decrease if it leads to a larger set after completion."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        if len(chosen) < 2:
            break
        # Try removing each element and adding a different one
        candidates = list(chosen)
        random.shuffle(candidates)
        for rem in candidates:
            if time.time() > deadline:
                break
            if rem not in chosen:
                continue
            # Remove rem
            new_chosen = set(chosen)
            new_chosen.discard(rem)
            # Rebuild sums for new_chosen
            new_sums = _rebuild_sums(new_chosen)
            # Try to add any element not in new_chosen
            best_add = None
            best_new_sums = None
            for add in _ALL:
                if add in new_chosen:
                    continue
                inv = _INV[add]
                shifted = {s + inv for s in new_sums}
                if new_sums.isdisjoint(shifted):
                    candidate_sums = new_sums | shifted
                    candidate_chosen = set(new_chosen)
                    candidate_chosen.add(add)
                    # Try to greedily add more elements
                    remaining = [x for x in _ALL if x not in candidate_chosen]
                    random.shuffle(remaining)
                    for add2 in remaining:
                        if time.time() > deadline:
                            break
                        inv2 = _INV[add2]
                        shifted2 = {s + inv2 for s in candidate_sums}
                        if candidate_sums.isdisjoint(shifted2):
                            candidate_sums = candidate_sums | shifted2
                            candidate_chosen.add(add2)
                    # Prefer larger sets; tie-break by more "room" (fewer collisions)
                    if len(candidate_chosen) > len(chosen):
                        best_add = add
                        best_new_sums = candidate_sums
                        # Store the full candidate for potential use
                        if len(candidate_chosen) >= len(chosen) + 2:
                            break
                    elif best_add is None:
                        # Keep as fallback if no improvement found yet
                        best_add = add
                        best_new_sums = candidate_sums
            if best_add is not None and best_new_sums is not None:
                # Rebuild the chosen set from best_add scenario
                # We need to reconstruct which elements were added
                # For simplicity, just use the best candidate we found
                # Actually, let's just try the best single swap and see if it helps
                new_chosen2 = set(chosen)
                new_chosen2.discard(rem)
                new_chosen2.add(best_add)
                new_sums2 = _rebuild_sums(new_chosen2)
                # Try to complete
                remaining = [x for x in _ALL if x not in new_chosen2]
                random.shuffle(remaining)
                for add in remaining:
                    if time.time() > deadline:
                        break
                    inv = _INV[add]
                    shifted = {s + inv for s in new_sums2}
                    if new_sums2.isdisjoint(shifted):
                        new_sums2 = new_sums2 | shifted
                        new_chosen2.add(add)
                if len(new_chosen2) > len(chosen):
                    chosen = new_chosen2
                    sums = new_sums2
                    improved = True
                    break
            if improved:
                break
    return chosen, sums

def construct():
    start = time.time()
    deadline = start + 55  # safety margin under 60s (evaluator kills at 90s)

    best = set()
    best_size = 0

    # Warm start: try different orderings including sorted by size
    # Larger elements have smaller reciprocals, less likely to collide
    warm_orders = [
        _ALL[:],              # natural order
        sorted(_ALL, reverse=True),  # descending
        sorted(_ALL, key=lambda x: _INV[x], reverse=True),  # by inverse desc
        sorted(_ALL, key=lambda x: _INV[x]),  # by inverse asc
    ]
    for order in warm_orders:
        if time.time() > deadline:
            break
        shuffled = order[:]
        random.shuffle(shuffled)
        chosen, sums = _greedy_order(shuffled, deadline)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best = set(chosen)
    # Also try pure greedy on sorted orders
    for order in warm_orders:
        if time.time() > deadline:
            break
        chosen, sums = _greedy_order(order, deadline)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best = set(chosen)

    # Main loop: random greedy + completion + local search
    # Use simulated annealing-style: allow occasional worse moves to escape local optima
    current_temp = 1.0
    cooling_rate = 0.95
    while time.time() < deadline - 2:
        order = _ALL[:]
        random.shuffle(order)
        chosen, sums = _greedy_order(order, deadline - 2)
        # completion
        chosen, sums = _complete(chosen, sums, deadline - 1)
        # local search to escape local optima
        if time.time() < deadline - 3:
            chosen, sums = _local_search(chosen, sums, deadline - 3)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best = set(chosen)
        # Simulated annealing: occasionally perturb the best solution
        if best_size > 0 and random.random() < 0.3:
            # Try to find a better solution by perturbing the current best
            perturbed = set(best)
            if len(perturbed) >= 3:
                # Remove a random element
                rem = random.choice(list(perturbed))
                perturbed.discard(rem)
                perturbed_sums = _rebuild_sums(perturbed)
                # Try to add different elements
                remaining = [x for x in _ALL if x not in perturbed]
                random.shuffle(remaining)
                for add in remaining:
                    if time.time() > deadline - 1:
                        break
                    inv = _INV[add]
                    shifted = {s + inv for s in perturbed_sums}
                    if perturbed_sums.isdisjoint(shifted):
                        perturbed_sums = perturbed_sums | shifted
                        perturbed.add(add)
                # Complete the perturbed solution
                perturbed, perturbed_sums = _complete(perturbed, perturbed_sums, deadline - 1)
                if len(perturbed) > best_size:
                    best_size = len(perturbed)
                    best = set(perturbed)
        current_temp *= cooling_rate

    if not best:
        best = {1}

    return sorted(best)

# EVOLVE-BLOCK-END
