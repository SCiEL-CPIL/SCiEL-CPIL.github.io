import random
import time
import itertools
# EVOLVE-BLOCK-START

_LCM = 232792560
_INV = {x: _LCM // x for x in range(1, 23)}
_ALL = list(range(1, 23))

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
    """Try swap-based local search: remove one element, try to add two others."""
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
            new_sums = {0}
            for x in new_chosen:
                inv = _INV[x]
                shifted = {s + inv for s in new_sums}
                new_sums = new_sums | shifted
            # Try to add any element not in new_chosen
            for add in _ALL:
                if add in new_chosen:
                    continue
                inv = _INV[add]
                shifted = {s + inv for s in new_sums}
                if new_sums.isdisjoint(shifted):
                    new_sums = new_sums | shifted
                    new_chosen.add(add)
                    # Try to add more
                    for add2 in _ALL:
                        if add2 in new_chosen:
                            continue
                        inv2 = _INV[add2]
                        shifted2 = {s + inv2 for s in new_sums}
                        if new_sums.isdisjoint(shifted2):
                            new_sums = new_sums | shifted2
                            new_chosen.add(add2)
                    if len(new_chosen) > len(chosen):
                        chosen = new_chosen
                        sums = new_sums
                        improved = True
                        break
            if improved:
                break
    return chosen, sums

def _local_search_2opt(chosen, sums, deadline):
    """Try 2-opt style local search: remove two elements, try to add 2-3 others."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        if len(chosen) < 3:
            break
        # Try all pairs of elements to remove
        candidates = list(chosen)
        random.shuffle(candidates)
        for i, rem1 in enumerate(candidates):
            if time.time() > deadline:
                break
            if rem1 not in chosen:
                continue
            for j in range(i+1, len(candidates)):
                if time.time() > deadline:
                    break
                rem2 = candidates[j]
                if rem2 not in chosen:
                    continue
                if rem1 == rem2:
                    continue
                # Remove both
                new_chosen = set(chosen)
                new_chosen.discard(rem1)
                new_chosen.discard(rem2)
                # Rebuild sums
                new_sums = {0}
                for x in new_chosen:
                    inv = _INV[x]
                    shifted = {s + inv for s in new_sums}
                    new_sums = new_sums | shifted
                
                # Try to add elements not in new_chosen, greedily up to 3
                added_count = 0
                for add in _ALL:
                    if add in new_chosen:
                        continue
                    inv = _INV[add]
                    shifted = {s + inv for s in new_sums}
                    if new_sums.isdisjoint(shifted):
                        new_sums = new_sums | shifted
                        new_chosen.add(add)
                        added_count += 1
                        if added_count >= 3:
                            break
                
                # Accept if we added more than we removed (net gain)
                if len(new_chosen) > len(chosen):
                    chosen = new_chosen
                    sums = new_sums
                    improved = True
                    break
            if improved:
                break
    return chosen, sums

def _polish(chosen, sums, deadline):
    """Final high-precision polish: try all single-element swaps systematically."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        if len(chosen) < 1:
            break
        # Try each element in chosen for removal
        for rem in list(chosen):
            if time.time() > deadline:
                break
            # Rebuild sums without rem
            new_chosen = set(chosen)
            new_chosen.discard(rem)
            new_sums = {0}
            for x in new_chosen:
                inv = _INV[x]
                shifted = {s + inv for s in new_sums}
                new_sums = new_sums | shifted
            
            # Try to add each element not in new_chosen
            for add in _ALL:
                if add in new_chosen:
                    continue
                inv = _INV[add]
                shifted = {s + inv for s in new_sums}
                if new_sums.isdisjoint(shifted):
                    candidate_sums = new_sums | shifted
                    candidate_chosen = new_chosen | {add}
                    # Check if this improves cardinality or is equal but we want to explore
                    if len(candidate_chosen) > len(chosen):
                        chosen = candidate_chosen
                        sums = candidate_sums
                        improved = True
                        break
            if improved:
                break
    return chosen, sums

def _exhaustive_completion(chosen, sums, deadline):
    """Try all possible additions in multiple orders until no more fit."""
    remaining = [x for x in _ALL if x not in chosen]
    if not remaining:
        return chosen, sums
    
    # Try multiple orderings of remaining elements
    orders = [
        remaining[:],
        sorted(remaining, key=lambda x: _INV[x], reverse=True),
        sorted(remaining, key=lambda x: _INV[x]),
        sorted(remaining, reverse=True),
        sorted(remaining),
    ]
    # Add some random shuffles
    for _ in range(5):
        shuffled = remaining[:]
        random.shuffle(shuffled)
        orders.append(shuffled)
    
    for order in orders:
        if time.time() > deadline:
            break
        # Try to add elements in this order
        improved = False
        for x in order:
            if x in chosen:
                continue
            inv = _INV[x]
            shifted = {s + inv for s in sums}
            if sums.isdisjoint(shifted):
                sums = sums | shifted
                chosen.add(x)
                improved = True
        if not improved:
            break
    
    return chosen, sums

def construct():
    start = time.time()
    deadline = start + 50  # safety margin under 60s (evaluator kills at 90s)

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
    while time.time() < deadline - 5:
        order = _ALL[:]
        random.shuffle(order)
        chosen, sums = _greedy_order(order, deadline - 5)
        # completion
        chosen, sums = _complete(chosen, sums, deadline - 4)
        # local search to escape local optima
        if time.time() < deadline - 6:
            chosen, sums = _local_search(chosen, sums, deadline - 6)
        # 2-opt local search for deeper exploration
        if time.time() < deadline - 7:
            chosen, sums = _local_search_2opt(chosen, sums, deadline - 7)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best = set(chosen)

    # Final polish pass on the best solution found
    if best and time.time() < deadline - 2:
        # Rebuild sums for best
        best_sums = {0}
        for x in best:
            inv = _INV[x]
            shifted = {s + inv for s in best_sums}
            best_sums = best_sums | shifted
        # Exhaustive completion first
        best, best_sums = _exhaustive_completion(best, best_sums, deadline - 1)
        # Then polish
        best, best_sums = _polish(best, best_sums, deadline)

    if not best:
        best = {1}

    return sorted(best)

# EVOLVE-BLOCK-END
