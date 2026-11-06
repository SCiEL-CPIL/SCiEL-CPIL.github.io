import random
import time

# EVOLVE-BLOCK-START

_LCM = 232792560
_INV = {x: _LCM // x for x in range(1, 23)}
_ALL = list(range(1, 23))

def _can_add(sums, w):
    """
    Check if weight w can be added to the set of subset sums 'sums'
    without causing any collisions.
    Collision occurs if there exists s1, s2 in sums such that s1 + w = s2.
    Equivalently, (s1 + w) is in sums.
    """
    for s in sums:
        if s + w in sums:
            return False
    return True

def _add_element(sums, w):
    """
    Add weight w to the set of subset sums.
    Returns the new set of sums.
    """
    # sums | {s + w for s in sums}
    # Using set comprehension and union is efficient in C
    return sums | {s + w for s in sums}

def _greedy(order, deadline):
    """
    Greedily add elements in the given order, maintaining dissociation.
    """
    sums = {0}
    chosen = set()
    for x in order:
        if x in chosen:
            continue
        w = _INV[x]
        if _can_add(sums, w):
            sums = _add_element(sums, w)
            chosen.add(x)
        if time.time() > deadline:
            break
    return chosen, sums

def _rebuild_sums(chosen):
    """
    Rebuild the set of subset sums from the chosen elements.
    Used when removing elements in local search.
    """
    sums = {0}
    for x in chosen:
        w = _INV[x]
        sums = _add_element(sums, w)
    return sums

def _local_search(chosen, sums, deadline):
    """
    Local search to improve the set size.
    Strategies:
    1. Try to add any remaining element.
    2. Try swapping one element from chosen with one from remaining.
    """
    remaining = [x for x in _ALL if x not in chosen]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to add any remaining element
        if remaining:
            random.shuffle(remaining)
            new_remaining = []
            for x in remaining:
                w = _INV[x]
                if _can_add(sums, w):
                    sums = _add_element(sums, w)
                    chosen.add(x)
                    improved = True
                else:
                    new_remaining.append(x)
            remaining = new_remaining
        
        # 2. If no simple add worked, try swap
        if not improved and remaining and chosen:
            candidates = list(chosen)
            random.shuffle(candidates)
            for rem in candidates:
                if time.time() > deadline:
                    break
                
                # Tentatively remove 'rem'
                new_chosen = set(chosen)
                new_chosen.discard(rem)
                
                # Rebuild sums for new_chosen
                new_sums = _rebuild_sums(new_chosen)
                
                # Try to add any element from remaining
                found = False
                for add in remaining:
                    w = _INV[add]
                    if _can_add(new_sums, w):
                        new_sums = _add_element(new_sums, w)
                        new_chosen.add(add)
                        
                        # After successful swap, try to add more
                        # We need to update 'remaining' conceptually, 
                        # but for this step, we just accept the swap if it increases size
                        if len(new_chosen) > len(chosen):
                            chosen = new_chosen
                            sums = new_sums
                            improved = True
                            found = True
                            break
                            
                if found:
                    # Update remaining list
                    remaining = [x for x in _ALL if x not in chosen]
                    break

    return chosen, sums

def construct():
    start = time.time()
    deadline = start + 85  # Safety margin
    best_size = 0
    best_set = set()

    # Generate a variety of orderings
    orderings = [
        sorted(_ALL),                # Ascending
        sorted(_ALL, reverse=True),  # Descending
        sorted(_ALL, key=lambda x: _INV[x]),       # Smallest reciprocal first (Largest i)
        sorted(_ALL, key=lambda x: _INV[x], reverse=True), # Largest reciprocal first (Smallest i)
    ]
    
    # Add random orderings
    for _ in range(100):
        order = _ALL[:]
        random.shuffle(order)
        orderings.append(order)

    for order in orderings:
        if time.time() > deadline:
            break
            
        chosen, sums = _greedy(order, deadline)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = set(chosen)
        
        # Local search
        if time.time() < deadline - 5:
            chosen, sums = _local_search(chosen, sums, deadline - 5)
            if len(chosen) > best_size:
                best_size = len(chosen)
                best_set = set(chosen)

    if not best_set:
        best_set = {1}

    return sorted(best_set)

# EVOLVE-BLOCK-END
