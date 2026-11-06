import time
import random
from fractions import Fraction

_DEADLINE = 50.0
_START = time.time()

def _deadline_hit():
    return (time.time() - _START) > _DEADLINE

def _check_valid_float(elements):
    """
    Check if the set of reciprocals of elements has distinct subset sums using float.
    Returns True if valid, False otherwise.
    """
    sums = [0.0]
    sums_set = {0.0}
    for a in elements:
        r = 1.0 / a
        new_sums = [s + r for s in sums]
        # Check for collisions
        for ns in new_sums:
            if ns in sums_set:
                return False
        sums.extend(new_sums)
        sums_set.update(new_sums)
    return True

def _try_greedy(order):
    """
    Greedy construction.
    Returns a list of chosen integers.
    """
    sums = [0.0]
    sums_set = {0.0}
    chosen = []
    chosen_set = set()
    
    for a in order:
        if a in chosen_set:
            continue
        r = 1.0 / a
        new_sums = [s + r for s in sums]
        
        collides = False
        for ns in new_sums:
            if ns in sums_set:
                collides = True
                break
        
        if collides:
            continue
            
        sums.extend(new_sums)
        sums_set.update(new_sums)
        chosen.append(a)
        chosen_set.add(a)
        
    return chosen

def construct():
    global _START
    _START = time.time()
    random.seed()
    full = list(range(1, 23))

    best = []
    best_k = 0

    # Run many greedy restarts
    # Each greedy run is O(K * 2^K). For K=10, 2^10=1024. Very fast.
    # We can do thousands of restarts.
    
    while not _deadline_hit():
        order = full[:]
        random.shuffle(order)
        cand = _try_greedy(order)
        
        if len(cand) > best_k:
            best_k = len(cand)
            best = cand
            
            # Simple local search: try a few random swaps to see if we can improve
            # Limit to 20 swap attempts to keep it fast
            current_set = set(best)
            for _ in range(20):
                if _deadline_hit():
                    break
                # Pick a random element to remove
                if not current_set:
                    break
                remove_a = random.choice(list(current_set))
                # Pick a random element to add (not in set)
                candidates = [x for x in full if x not in current_set]
                if not candidates:
                    break
                add_a = random.choice(candidates)
                
                new_set = (current_set - {remove_a}) | {add_a}
                
                # Check validity
                if _check_valid_float(list(new_set)):
                    current_set = new_set
                    best = list(current_set)
                    best_k = len(best)

    # Final verification with Fraction for the best candidate
    # This is exact and ensures correctness
    if best:
        sums = {Fraction(0)}
        valid = True
        for a in best:
            r = Fraction(1, a)
            new_sums = [s + r for s in sums]
            for ns in new_sums:
                if ns in sums:
                    valid = False
                    break
            if not valid:
                break
            sums.update(new_sums)
            
        if valid:
            return list(best)
        else:
            # Fallback to a known small valid set
            return [1, 2, 3, 5, 7, 11, 13, 17, 19]
            
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
