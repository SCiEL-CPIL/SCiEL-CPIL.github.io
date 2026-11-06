"""Rewritten program: Optimized construction of max-size 200-free subset."""
# EVOLVE-BLOCK-START
import time

TARGET = 200
MASK = (1 << (TARGET + 1)) - 1  # bits 0..TARGET


def _creates_target(reachable, x):
    """True if adding x makes TARGET reachable (i.e., TARGET-x already reachable)."""
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


def construct():
    start = time.time()
    deadline = start + 55.0
    
    # Theoretical maximum cardinality is 35.
    # The set {66, ..., 100} has cardinality 35 and is 200-free.
    best = list(range(66, 101))
    
    # Verify validity
    if not _is_200_free(best):
        # Fallback to a known smaller valid set if analysis was wrong
        best = list(range(67, 101))
        
    # Attempt to improve via local search:
    # Check if any element from 1..65 can be added without creating a sum of 200.
    # Based on analysis, no such element exists for this specific set.
    # We run the check anyway to be rigorous.
    
    reachable = _reachable_of(best)
    best_set = set(best)
    
    for x in range(1, 66):
        if x not in best_set:
            if not _creates_target(reachable, x):
                best.append(x)
                best_set.add(x)
                reachable = _add(reachable, x)
                if time.time() > deadline:
                    break

    # Final check
    if not _is_200_free(best):
        # Emergency fallback
        best = list(range(66, 101))
        
    return best
# EVOLVE-BLOCK-END
