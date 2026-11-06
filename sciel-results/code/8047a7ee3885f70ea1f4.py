"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _is_valid(S):
    """Check if set S is reciprocal-subsum-free."""
    if len(S) <= 1:
        return True
    S_list = sorted(S)
    for a in S_list:
        target = Fraction(1, a)
        others = [Fraction(1, x) for x in S_list if x != a]
        # Build set of achievable subset sums
        achievable = {Fraction(0, 1)}
        for frac in others:
            new_achievable = set()
            for s in achievable:
                ns = s + frac
                new_achievable.add(ns)
                if ns == target:
                    return False
            achievable |= new_achievable
            if target in achievable:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # Start with all elements
    S = set(range(1, 25))
    
    # First, remove elements that cause violations
    # Greedy removal
    changed = True
    while changed:
        changed = False
        for a in sorted(S):
            if not _is_valid(S):
                # Find which element to remove
                pass
        # This approach is too slow. Let me do local search instead.
    
    # Local search: start with a valid set, try to add elements
    # Start with large elements (less likely to be subset sums)
    S = set(range(13, 25))  # 12 elements, likely valid
    
    # Verify
    if not _is_valid(S):
        # Remove problematic elements
        for a in sorted(S):
            S2 = S - {a}
            if _is_valid(S2):
                S = S2
                break
    
    best = set(S)
    
    # Try to add elements one by one
    for a in range(1, 25):
        if a not in S:
            if time.time() > deadline:
                break
            S2 = S | {a}
            if _is_valid(S2):
                S = S2
                best = set(S)
    
    # Local search: try swapping
    for _ in range(100):
        if time.time() > deadline:
            break
        improved = False
        for a in range(1, 25):
            if a in S:
                continue
            for b in sorted(S):
                S2 = (S - {b}) | {a}
                if len(S2) > len(best) and _is_valid(S2):
                    S = S2
                    best = set(S)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return list(best)
# EVOLVE-BLOCK-END
