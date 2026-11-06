"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _is_3ap_free(S):
    """Check if set S has no 3-term AP."""
    Sset = set(S)
    for a in S:
        for b in S:
            if a < b:
                c = 2*b - a
                if c in Sset and c > b:
                    return False
    return True

def _reciprocal_sum(S):
    """Compute exact reciprocal sum as Fraction."""
    total = Fraction(0, 1)
    for x in S:
        total += Fraction(1, x)
    return total

def _valid_addition(S_set, x):
    """Check if adding x to S_set creates a 3-AP."""
    for a in S_set:
        if a < x:
            # x could be the middle: a, x, c where c = 2x - a
            c = 2*x - a
            if c in S_set and c > x:
                return False
            # x could be the end: a, b, x where b = (a+x)/2
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if b in S_set and a < b < x:
                    return False
        elif a > x:
            # x could be the start: x, b, a where b = (x+a)/2
            s = x + a
            if s % 2 == 0:
                b = s // 2
                if b in S_set and x < b < a:
                    return False
                # x could be the middle: x, a, c where c = 2a - x
                c = 2*a - x
                if c in S_set and c > a:
                    return False
    return True

def construct():
    deadline = time.time() + 85  # safety margin
    
    # Phase 1: Greedy construction (include small numbers for high reciprocal weight)
    selected = []
    S_set = set()
    for x in range(1, 257):
        if _valid_addition(S_set, x):
            selected.append(x)
            S_set.add(x)
    
    best_set = set(selected)
    best_score = _reciprocal_sum(selected)
    
    # Phase 2: Local search - try swaps to improve reciprocal sum
    # Strategy: try removing a large element and adding a small one
    all_numbers = set(range(1, 257))
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        # Try all possible single-element swaps: remove r, add a
        # Only consider a not in set, r in set
        for r in sorted(best_set, reverse=True):  # remove larger elements
            if time.time() >= deadline - 2:
                break
            for a in range(1, 257):
                if a in best_set:
                    continue
                # Check if swap is valid
                new_set = best_set - {r} | {a}
                if _is_3ap_free(new_set):
                    new_score = _reciprocal_sum(new_set)
                    if new_score > best_score:
                        best_score = new_score
                        best_set = new_set
                        improved = True
                        break  # restart inner loop with new set
    
    # Phase 3: Try adding elements that fit
    if time.time() < deadline - 1:
        for a in range(1, 257):
            if a in best_set:
                continue
            new_set = best_set | {a}
            if _is_3ap_free(new_set):
                new_score = _reciprocal_sum(new_set)
                if new_score > best_score:
                    best_score = new_score
                    best_set = new_set
    
    # Phase 4: More aggressive local search - try removing two and adding one, etc.
    if time.time() < deadline - 1:
        for r1 in sorted(best_set, reverse=True):
            if time.time() >= deadline - 1:
                break
            for r2 in sorted(best_set, reverse=True):
                if r2 >= r1:
                    break
                if time.time() >= deadline - 1:
                    break
                for a in range(1, 257):
                    if a in best_set:
                        continue
                    new_set = best_set - {r1, r2} | {a}
                    if _is_3ap_free(new_set):
                        new_score = _reciprocal_sum(new_set)
                        if new_score > best_score:
                            best_score = new_score
                            best_set = new_set
                            break
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
