"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

DEADLINE = 80.0  # safety margin before 90s kill

def _is_valid(selected):
    """Check if selected set is reciprocal-subsum-free."""
    sel = list(selected)
    n = len(sel)
    if n == 0:
        return True
    # For each element i, check if 1/sel[i] equals sum of 1/sel[j] for some subset of other elements
    for i in range(n):
        target = Fraction(1, sel[i])
        others = [sel[j] for j in range(n) if j != i]
        m = len(others)
        # Check all non-empty subsets of others
        for r in range(1, m + 1):
            for subset in combinations(others, r):
                s = Fraction(0, 1)
                for x in subset:
                    s += Fraction(1, x)
                    if s > target:
                        break
                if s == target:
                    return False
    return True

def _is_valid_fast(selected):
    """Faster validity check using early termination and sorting."""
    sel = sorted(selected)
    n = len(sel)
    if n == 0:
        return True
    # Precompute reciprocals as Fractions
    recips = [Fraction(1, x) for x in sel]
    for i in range(n):
        target = recips[i]
        others_idx = [j for j in range(n) if j != i]
        m = len(others_idx)
        # Sort others by reciprocal descending for early termination
        others_sorted = sorted(others_idx, key=lambda j: -recips[j])
        # Check all non-empty subsets
        for r in range(1, m + 1):
            for subset in combinations(others_sorted, r):
                s = Fraction(0, 1)
                for j in subset:
                    s += recips[j]
                    if s > target:
                        break
                if s == target:
                    return False
    return True

def _greedy_construct():
    """Greedy: try to add elements one by one, preferring larger elements."""
    selected = set()
    # Try elements from large to small (larger elements have smaller reciprocals,
    # making it harder for them to be subsums)
    for x in range(24, 0, -1):
        candidate = selected | {x}
        if _is_valid(candidate):
            selected = candidate
    return selected

def _local_search_improve(selected, time_limit):
    """Try to improve the set via local search (add/remove/swap)."""
    selected = set(selected)
    start_time = time.time()
    best = selected
    best_size = len(selected)
    
    while time.time() - start_time < time_limit:
        improved = False
        current = set(selected)
        
        # Try adding elements
        for x in range(1, 25):
            if x not in current:
                candidate = current | {x}
                if _is_valid(candidate):
                    current = candidate
                    improved = True
                    break
        
        if not improved:
            # Try swapping: remove one, add another
            found_swap = False
            for x in list(current):
                for y in range(1, 25):
                    if y not in current:
                        candidate = (current - {x}) | {y}
                        if len(candidate) > len(current) or (len(candidate) == len(current) and _is_valid(candidate)):
                            if _is_valid(candidate):
                                current = candidate
                                found_swap = True
                                break
                if found_swap:
                    break
        
        if len(current) > best_size:
            best = set(current)
            best_size = len(current)
            selected = current
        elif not improved and not found_swap:
            # Stuck, try random perturbation
            import random
            if len(current) > 1:
                to_remove = random.sample(list(current), 1)[0]
                current.discard(to_remove)
                # Try to add something
                for x in range(24, 0, -1):
                    if x not in current:
                        candidate = current | {x}
                        if _is_valid(candidate):
                            current = candidate
                            break
            if len(current) > best_size:
                best = set(current)
                best_size = len(current)
                selected = current
            else:
                # Reset to best if we got worse
                current = set(best)
                if time.time() - start_time > time_limit * 0.7:
                    break
    
    return best

def _systematic_search():
    """Try to find a large valid set using systematic approach."""
    # Start with greedy
    best = _greedy_construct()
    best_size = len(best)
    
    # Try different orderings
    for start in range(24, 0, -1):
        selected = set()
        for x in range(start, 0, -1):
            candidate = selected | {x}
            if _is_valid(candidate):
                selected = candidate
        if len(selected) > best_size:
            best = selected
            best_size = len(selected)
    
    # Try starting with small elements
    for start in range(1, 25):
        selected = set()
        for x in range(start, 25):
            candidate = selected | {x}
            if _is_valid(candidate):
                selected = candidate
        if len(selected) > best_size:
            best = selected
            best_size = len(selected)
    
    return best

def construct():
    start_time = time.time()
    
    # Get a good initial solution via multiple strategies
    candidates = []
    
    # Greedy from large to small
    c1 = _greedy_construct()
    candidates.append(c1)
    
    # Systematic search with different orderings
    c2 = _systematic_search()
    candidates.append(c2)
    
    # Pick the best candidate
    best = max(candidates, key=len)
    
    # Local search to improve
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 5:
        improved = _local_search_improve(best, remaining - 5)
        if len(improved) > len(best):
            best = improved
    
    # Ensure non-empty
    if not best:
        return [24]
    
    return sorted(best)
# EVOLVE-BLOCK-END
