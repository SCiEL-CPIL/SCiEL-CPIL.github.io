"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
import random

def _check_valid(selected):
    """Check if the selected set satisfies the constraint."""
    n = len(selected)
    sel_set = set(selected)
    for i, a in enumerate(selected):
        # Check if a divides sum of any 3 distinct other elements
        others = [selected[j] for j in range(n) if j != i]
        if len(others) < 3:
            continue
        for combo in combinations(others, 3):
            if sum(combo) % a == 0:
                return False
    return True

def _check_addition(selected, new_elem):
    """Check if adding new_elem to selected maintains validity."""
    sel_set = set(selected)
    if new_elem in sel_set:
        return False
    
    # Check: does new_elem divide sum of any 3 other selected elements?
    others = [x for x in selected if x != new_elem]
    if len(others) >= 3:
        for combo in combinations(others, 3):
            if sum(combo) % new_elem == 0:
                return False
    
    # Check: for each existing element a, does a divide sum of any 3 elements from (selected \ {a}) ∪ {new_elem}
    # where the 3 elements must be distinct from a and from each other
    # The 3 elements can include new_elem
    for a in selected:
        # 3-element subsets of (selected \ {a}) that may include new_elem
        # Case 1: 3 from selected \ {a} (already checked before adding new_elem, skip)
        # Case 2: 2 from selected \ {a} + new_elem
        # Case 3: 1 from selected \ {a} + new_elem + ... wait, need 3 distinct others
        # So the 3 others are chosen from (selected \ {a}) ∪ {new_elem}, all distinct from a
        
        # Subsets of size 3 from (selected \ {a}) ∪ {new_elem} that include new_elem:
        # - {new_elem, x, y} where x,y ∈ selected \ {a}, x≠y
        # - {new_elem, x} where x ∈ selected \ {a} ... no, need 3 elements
        
        # So: pick new_elem and 2 from selected \ {a}
        rest = [x for x in selected if x != a]
        for i in range(len(rest)):
            for j in range(i+1, len(rest)):
                s = new_elem + rest[i] + rest[j]
                if s % a == 0:
                    return False
        
        # Also: pick 3 from selected \ {a} that don't include new_elem - already valid
        # And: can we pick new_elem + 2 others where one is... no, new_elem is only one copy
    
    return True

def _greedy_largest_first():
    """Greedy: try adding numbers from largest to smallest."""
    selected = []
    for x in range(192, 0, -1):
        if _check_addition(selected, x):
            selected.append(x)
    return selected

def _local_search(initial, deadline):
    """Local search to try to improve the solution."""
    best = list(initial)
    best_size = len(best)
    best_set = set(best)
    
    # Precompute: for each candidate not in best, and each element in best,
    # check if adding candidate would violate constraints
    
    random.seed(42)
    
    while time.time() < deadline:
        # Try to add a random element not in best
        candidates = [x for x in range(1, 193) if x not in best_set]
        if not candidates:
            break
        
        # Try random swaps: remove one, add one or two
        improved = False
        
        # Try adding one element
        for _ in range(100):
            if time.time() >= deadline:
                break
            cand = random.choice(candidates)
            new_set = list(best) + [cand]
            if _check_valid(new_set):
                best = new_set
                best_set = set(best)
                best_size = len(best)
                candidates = [x for x in range(1, 193) if x not in best_set]
                improved = True
                break
        
        if improved:
            continue
        
        # Try swap: remove one element, add one
        for _ in range(200):
            if time.time() >= deadline:
                break
            rem_idx = random.randint(0, len(best)-1)
            rem = best[rem_idx]
            cand = random.choice([x for x in range(1, 193) if x not in best_set])
            new_set = [best[i] for i in range(len(best)) if i != rem_idx] + [cand]
            if _check_valid(new_set):
                best = new_set
                best_set = set(best)
                candidates = [x for x in range(1, 193) if x not in best_set]
                improved = True
                break
        
        if improved:
            continue
        
        # Try remove one, add two
        for _ in range(100):
            if time.time() >= deadline:
                break
            rem_idx = random.randint(0, len(best)-1)
            rem = best[rem_idx]
            remaining = [best[i] for i in range(len(best)) if i != rem_idx]
            remaining_set = set(remaining)
            cands = [x for x in range(1, 193) if x not in remaining_set and x != rem]
            if len(cands) < 2:
                continue
            c1, c2 = random.sample(cands, 2)
            new_set = remaining + [c1, c2]
            if _check_valid(new_set):
                best = new_set
                best_set = set(best)
                best_size = len(best)
                candidates = [x for x in range(1, 193) if x not in best_set]
                improved = True
                break
        
        if not improved:
            # Try a few more random restarts of local moves
            for _ in range(500):
                if time.time() >= deadline:
                    break
                # Random flip: remove a random element
                if len(best) > 1:
                    rem_idx = random.randint(0, len(best)-1)
                    best = [best[i] for i in range(len(best)) if i != rem_idx]
                    best_set = set(best)
                # Then try to add
                cands = [x for x in range(1, 193) if x not in best_set]
                if cands:
                    cand = random.choice(cands)
                    if _check_addition(best, cand):
                        best.append(cand)
                        best_set.add(cand)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Phase 1: Greedy from largest
    selected = _greedy_largest_first()
    current_best = list(selected)
    
    # Phase 2: Local search to improve
    if time.time() < deadline:
        improved = _local_search(selected, deadline)
        if len(improved) > len(current_best):
            current_best = improved
    
    return current_best
# EVOLVE-BLOCK-END
