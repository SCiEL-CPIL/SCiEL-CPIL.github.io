"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_valid(selected):
    """Check if no element in selected divides the sum of any 3 distinct others."""
    n = len(selected)
    for i, a in enumerate(selected):
        others = [v for j, v in enumerate(selected) if j != i]
        for combo in combinations(others, 3):
            if sum(combo) % a == 0:
                return False
    return True

def _greedy_construct():
    """Greedy: try adding largest numbers first, checking validity."""
    selected = []
    for x in range(192, 0, -1):
        candidate = selected + [x]
        # Only need to check new combinations involving x
        # For each a in candidate, check if a divides sum of 3 distinct others
        # But we can be smarter: only check if adding x creates a violation
        # A violation is: exists a in selected such that a divides sum(x + two others from selected)
        # OR x divides sum of 3 others from selected
        # OR x divides sum(a + two others) for some a in selected
        
        # Check if x divides sum of any 3 from selected
        if len(selected) >= 3:
            for combo in combinations(selected, 3):
                if sum(combo) % x == 0:
                    break
            else:
                # x doesn't divide any 3-sum from selected
                # Now check if any a in selected divides sum involving x
                ok = True
                for a in selected:
                    # a divides x + b + c for some b,c in selected, b!=c, b,c != a, x not in selected yet
                    # i.e., a | (x + b + c) where b,c are distinct elements of selected, both != a
                    others = [v for v in selected if v != a]
                    if len(others) >= 2:
                        for b_idx, b in enumerate(others):
                            for c in others[b_idx+1:]:
                                if (x + b + c) % a == 0:
                                    ok = False
                                    break
                            if not ok:
                                break
                    if not ok:
                        break
                if ok:
                    selected.append(x)
            continue
        else:
            # Less than 3 in selected, just add x
            selected.append(x)
    return selected

def _local_search(selected, deadline):
    """Try to improve the set by swapping elements or adding/removing."""
    best = list(selected)
    best_size = len(best)
    
    # Precompute for speed
    def is_valid_fast(sel):
        n = len(sel)
        for i, a in enumerate(sel):
            others = [v for j, v in enumerate(sel) if j != i]
            for combo in combinations(others, 3):
                if sum(combo) % a == 0:
                    return False
        return True
    
    # Try to add elements
    for x in range(192, 0, -1):
        if time.time() > deadline:
            break
        if x in best:
            continue
        candidate = best + [x]
        if _is_valid(candidate):
            best = candidate
            best_size += 1
    
    # Try swaps: remove one, add one
    for _ in range(50):
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(best)):
            if time.time() > deadline:
                break
            for x in range(192, 0, -1):
                if x in best:
                    continue
                candidate = [v for j, v in enumerate(best) if j != i] + [x]
                if _is_valid(candidate):
                    best = candidate
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 15s safety margin
    
    # Greedy construction
    selected = _greedy_construct()
    
    # Local search to improve
    if time.time() < deadline:
        selected = _local_search(selected, deadline)
    
    return selected
# EVOLVE-BLOCK-END
