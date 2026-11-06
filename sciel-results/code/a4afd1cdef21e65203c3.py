import time
import random
from itertools import combinations

def _is_valid(selected):
    """Check if no element in selected divides the sum of any 3 distinct OTHER elements."""
    n = len(selected)
    if n < 4:
        return True
    s = set(selected)
    for a in selected:
        others = [x for x in selected if x != a]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _can_add(selected, x):
    """Check if x can be added to selected without violating the condition."""
    # After adding x, we need to check:
    # 1. x does not divide sum of any 3 distinct others (from selected)
    # 2. No element a in selected divides sum of any 3 from (selected \ {a}) ∪ {x}
    #    i.e., any triple from selected\{a} plus possibly x
    
    if len(selected) < 3:
        # If we have fewer than 3 others, no triple of OTHERS can be formed that excludes x
        # but we need 3 distinct OTHERS for x to be tested. If len(selected) < 3, x has < 3 others, OK.
        # Also need to check existing elements: adding x could create new triples containing x.
        pass
    
    # Check 1: x does not divide sum of any 3 from selected
    if len(selected) >= 3:
        for t in combinations(selected, 3):
            if sum(t) % x == 0:
                return False
    
    # Check 2: For each a in selected, check all triples from (selected \ {a}) that may include x
    # A triple from selected\{a} could be:
    #   - 3 from selected\{a} (already checked if _is_valid was true before, but let's be safe)
    #   - 2 from selected\{a} and x
    #   - 1 from selected\{a} and... no, we need 3 distinct elements
    # So new triples involving x: choose 2 from selected\{a}, plus x
    
    others = [v for v in selected if v != x]  # this is just selected since x not in it yet
    for a in selected:
        # Triples that include x and 2 elements from selected\{a}
        rest = [v for v in selected if v != a]
        if len(rest) >= 2:
            for t2 in combinations(rest, 2):
                if (t2[0] + t2[1] + x) % a == 0:
                    return False
        # Also need to check triples entirely from selected\{a} that we haven't checked?
        # Those were already valid if selected was valid. So skip.
    
    return True

def _greedy_order(candidates, selected, rng):
    """Order candidates by some heuristic to pick which to try first."""
    # Try larger numbers first (they're less likely to divide sums)
    # But also consider: numbers that are "harder to add" should be tried later?
    # Actually, for maximizing cardinality, we want to pick numbers that are "compatible" with many others.
    # A simple heuristic: pick from larger to smaller (larger numbers divide fewer sums)
    # Or: randomize with bias toward larger numbers
    
    # Strategy: sort by descending value, but with some randomization
    # Actually, let's just try from largest to smallest first
    return sorted(candidates, key=lambda x: -x)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    best_selected = []
    
    # Initial greedy: add numbers from 192 down to 1
    selected = []
    for x in range(192, 0, -1):
        if time.time() > deadline - 10:
            break
        if _can_add(selected, x):
            selected.append(x)
    
    best_selected = list(selected)
    best_len = len(selected)
    
    if time.time() > deadline - 20:
        return best_selected
    
    # Try to improve: local search / restarts
    # Idea: remove some elements and try to add others
    # Or: try different orderings
    
    rng = random.Random(42)
    
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        # Try a randomized greedy: shuffle candidates but bias toward larger
        candidates = list(range(192, 0, -1))
        
        # Perturbation: randomly skip some initial candidates
        selected = []
        for x in candidates:
            if time.time() > deadline - 10:
                break
            # With some probability, skip this candidate to allow different selection
            if rng.random() < 0.1 and len(selected) > 5:
                continue
            if _can_add(selected, x):
                selected.append(x)
        
        if len(selected) > best_len:
            best_len = len(selected)
            best_selected = list(selected)
            # Try to improve further from this state
            # Local search: try swapping elements
            for _ in range(20):
                if time.time() > deadline - 10:
                    break
                # Try removing a random element and adding a better one
                if len(selected) == 0:
                    break
                idx = rng.randrange(len(selected))
                removed = selected[idx]
                selected.pop(idx)
                # Try to add something better
                for x in range(192, removed, -1):
                    if x not in selected:
                        if _can_add(selected, x):
                            selected.append(x)
                            break
                else:
                    # Couldn't add anything better, restore
                    selected.append(removed)
                    # Try from below
                    for x in range(removed - 1, 0, -1):
                        if x not in selected:
                            if _can_add(selected, x):
                                selected.append(x)
                                break
                    else:
                        # Restore original
                        pass
                # Re-sort for consistency (not strictly needed)
        
        # Also try: take best_selected and try to improve it
        if len(best_selected) > 0:
            trial = list(best_selected)
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                # Try to swap out one element for two others? Complex.
                # Try removing one and adding one that allows more
                for i in range(len(trial)):
                    if time.time() > deadline - 10:
                        break
                    a = trial[i]
                    trial2 = [v for v in trial if v != a]
                    # Try adding numbers not in trial2
                    added = False
                    for x in range(192, 0, -1):
                        if x not in trial2:
                            if _can_add(trial2, x):
                                if len(trial2) + 1 > len(trial):
                                    trial = trial2 + [x]
                                    improved = True
                                    added = True
                                    break
                            # Don't break if same length, keep trying
                    if added:
                        break
            
            if len(trial) > best_len:
                best_len = len(trial)
                best_selected = list(trial)
    
    return best_selected
