"""B_3[1] set construction: maximize distinct positive integers <=4096
where every unordered triple (with repetitions) has a unique sum."""

import time
import random

def _check_new_triples(existing, x):
    """Check if adding x to existing set creates any duplicate triple sum.
    Only checks triples that include x (since existing set is already valid).
    Returns True if x can be added without conflict."""
    n = len(existing)
    # Collect all new triple sums involving x
    # (x,x,x)
    seen = {3 * x}
    # (x,x,y) for each y in existing
    for y in existing:
        s = 2 * x + y
        if s in seen:
            return False
        seen.add(s)
    # (x,y,y) for each y in existing
    for y in existing:
        s = x + 2 * y
        if s in seen:
            return False
        seen.add(s)
    # (x,y,z) for y < z in existing
    for i in range(n):
        y = existing[i]
        for j in range(i + 1, n):
            z = existing[j]
            s = x + y + z
            if s in seen:
                return False
            seen.add(s)
    return True

def construct():
    deadline = time.time() + 80  # 80s safety margin before 90s kill
    best_set = []
    
    # Try multiple random orderings and greedy builds
    best_size = 0
    best_result = []
    
    # Strategy 1: Greedy from small to large (deterministic baseline)
    selected = []
    for x in range(1, 4097):
        if time.time() > deadline:
            break
        if _check_new_triples(selected, x):
            selected.append(x)
            if len(selected) > best_size:
                best_size = len(selected)
                best_result = list(selected)
    
    # Strategy 2: Randomized restarts with different orderings
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        # Random permutation of candidates, but bias towards smaller numbers
        # to allow more room for larger ones later
        candidates = list(range(1, 4097))
        # Partial shuffle: keep first ~200 in order, shuffle rest
        shuffle_start = 200
        candidates[shuffle_start:] = sorted(rng.sample(candidates[shuffle_start:], 
                                                       len(candidates) - shuffle_start))
        
        selected = []
        for x in candidates:
            if time.time() > deadline - 2:
                break
            if _check_new_triples(selected, x):
                selected.append(x)
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_result = list(selected)
        
        # If we got a good result, try to extend it greedily
        if best_size > 30:
            # Try adding remaining small numbers that weren't in the set
            current_set = set(selected)
            for x in range(1, 4097):
                if time.time() > deadline - 1:
                    break
                if x not in current_set:
                    if _check_new_triples(selected, x):
                        selected.append(x)
                        current_set.add(x)
                        if len(selected) > best_size:
                            best_size = len(selected)
                            best_result = list(selected)
    
    # Return the best result found
    if best_result:
        return best_result
    return [1, 2, 4]  # Fallback (should never happen)
