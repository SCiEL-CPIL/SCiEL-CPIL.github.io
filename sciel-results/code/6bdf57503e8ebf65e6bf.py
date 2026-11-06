"""Rewritten program: greedy + randomized restarts with efficient constraint checking."""
import time
import random
from itertools import combinations

def _check_constraint(a, others):
    """Check that a does not divide sum of any 3 distinct elements from others."""
    if a <= 0:
        return False
    n = len(others)
    if n < 3:
        return True
    # For large a, the max sum is at most 3*192=576, so if a > 576, always safe
    if a > 576:
        return True
    # Check all triples
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                s = others[i] + others[j] + others[k]
                if s % a == 0:
                    return False
    return True

def _check_add(candidate, selected_set, selected_list):
    """Check if we can add candidate to selected_set."""
    # New set = selected_list + [candidate]
    new_set = selected_set | {candidate}
    new_list = selected_list + [candidate]
    
    # For each element a in new_list, check constraint
    # Only need to check elements that could potentially have constraints
    # An element a is constrained if there exist 3 other elements whose sum is divisible by a
    
    for a in new_list:
        others = [x for x in new_list if x != a]
        if not _check_constraint(a, others):
            return False
    return True

def _greedy_descending(n=192, max_elements=48, deadline=None):
    """Greedy: try numbers from large to small."""
    selected = []
    selected_set = set()
    
    for x in range(n, 0, -1):
        if deadline and time.time() > deadline:
            break
        if len(selected) >= max_elements:
            break
        if _check_add(x, selected_set, selected):
            selected.append(x)
            selected_set.add(x)
    return selected

def _greedy_random(n=192, max_elements=48, deadline=None, seed=None):
    """Greedy with random order."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    candidates = list(range(n, 0, -1))
    rng.shuffle(candidates)
    
    selected = []
    selected_set = set()
    
    for x in candidates:
        if deadline and time.time() > deadline:
            break
        if len(selected) >= max_elements:
            break
        if _check_add(x, selected_set, selected):
            selected.append(x)
            selected_set.add(x)
    return selected

def _greedy_hybrid(n=192, max_elements=48, deadline=None, seed=None):
    """Hybrid: prefer large numbers but with some randomness."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    # Create weighted candidates: larger numbers more likely
    candidates = list(range(n, 0, -1))
    # Shuffle but bias toward larger numbers
    for i in range(len(candidates) - 1, 0, -1):
        # With some probability, swap with a random earlier position
        if rng.random() < 0.3:
            j = rng.randint(0, i)
            candidates[i], candidates[j] = candidates[j], candidates[i]
    
    selected = []
    selected_set = set()
    
    for x in candidates:
        if deadline and time.time() > deadline:
            break
        if len(selected) >= max_elements:
            break
        if _check_add(x, selected_set, selected):
            selected.append(x)
            selected_set.add(x)
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    best = []
    
    # First, try the deterministic descending greedy
    result = _greedy_descending(deadline=deadline)
    if len(result) > len(best):
        best = result
    
    # Try multiple randomized restarts
    seed = 42
    while time.time() < deadline - 5.0:  # Leave 5s safety margin
        result = _greedy_random(deadline=deadline, seed=seed)
        if len(result) > len(best):
            best = result
            seed += 1
        
        result = _greedy_hybrid(deadline=deadline, seed=seed)
        if len(result) > len(best):
            best = result
            seed += 1
    
    return best

# Quick test
if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result)} elements")
    print(result)
    # Verify
    from itertools import combinations
    valid = True
    for a in result:
        others = [x for x in result if x != a]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                valid = False
                print(f"Invalid: {a} divides {sum(t)} from {t}")
                break
        if not valid:
            break
    print(f"Valid: {valid}")
