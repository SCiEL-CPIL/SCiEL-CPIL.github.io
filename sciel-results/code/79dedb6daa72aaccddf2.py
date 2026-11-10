"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 12, 'm': 40}

def _count_comparable(family):
    """Count unordered pairs where one set is a strict subset of the other."""
    n = len(family)
    # Convert to frozensets for fast subset checking
    fs = [frozenset(s) for s in family]
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if fs[i] < fs[j] or fs[j] < fs[i]:
                count += 1
    return count

def _greedy_construct():
    """
    Greedy construction: start with empty family, add sets one by one
    choosing the set that maximizes the number of comparable pairs with existing family.
    """
    n = P['n']
    m = P['m']
    
    # Generate all 2^12 subsets as frozensets
    all_sets = []
    for mask in range(1 << n):
        s = frozenset(b for b in range(n) if (mask >> b) & 1)
        all_sets.append(s)
    
    # Start with empty set (maximizes comparability)
    family = [frozenset()]
    used = {frozenset()}
    
    for _ in range(m - 1):
        best_set = None
        best_score = -1
        for s in all_sets:
            if s in used:
                continue
            # Count comparable pairs with current family
            score = 0
            for f in family:
                if f < s or s < f:
                    score += 1
            if score > best_score:
                best_score = score
                best_set = s
        if best_set is None:
            break
        family.append(best_set)
        used.add(best_set)
    
    return [list(s) for s in family]

def _local_search(family, max_iterations=10000, deadline=None):
    """
    Local search: try swapping one set in the family with another set
    to improve the count of comparable pairs.
    """
    n = P['n']
    m = P['m']
    current = [frozenset(s) for s in family]
    current_score = _count_comparable(current)
    
    # Generate all possible sets
    all_sets = [frozenset(b for b in range(n) if (mask >> b) & 1) for mask in range(1 << n)]
    all_set_list = list(all_sets)
    
    used = set(current)
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        if deadline and time.time() > deadline - 5:
            break
        
        # Try swapping each set in family with each unused set
        for i in range(m):
            old_set = current[i]
            for new_set in all_set_list:
                if new_set in used:
                    continue
                # Remove old_set, add new_set
                current[i] = new_set
                new_score = _count_comparable(current)
                if new_score > current_score:
                    used.discard(old_set)
                    used.add(new_set)
                    current_score = new_score
                    improved = True
                else:
                    current[i] = old_set
        
        if not improved:
            # Try random restarts within local search
            for _ in range(50):
                if deadline and time.time() > deadline - 5:
                    break
                # Pick a random set to replace
                i = random.randint(0, m - 1)
                old_set = current[i]
                # Pick a random unused set
                unused = [s for s in all_set_list if s not in used]
                if not unused:
                    break
                new_set = random.choice(unused)
                current[i] = new_set
                new_score = _count_comparable(current)
                if new_score >= current_score:
                    used.discard(old_set)
                    used.add(new_set)
                    current_score = new_score
                    improved = True
                else:
                    current[i] = old_set
                if improved:
                    break
    
    return [list(s) for s in current], current_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline, 10s margin
    
    # Start with greedy construction
    family = _greedy_construct()
    best_family = family
    best_score = _count_comparable(family)
    
    # Run local search
    if time.time() < deadline - 10:
        family, score = _local_search(family, max_iterations=5000, deadline=deadline)
        if score > best_score:
            best_family = family
            best_score = score
    
    # Try multiple random restarts if time permits
    restart = 0
    while time.time() < deadline - 15:
        restart += 1
        # Random family initialization
        random.seed(restart * 12345)
        all_sets = [frozenset(b for b in range(P['n']) if (mask >> b) & 1) for mask in range(1 << P['n'])]
        random.shuffle(all_sets)
        # Take a random subset of 40, but ensure diversity
        family = [list(s) for s in all_sets[:40]]
        score = _count_comparable(family)
        if score > best_score:
            best_family = family
            best_score = score
        # Local search on this
        if time.time() < deadline - 10:
            family, score = _local_search(family, max_iterations=2000, deadline=deadline)
            if score > best_score:
                best_family = family
                best_score = score
    
    # Ensure valid output
    result = []
    seen = set()
    for s in best_family:
        fs = frozenset(s)
        if fs not in seen:
            seen.add(fs)
            result.append(list(fs))
    
    # If we have fewer than 40, add more sets
    if len(result) < P['m']:
        all_sets = [frozenset(b for b in range(P['n']) if (mask >> b) & 1) for mask in range(1 << P['n'])]
        for s in all_sets:
            if s not in seen:
                result.append(list(s))
                seen.add(s)
            if len(result) >= P['m']:
                break
    
    return result[:P['m']]
# EVOLVE-BLOCK-END
