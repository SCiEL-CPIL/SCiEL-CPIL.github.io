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
        # To speed up, we can just check all, it's only 4096 * 40 * 40 ~ 6.5M ops, which is fast enough in Python if optimized?
        # Actually 6.5M Python loop iterations might take ~1-2 seconds. Acceptable.
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

def _local_search(family, max_iterations=100, deadline=None):
    """
    Local search: try swapping one set in the family with another set
    to improve the count of comparable pairs.
    Optimized to run within time limits by sampling candidates.
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
    
    # Number of random candidates to try per position per pass
    sample_size = 50
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        if deadline and time.time() > deadline - 2:
            break
        
        # Try swapping each set in family with a sample of unused sets
        for i in range(m):
            if deadline and time.time() > deadline - 2:
                break
            old_set = current[i]
            # Get a random sample of unused sets to try
            unused_sets = [s for s in all_set_list if s not in used]
            if not unused_sets:
                continue
            candidates = random.sample(unused_sets, min(sample_size, len(unused_sets)))
            
            best_new_set = None
            best_new_score = current_score
            
            for new_set in candidates:
                # Remove old_set, add new_set
                current[i] = new_set
                new_score = _count_comparable(current)
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_new_set = new_set
                else:
                    current[i] = old_set
            
            if best_new_set is not None:
                used.discard(old_set)
                used.add(best_new_set)
                current[i] = best_new_set
                current_score = best_new_score
                improved = True
        
        if not improved:
            # If no improvement in a full pass, do a few random kicks to escape local minima
            for _ in range(10):
                if deadline and time.time() > deadline - 2:
                    break
                # Pick a random set to replace
                i = random.randint(0, m - 1)
                old_set = current[i]
                # Pick a random unused set
                unused_sets = [s for s in all_set_list if s not in used]
                if not unused_sets:
                    break
                new_set = random.choice(unused_sets)
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
    deadline = start_time + 50  # 50 second deadline, 10s margin for safety
    
    # Start with greedy construction
    family = _greedy_construct()
    best_family = family
    best_score = _count_comparable(family)
    
    # Run local search
    if time.time() < deadline - 5:
        family, score = _local_search(family, max_iterations=50, deadline=deadline)
        if score > best_score:
            best_family = family
            best_score = score
    
    # Try multiple random restarts if time permits
    restart = 0
    while time.time() < deadline - 10:
        restart += 1
        # Random family initialization
        random.seed(restart * 12345)
        all_sets = [frozenset(b for b in range(P['n']) if (mask >> b) & 1) for mask in range(1 << P['n'])]
        random.shuffle(all_sets)
        # Take a random subset of 40
        family = [list(s) for s in all_sets[:40]]
        score = _count_comparable(family)
        if score > best_score:
            best_family = family
            best_score = score
        # Local search on this
        if time.time() < deadline - 5:
            family, score = _local_search(family, max_iterations=20, deadline=deadline)
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
