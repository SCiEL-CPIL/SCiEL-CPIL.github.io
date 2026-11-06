"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _precompute_triples(N=180):
    """
    Precompute all forbidden triples (a, b, c) with distinct a, b, c in [1, N]
    such that 1/a = 1/b + 1/c.
    Returns list of sets of (y,z) pairs for each element a.
    """
    triples_by_element = [set() for _ in range(N + 1)]
    
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            numerator = b * c
            denominator = b + c
            if numerator % denominator == 0:
                a = numerator // denominator
                if a >= 1 and a <= N and a != b and a != c:
                    triples_by_element[a].add((b, c))
                    triples_by_element[b].add((a, c))
                    triples_by_element[c].add((a, b))
    
    return triples_by_element

def _build_conflict_masks(triples_by_element, N=180):
    """
    Build bitset masks for fast conflict checking.
    conflict_mask[x] is a bitset where bit y is set if x and y are in some forbidden triple.
    pair_conflict[x][y] indicates if {x,y} together conflict with some third element.
    """
    # For fast checking: can_add(x, current_mask) checks if adding x creates a conflict
    # A conflict occurs if there exist y,z in current such that (x,y,z) is a forbidden triple
    # i.e., for some (y,z) in triples_by_element[x], both y and z are in current
    
    # Precompute for each x, the list of (y,z) pairs
    # For fast check: for each x, compute a list of bitsets where each bitset represents
    # "if both bits in this pair are set in current, then x conflicts"
    
    # Actually, let's use a different approach:
    # For each element x, precompute a list of "witness" pairs.
    # When checking if x can be added to current (represented as a set or bitmask):
    #   for each (y,z) in triples_by_element[x]: if y in current and z in current: conflict
    
    # For speed, we'll keep triples_by_element as-is and use set membership.
    # The key optimization is to avoid rebuilding sets in local search.
    
    return None  # Placeholder; we'll optimize the checking functions directly

def _can_add(x, current, triples_by_element):
    """Check if x can be added to current without creating a forbidden triple."""
    for (y, z) in triples_by_element[x]:
        if y in current and z in current:
            return False
    return True

def _count_conflicts(x, current, triples_by_element):
    """Count how many conflicts x would have if added to current."""
    count = 0
    for (y, z) in triples_by_element[x]:
        if y in current and z in current:
            count += 1
    return count

def _greedy_descending(N=180, triples_by_element=None):
    """Greedy: include large numbers first."""
    if triples_by_element is None:
        triples_by_element = _precompute_triples(N)
    
    current = set()
    for x in range(N, 0, -1):
        if _can_add(x, current, triples_by_element):
            current.add(x)
    
    return current

def _greedy_by_conflict_order(N=180, triples_by_element=None, start_from=None):
    """
    Greedy: at each step, add the element with fewest conflicts.
    """
    if triples_by_element is None:
        triples_by_element = _precompute_triples(N)
    
    if start_from is not None:
        current = set(start_from)
    else:
        current = set()
    
    candidates = [x for x in range(1, N + 1) if x not in current]
    
    while candidates:
        best_x = None
        best_conflicts = float('inf')
        for x in candidates:
            conflicts = _count_conflicts(x, current, triples_by_element)
            if conflicts < best_conflicts:
                best_conflicts = conflicts
                best_x = x
        
        if best_conflicts > 0:
            break
        
        current.add(best_x)
        candidates.remove(best_x)
    
    return current

def _parity_based_construct(N=180, triples_by_element=None, deadline=None):
    """
    Use parity-based partition. Find max valid subset in evens and odds separately.
    """
    if triples_by_element is None:
        triples_by_element = _precompute_triples(N)
    
    evens = [x for x in range(1, N + 1) if x % 2 == 0]
    odds = [x for x in range(1, N + 1) if x % 2 == 1]
    
    # For each parity class, find max valid subset using greedy with conflict heuristic
    even_subset = _greedy_by_conflict_order(N, triples_by_element, start_from=set(evens))
    odd_subset = _greedy_by_conflict_order(N, triples_by_element, start_from=set(odds))
    
    combined = even_subset | odd_subset
    
    # Verify and fix if needed
    if not all(_can_add(x, combined - {x}, triples_by_element) for x in combined):
        # Rebuild greedily
        combined = set()
        for x in range(N, 0, -1):
            if _can_add(x, combined, triples_by_element):
                combined.add(x)
    
    return combined

def _local_search_improve(subset, N=180, triples_by_element=None, max_iters=500, deadline=None):
    """
    Local search: add elements, then swap to improve.
    """
    if triples_by_element is None:
        triples_by_element = _precompute_triples(N)
    
    current = set(subset)
    improved = True
    iters = 0
    
    while improved and iters < max_iters:
        if deadline and time.time() > deadline:
            break
        improved = False
        iters += 1
        
        # Phase 1: Try adding any element that doesn't conflict
        for y in range(1, N + 1):
            if y in current:
                continue
            if _can_add(y, current, triples_by_element):
                current.add(y)
                improved = True
                break
        
        # Phase 2: Try swapping - remove one, add another
        if not improved:
            found_swap = False
            for x in list(current):
                temp = current - {x}
                for y in range(1, N + 1):
                    if y in temp:
                        continue
                    if _can_add(y, temp, triples_by_element):
                        temp.add(y)
                        current = temp
                        found_swap = True
                        break
                if found_swap:
                    break
            if found_swap:
                improved = True
    
    return current

def _local_search_with_swaps(subset, N=180, triples_by_element=None, max_iters=200, deadline=None):
    """
    More aggressive local search with 2-swap moves.
    """
    if triples_by_element is None:
        triples_by_element = _precompute_triples(N)
    
    current = set(subset)
    best = set(subset)
    best_size = len(best)
    iters = 0
    
    while iters < max_iters:
        if deadline and time.time() > deadline:
            break
        iters += 1
        improved = False
        
        # Try adding
        for y in range(1, N + 1):
            if y in current:
                continue
            if _can_add(y, current, triples_by_element):
                current.add(y)
                if len(current) > best_size:
                    best = set(current)
                    best_size = len(current)
                improved = True
                break
        
        # Try 1-swap
        if not improved:
            found = False
            for x in list(current):
                temp = current - {x}
                for y in range(1, N + 1):
                    if y in temp:
                        continue
                    if _can_add(y, temp, triples_by_element):
                        temp.add(y)
                        current = temp
                        if len(current) > best_size:
                            best = set(current)
                            best_size = len(current)
                        found = True
                        break
                if found:
                    break
            if found:
                improved = True
        
        # Try 2-swap: remove two, add two (limited for speed)
        if not improved and len(current) >= 2:
            elements = list(current)
            # Limit the search space for 2-swaps
            max_x1 = min(10, len(elements))
            max_y = 30
            for i in range(max_x1):
                for j in range(i + 1, max_x1):
                    x1, x2 = elements[i], elements[j]
                    temp = current - {x1, x2}
                    for y1 in range(1, max_y + 1):
                        if y1 in temp:
                            continue
                        if not _can_add(y1, temp, triples_by_element):
                            continue
                        temp2 = temp | {y1}
                        for y2 in range(y1 + 1, max_y + 1):
                            if y2 in temp2:
                                continue
                            if _can_add(y2, temp2, triples_by_element):
                                temp2.add(y2)
                                current = temp2
                                if len(current) > best_size:
                                    best = set(current)
                                    best_size = len(current)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    
    return best

def _multi_start_construct(N=180, triples_by_element=None, deadline=None):
    """
    Try multiple different initial constructions and return the best.
    """
    if triples_by_element is None:
        triples_by_element = _precompute_triples(N)
    
    best_subset = set()
    best_size = 0
    
    # Strategy 1: Greedy descending
    s1 = _greedy_descending(N, triples_by_element)
    if len(s1) > best_size:
        best_subset = s1
        best_size = len(s1)
    
    if time.time() > deadline:
        return best_subset
    
    # Strategy 2: Start from upper half
    upper_half = set(range(91, N + 1))
    s2 = _greedy_by_conflict_order(N, triples_by_element, start_from=upper_half)
    if len(s2) > best_size:
        best_subset = s2
        best_size = len(s2)
    
    if time.time() > deadline:
        return best_subset
    
    # Strategy 3: Parity-based
    s3 = _parity_based_construct(N, triples_by_element, deadline)
    if len(s3) > best_size:
        best_subset = s3
        best_size = len(s3)
    
    if time.time() > deadline:
        return best_subset
    
    # Strategy 4: Start from upper third
    upper_third = set(range(121, N + 1))
    s4 = _greedy_by_conflict_order(N, triples_by_element, start_from=upper_third)
    if len(s4) > best_size:
        best_subset = s4
        best_size = len(s4)
    
    return best_subset

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75 second deadline with 15s safety margin
    
    N = 180
    triples_by_element = _precompute_triples(N)
    
    # Multi-start construction
    best_subset = _multi_start_construct(N, triples_by_element, deadline)
    
    # Local search improvement
    if time.time() < deadline - 5:
        improved = _local_search_improve(best_subset, N, triples_by_element, max_iters=500, deadline=deadline)
        if len(improved) > len(best_subset):
            best_subset = improved
    
    # More aggressive local search if time permits
    if time.time() < deadline - 10:
        improved2 = _local_search_with_swaps(best_subset, N, triples_by_element, max_iters=200, deadline=deadline)
        if len(improved2) > len(best_subset):
            best_subset = improved2
    
    # Final: try multiple restarts with random perturbations if time allows
    if time.time() < deadline - 5:
        best_size = len(best_subset)
        rng = random.Random(42)
        for restart in range(10):
            if time.time() > deadline - 5:
                break
            # Create a perturbed version
            perturbed = set(best_subset)
            num_remove = rng.randint(2, 4)
            elements_to_remove = rng.sample(list(perturbed), min(num_remove, len(perturbed)))
            for e in elements_to_remove:
                perturbed.discard(e)
            
            # Try to fill back up
            filled = _local_search_improve(perturbed, N, triples_by_element, max_iters=100, deadline=deadline)
            if len(filled) > best_size:
                best_subset = filled
                best_size = len(filled)
    
    # Convert to sorted list
    result = sorted(best_subset)
    
    return result
# EVOLVE-BLOCK-END
