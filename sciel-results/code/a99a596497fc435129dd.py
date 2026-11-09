"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

DEADLINE = time.time() + 80  # 80s safety margin before 90s kill

def _is_valid_subset_sums(elements):
    """Check if all non-empty subset sums form an antichain under divisibility."""
    n = len(elements)
    if n == 0:
        return True
    # Generate all non-empty subset sums
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sums.add(s)
    
    # Check no two distinct sums divide each other
    sorted_sums = sorted(sums)
    for i in range(len(sorted_sums)):
        for j in range(i + 1, len(sorted_sums)):
            if sorted_sums[j] % sorted_sums[i] == 0:
                return False
    return True

def _incremental_check(elements, new_elem, existing_sums):
    """Check if adding new_elem to elements keeps the property.
    existing_sums is the set of all non-empty subset sums of elements.
    Returns (valid, new_sums) if valid, else (False, None)."""
    new_sums = set(existing_sums)
    # New subset sums are: new_elem alone, and new_elem + s for each s in existing_sums
    new_sums.add(new_elem)
    for s in existing_sums:
        new_sums.add(s + new_elem)
    
    # Check divisibility: for each pair of distinct sums, neither divides the other
    sorted_sums = sorted(new_sums)
    for i in range(len(sorted_sums)):
        si = sorted_sums[i]
        for j in range(i + 1, len(sorted_sums)):
            sj = sorted_sums[j]
            if sj % si == 0:
                return False, None
    return True, new_sums

def _compute_sums(elements):
    """Compute all non-empty subset sums."""
    n = len(elements)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sums.add(s)
    return sums

def _greedy_build(candidates, max_size=30):
    """Greedy build: add elements one by one if valid."""
    current = []
    current_sums = set()
    for elem in candidates:
        if len(current) >= max_size:
            break
        valid, new_sums = _incremental_check(current, elem, current_sums)
        if valid:
            current.append(elem)
            current_sums = new_sums
    return current, current_sums

def construct():
    start_time = time.time()
    best_set = [31, 32, 34]
    best_size = 3
    
    # Try multiple random orderings of candidate elements
    # Focus on larger elements for better divisibility properties
    all_candidates = list(range(1, 257))
    
    # Strategy 1: Greedy with different orderings
    for attempt in range(50):
        if time.time() > DEADLINE:
            break
        
        if attempt == 0:
            # Try descending order (largest first)
            candidates = sorted(all_candidates, reverse=True)
        elif attempt == 1:
            # Try a biased random order favoring larger numbers
            candidates = sorted(all_candidates, key=lambda x: -x + random.random() * 50)
        else:
            # Random order
            candidates = all_candidates[:]
            random.shuffle(candidates)
        
        current, current_sums = _greedy_build(candidates, max_size=30)
        
        if len(current) > best_size:
            best_set = current[:]
            best_size = len(current)
    
    # Strategy 2: Local search - try to swap elements to improve
    if best_size < 30 and time.time() < DEADLINE:
        for _ in range(200):
            if time.time() > DEADLINE:
                break
            
            current = best_set[:]
            current_sums = _compute_sums(current)
            improved = False
            
            # Try adding a random element
            used = set(current)
            candidates = [x for x in range(1, 257) if x not in used]
            random.shuffle(candidates)
            
            for elem in candidates[:20]:  # Try up to 20 random candidates
                valid, new_sums = _incremental_check(current, elem, current_sums)
                if valid:
                    current.append(elem)
                    current_sums = new_sums
                    improved = True
                    if len(current) > best_size:
                        best_set = current[:]
                        best_size = len(current)
                    break
            
            # Try removing one element and adding two
            if not improved and len(current) > 2 and time.time() < DEADLINE:
                for remove_idx in range(min(3, len(current))):
                    if time.time() > DEADLINE:
                        break
                    test_set = current[:remove_idx] + current[remove_idx+1:]
                    test_sums = _compute_sums(test_set)
                    
                    used = set(test_set)
                    candidates = [x for x in range(1, 257) if x not in used]
                    random.shuffle(candidates)
                    
                    for elem in candidates[:15]:
                        valid, new_sums = _incremental_check(test_set, elem, test_sums)
                        if valid:
                            test_set.append(elem)
                            test_sums = new_sums
                            # Try to add one more
                            used2 = set(test_set)
                            candidates2 = [x for x in range(1, 257) if x not in used2]
                            random.shuffle(candidates2)
                            for elem2 in candidates2[:15]:
                                valid2, new_sums2 = _incremental_check(test_set, elem2, test_sums)
                                if valid2:
                                    test_set.append(elem2)
                                    if len(test_set) > best_size:
                                        best_set = test_set[:]
                                        best_size = len(test_set)
                                    break
                            break
    
    # Ensure we return a valid set
    if not _is_valid_subset_sums(best_set):
        # Fallback to a known valid small set
        best_set = [31, 32, 34]
    
    return list(best_set)
# EVOLVE-BLOCK-END
