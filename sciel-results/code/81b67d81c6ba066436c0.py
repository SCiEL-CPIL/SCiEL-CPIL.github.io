import random
import time
from itertools import combinations

def _get_sums(elements):
    """Get all non-empty subset sums as a set."""
    n = len(elements)
    sums = set()
    # Use iterative construction for speed
    # Start with {0}, then add elements
    # But we need non-empty.
    # Efficient way:
    # sums = {0}
    # for x in elements:
    #   sums = sums | {s+x for s in sums}
    # return sums - {0}
    
    current_sums = {0}
    for x in elements:
        new_sums = set()
        for s in current_sums:
            new_sums.add(s + x)
        current_sums.update(new_sums)
    current_sums.discard(0)
    return current_sums

def _is_valid(elements):
    """Check if the set satisfies all constraints."""
    if not elements:
        return True
    # Check distinct
    if len(elements) != len(set(elements)):
        return False
    # Check range
    for x in elements:
        if not (1 <= x <= 256):
            return False
    sums = _get_sums(elements)
    # Check no divisibility
    sums_list = sorted(sums)
    for i in range(len(sums_list)):
        for j in range(i + 1, len(sums_list)):
            if sums_list[j] % sums_list[i] == 0:
                return False
    return True

def _check_addition(current_sums, new_elem):
    """Check if adding new_elem to a set with current_sums maintains validity.
    Returns new_sums if valid, None otherwise."""
    # New sums: current_sums ∪ {s + new_elem for s in current_sums} ∪ {new_elem}
    # Note: current_sums does NOT contain 0.
    # So shifted sums are {s + new_elem for s in current_sums}
    # And {new_elem} is the sum of the subset {new_elem}
    
    new_sums = set(current_sums)
    new_sums.add(new_elem)
    for s in current_sums:
        new_sums.add(s + new_elem)
    
    # Check disjointness (unique sums)
    # If any s + new_elem is in current_sums, or new_elem is in current_sums,
    # or any s1 + new_elem == s2 + new_elem (impossible if s1!=s2)
    # Or s + new_elem == new_elem (implies s=0, not in current_sums)
    
    # Actually, if current_sums has unique sums, and we add new_elem,
    # the new sums are old_sums and old_sums + new_elem.
    # They are disjoint iff no s in old_sums satisfies s + new_elem in old_sums
    # AND new_elem not in old_sums.
    
    if new_elem in current_sums:
        return None
        
    for s in current_sums:
        if s + new_elem in current_sums:
            return None
            
    # Check divisibility
    # We only need to check pairs involving new sums?
    # No, we need to check all pairs in new_sums.
    # But we know old_sums were valid.
    # So we only need to check:
    # 1. new_elem vs all s in old_sums
    # 2. s + new_elem vs all t in old_sums
    # 3. s + new_elem vs t + new_elem (implies s=t, so same sum, but we checked disjointness)
    # 4. s + new_elem vs new_elem (implies s=0, not in old_sums)
    
    # So we just check new elements against old elements.
    new_items = [new_elem] + [s + new_elem for s in current_sums]
    old_items = list(current_sums)
    
    for ni in new_items:
        for oi in old_items:
            if ni > oi:
                if ni % oi == 0:
                    return None
            elif oi > ni:
                if oi % ni == 0:
                    return None
    
    return new_sums

def _greedy_grow(start_elements, deadline, rng):
    """Try to grow the set greedily."""
    elements = list(start_elements)
    sums = _get_sums(elements)
    
    # Try adding elements from 256 down to 1, but shuffled
    candidates = [x for x in range(256, 0, -1) if x not in elements]
    rng.shuffle(candidates)
    
    while time.time() < deadline:
        added = False
        for c in candidates:
            new_sums = _check_addition(sums, c)
            if new_sums is not None:
                elements.append(c)
                sums = new_sums
                candidates.remove(c)
                added = True
                break
        if not added:
            break
    
    return elements

def construct():
    deadline = time.time() + 80  # 80 second budget
    
    best = [100, 101, 102, 103, 104, 105, 106] # Placeholder, will be overwritten
    best_len = 0
    
    rng = random.Random(42)
    
    # Strategy 1: Random restarts with greedy growth
    # Focus on finding size 8
    
    for trial in range(1000):
        if time.time() > deadline:
            break
            
        # Randomly select a starting set of size 3-5
        start_size = rng.randint(3, 5)
        # Bias towards larger numbers
        start_pool = rng.sample(range(100, 257), start_size)
        
        if not _is_valid(start_pool):
            continue
            
        result = _greedy_grow(start_pool, deadline, rng)
        
        if len(result) > best_len:
            best = result
            best_len = len(result)
            
            # If we found size 8, we can try to verify and return early?
            # Or keep searching for 9? Unlikely.
            if best_len >= 8:
                # Verify one last time
                if _is_valid(best):
                    return best

    # If no size 8 found, return best found (likely 7)
    return best
