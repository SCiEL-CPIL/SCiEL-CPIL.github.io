"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
import time
import random
import sys

def _compute_sums(selected):
    """Compute all pairwise sums a+b for a<=b in selected."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        sums.add(2 * a)
        for j in range(i + 1, n):
            sums.add(a + selected[j])
    return sums

def _is_sidon(selected):
    """Check if selected is a Sidon set."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        if 2 * a in sums:
            return False
        sums.add(2 * a)
        for j in range(i + 1, n):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, sums):
    """Check if selected is inclusion-maximal. Returns True if maximal."""
    selected_set = set(selected)
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        # Check if adding x breaks Sidon property
        # x+x must not be in sums
        if 2 * x in sums:
            continue
        # x + a for a in selected must not be in sums
        # and must be distinct from each other (which they are if x is fixed and a are distinct)
        # and must not be equal to 2x (already checked)
        valid = True
        for a in selected:
            s = x + a
            if s in sums:
                valid = False
                break
        if valid:
            return False
    return True

def _greedy_construct():
    """Generate an initial maximal Sidon set using ascending greedy."""
    selected, sums = [], set()
    for x in range(1, N + 1):
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected, sums

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    # 1. Generate initial maximal Sidon set
    selected, sums = _greedy_construct()
    best_set = list(selected)
    best_sums = sums
    best_len = len(selected)
    
    # 2. Local Search: Try to reduce cardinality by swapping
    # Strategy: 
    # - Identify elements that are "redundant" in terms of maximality? 
    #   Actually, we can't just remove an element because it might break maximality.
    #   If we remove an element 'a', the set is still Sidon. 
    #   But it might not be maximal anymore.
    #   If it's not maximal, there exists some 'x' that can be added.
    #   If we add 'x', the set size is the same. 
    #   However, maybe we can remove 'a' and add TWO elements? No, that increases size.
    #   
    #   The key insight for minimal maximal Sidon sets:
    #   We want a set S such that S is Sidon, and for every x not in S, S U {x} is not Sidon.
    #   This means the "shadow" of S (sums) covers all possible "extensions".
    #   
    #   A move that reduces size:
    #   If we remove 'a', and the set S' = S \ {a} is NOT maximal, there is some x that can be added.
    #   If we add x, we get S'' = S' U {x}. Size is same.
    #   This doesn't help reduce size.
    #   
    #   Wait, how can we reduce size?
    #   We cannot reduce size by removing elements if maximality is required, UNLESS we can replace one element with none? No.
    #   Or replace one element with a different one that allows us to remove ANOTHER element?
    #   
    #   Let's look at the structure.
    #   If S is maximal, every x not in S conflicts with S.
    #   If we remove a, S' is Sidon.
    #   If S' is maximal, then S' is a smaller maximal Sidon set.
    #   So, simply trying to remove elements is a valid strategy IF the resulting set is still maximal.
    #   Usually, removing an element makes the set non-maximal (there are holes).
    #   But sometimes, the element 'a' was the only thing preventing multiple x's from being added, 
    #   or its removal doesn't open up any new valid x's?
    #   If removing 'a' doesn't open up any valid x's, then S' is maximal and smaller.
    #   
    #   So, Step A: Try to remove elements one by one.
    #   If removal preserves maximality, keep it.
    
    # 3. Iterative removal
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(selected) - 1, -1, -1):
            a = selected[i]
            # Try removing a
            new_selected = selected[:i] + selected[i+1:]
            new_sums = _compute_sums(new_selected)
            
            # Check if new_selected is maximal
            if _is_maximal(new_selected, new_sums):
                selected = new_selected
                sums = new_sums
                best_set = list(selected)
                best_len = len(selected)
                improved = True
                break # Restart loop as indices changed
    
    # 4. If no simple removal works, try swaps?
    # A swap (remove a, add b) keeps size same. Doesn't help minimize cardinality directly.
    # But maybe a swap enables a subsequent removal?
    # Let's try a more aggressive local search:
    # Randomly remove an element. If not maximal, try to "fix" it by adding a different element?
    # No, that increases or maintains size.
    
    # Actually, the "removal" strategy above is the most direct way to reduce cardinality.
    # If we can't remove any single element while maintaining maximality, we are in a local minimum.
    
    # Let's add a randomization component to escape local minima.
    # Random restarts with greedy + removal?
    
    # Let's try a different initial construction: Random Greedy?
    # Or just multiple random restarts of the removal process?
    # The removal process is deterministic given the set.
    
    # Let's try to find a set that is "close" to non-maximal but we can swap to make it maximal with fewer elements?
    # This is complex.
    
    # Let's stick to: 
    # 1. Generate multiple random maximal Sidon sets (using random ordering for greedy).
    # 2. Apply the "removal" heuristic to each.
    # 3. Keep the best.
    
    best_overall = best_set
    
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break
        
        # Random Greedy Construction
        random.shuffle(range(1, N + 1)) # This shuffles the list in place, need to copy or use permutation
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        selected, sums = [], set()
        for x in order:
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        
        # This set is maximal Sidon.
        
        # Apply removal heuristic
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(len(selected) - 1, -1, -1):
                a = selected[i]
                new_selected = selected[:i] + selected[i+1:]
                new_sums = _compute_sums(new_selected)
                
                if _is_maximal(new_selected, new_sums):
                    selected = new_selected
                    sums = new_sums
                    improved = True
                    break
        
        if len(selected) < len(best_overall):
            best_overall = list(selected)
            
    return best_overall
# EVOLVE-BLOCK-END
