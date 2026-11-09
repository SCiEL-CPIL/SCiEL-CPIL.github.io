"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def _has_3ap_in_list(lst):
    """Check if a sorted list has any nonconstant 3-term AP."""
    s = set(lst)
    n = len(lst)
    for i in range(n):
        for j in range(i+1, n):
            a, b = lst[i], lst[j]
            c = 2*b - a
            if c in s and c != a:
                return True
    return False

def _reciprocal_sum(lst):
    """Exact sum of reciprocals as Fraction."""
    total = Fraction(0)
    for x in lst:
        total += Fraction(1, x)
    return total

def _can_add(candidate, selected):
    """Check if adding candidate to selected creates a 3-AP.
    For a new element x, a 3-AP involving x must be:
      x, y, z where 2y = x+z (x is first)
      y, x, z where 2x = y+z (x is middle)
      y, z, x where 2z = y+x (x is last)
    """
    s = set(selected)
    x = candidate
    # x is first: x, y, z -> 2y = x + z -> z = 2y - x
    for y in selected:
        if y > x:
            z = 2*y - x
            if z in s:
                return False
    # x is middle: y, x, z -> 2x = y + z -> z = 2x - y
    for y in selected:
        if y != x:
            z = 2*x - y
            if z in s:
                return False
    # x is last: y, z, x -> 2z = y + x -> z = (y+x)/2, need y+x even
    for y in selected:
        if (y + x) % 2 == 0:
            z = (y + x) // 2
            if z in s and z != x:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    # Phase 1: Greedy construction - add elements in order of largest reciprocal weight
    # i.e., smallest indices first (1/1 > 1/2 > 1/3 > ...)
    # This gives a good baseline 3-AP-free set.
    
    # Actually, let's think more carefully. We want to maximize sum of 1/x.
    # Greedy: try to include 1, then 2, then 3, etc.
    # But including small numbers constrains us more.
    # Let's try greedy smallest-first.
    
    selected = []
    selected_set = set()
    
    for x in range(1, 257):
        if time.time() > deadline:
            break
        if _can_add(x, selected):
            selected.append(x)
            selected_set.add(x)
    
    best_set = list(selected)
    best_score = _reciprocal_sum(best_set)
    
    # Phase 2: Local search - try to swap elements
    # For each element in selected, try replacing it with a larger element not in selected
    # that maintains 3-AP-free property and improves the score.
    # Since we want to maximize sum of 1/x, we prefer smaller x.
    # So the greedy already prefers small x. The local search should try to
    # see if removing a large element allows adding a smaller one that was blocked.
    
    # Better local search: try removing each element and see if we can add back a smaller one.
    # Or: try swapping a large element for a smaller one not currently in the set.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set = list(best_set)
        current_set_set = set(current_set)
        
        # Try each element as a candidate for removal
        for i, elem in enumerate(current_set):
            if time.time() > deadline:
                break
            
            # Try removing elem and adding a smaller element not in set
            # that was previously blocked
            without_elem = current_set_set - {elem}
            without_elem_list = [x for x in current_set if x != elem]
            
            # Try to add smaller elements
            for new_elem in range(1, elem):
                if new_elem in current_set_set:
                    continue
                if _can_add(new_elem, without_elem_list):
                    new_set = sorted(without_elem_list + [new_elem])
                    new_score = _reciprocal_sum(new_set)
                    if new_score > best_score:
                        best_set = new_set
                        best_score = new_score
                        current_set = new_set
                        current_set_set = set(current_set)
                        improved = True
                        break  # restart the outer loop
            if improved:
                break
    
    # Phase 3: Try adding elements that might have been skipped
    # (in case greedy order missed something)
    for x in range(1, 257):
        if time.time() > deadline:
            break
        if x not in best_set:
            if _can_add(x, best_set):
                best_set = sorted(best_set + [x])
                best_score = _reciprocal_sum(best_set)
    
    # Phase 4: More aggressive local search - try all swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set = list(best_set)
        current_set_set = set(current_set)
        
        # Try swapping: remove one element, add another
        for i, elem in enumerate(current_set):
            if time.time() > deadline:
                break
            
            without_elem = current_set_set - {elem}
            without_elem_list = [x for x in current_set if x != elem]
            
            # Try adding any element not in set (both smaller and larger)
            for new_elem in range(1, 257):
                if new_elem in current_set_set:
                    continue
                if _can_add(new_elem, without_elem_list):
                    new_set = sorted(without_elem_list + [new_elem])
                    new_score = _reciprocal_sum(new_set)
                    if new_score > best_score:
                        best_set = new_set
                        best_score = new_score
                        current_set = new_set
                        current_set_set = current_set_set
                        improved = True
                        break
            if improved:
                break
    
    return best_set
# EVOLVE-BLOCK-END
