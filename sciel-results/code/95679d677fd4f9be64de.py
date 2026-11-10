"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _has_3ap(selected, x):
    """Check if adding x to selected creates a 3-term AP."""
    # Check if x is the middle term: a, x, c where c = 2*x - a
    # Check if x is the first term: x, b, c where c = 2*b - x
    # Check if x is the last term: a, b, x where b = (a+x)/2
    for a in selected:
        # x is middle: a, x, 2*x-a
        c = 2*x - a
        if c != x and c in selected:
            return True
        # x is first: x, (x+c)/2, c - need c such that (x+c)/2 is in selected
        # x is last: a, (a+x)/2, x - need (a+x)/2 to be integer and in selected
        if (a + x) % 2 == 0:
            b = (a + x) // 2
            if b != a and b != x and b in selected:
                return True
    return False

def _reciprocal_sum(selected):
    """Compute exact sum of reciprocals."""
    return sum(Fraction(1, x) for x in selected)

def _greedy_descending():
    """Greedy construction adding elements in descending order of reciprocal value."""
    selected = []
    selected_set = set()
    # Process elements in order of decreasing 1/x (i.e., increasing x is wrong, we want small x first)
    # Actually, to maximize sum of reciprocals, we want to include small numbers
    # So process in ascending order of x (which gives descending order of 1/x)
    for x in range(1, 257):
        if not _has_3ap(selected_set, x):
            selected.append(x)
            selected_set.add(x)
    return selected

def _local_search(selected, time_budget=50):
    """Local hill-climbing search to improve the set."""
    selected_set = set(selected)
    current_sum = _reciprocal_sum(selected)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_budget:
        improved = False
        
        # Try adding elements not in the set
        for x in range(1, 257):
            if x not in selected_set:
                if not _has_3ap(selected_set, x):
                    new_sum = current_sum + Fraction(1, x)
                    if new_sum > current_sum:
                        selected.append(x)
                        selected_set.add(x)
                        current_sum = new_sum
                        improved = True
        
        # Try swapping: remove one element, add another
        for i, x in enumerate(selected):
            if (time.time() - start_time) >= time_budget:
                break
            # Try removing x
            selected_set.remove(x)
            for y in range(1, 257):
                if y not in selected_set and y != x:
                    if not _has_3ap(selected_set, y):
                        new_sum = current_sum - Fraction(1, x) + Fraction(1, y)
                        if new_sum > current_sum:
                            selected[i] = y
                            selected_set.add(y)
                            current_sum = new_sum
                            improved = True
                            break
            # Restore x if not swapped
            if x not in selected_set:
                selected_set.add(x)
    
    return selected, current_sum

def construct():
    start_time = time.time()
    
    # Start with greedy construction
    selected = _greedy_descending()
    
    # Apply local search with time budget
    selected, current_sum = _local_search(selected, time_budget=55)
    
    return selected
# EVOLVE-BLOCK-END
