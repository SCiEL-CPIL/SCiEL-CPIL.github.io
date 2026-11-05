"""Reciprocal weight of three-progression-free sets."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def _has_3ap(a, b, c):
    """Check if a, b, c form a 3-term AP (distinct values)."""
    if a == b or b == c or a == c:
        return False
    return 2 * b == a + c

def _conflicts_with(x, selected_set):
    """Check if adding x to selected_set creates a 3-term AP.
    For each pair (a, b) in selected_set with a < b:
      - if 2*b - a == x, then (a, b, x) is a 3-AP
      - if a + x = 2*b, i.e., x = 2*b - a (same check)
    Also check x as middle: for a, b in selected_set, if a + b = 2*x
    Also check x as first: for a, b in selected_set, if x + b = 2*a -> x = 2*a - b
    """
    # x as third term: exists a, b in S with a < b and 2b - a = x
    # x as middle term: exists a, b in S with a < b and a + b = 2x
    # x as first term: exists a, b in S with a < b and x + b = 2a, i.e., x = 2a - b
    
    for a in selected_set:
        for b in selected_set:
            if a == b:
                continue
            # x is third: 2b - a = x
            if 2 * b - a == x:
                return True
            # x is first: 2a - b = x
            if 2 * a - b == x:
                return True
    # x is middle: a + b = 2x for a, b in S
    for a in selected_set:
        for b in selected_set:
            if a == b:
                continue
            if a + b == 2 * x:
                return True
    return False

def _conflicts_fast(x, selected_list):
    """Faster conflict check using the list representation."""
    n = len(selected_list)
    for i in range(n):
        a = selected_list[i]
        for j in range(n):
            if i == j:
                continue
            b = selected_list[j]
            # x as third term of AP (a, b, x): 2b - a = x
            if 2 * b - a == x:
                return True
            # x as first term of AP (x, a, b): 2a - b = x
            if 2 * a - b == x:
                return True
    # x as middle: a + b = 2x
    for i in range(n):
        a = selected_list[i]
        for j in range(n):
            if i == j:
                continue
            b = selected_list[j]
            if a + b == 2 * x:
                return True
    return False

def _is_valid_3ap_free(selected):
    """Verify the entire set has no 3-term AP."""
    s = set(selected)
    for a in s:
        for b in s:
            if a == b:
                continue
            c = 2 * b - a
            if c in s and c != a and c != b:
                return False
    return True

def _reciprocal_sum(selected):
    """Compute exact reciprocal sum as Fraction."""
    total = Fraction(0, 1)
    for x in selected:
        total += Fraction(1, x)
    return total

def _greedy_initial(n=256):
    """Greedy: add numbers in order of decreasing reciprocal (i.e., increasing value)
    if they don't create a 3-AP."""
    selected = []
    selected_set = set()
    for x in range(1, n + 1):
        if not _conflicts_with(x, selected_set):
            selected.append(x)
            selected_set.add(x)
    return selected

def _greedy_by_reciprocal(n=256):
    """Greedy: try to add numbers in order of 1/x (largest first = smallest x first).
    This is the same as increasing order since 1/x decreases with x."""
    # Actually, to maximize reciprocal sum, we want small numbers.
    # Greedy in increasing order of x.
    selected = []
    selected_set = set()
    for x in range(1, n + 1):
        if not _conflicts_with(x, selected_set):
            selected.append(x)
            selected_set.add(x)
    return selected

def _local_search(selected, n=256, max_iter=5000, deadline=None):
    """Local search: try adding, removing, or swapping elements to improve reciprocal sum."""
    selected_set = set(selected)
    current_sum = _reciprocal_sum(selected)
    best_selected = list(selected)
    best_sum = current_sum
    
    # Precompute which numbers are available (not in selected, in [1, n])
    all_numbers = set(range(1, n + 1))
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # Try adding a single element
        for x in range(1, n + 1):
            if x in selected_set:
                continue
            # Check if x can be added
            if not _conflicts_with(x, selected_set):
                new_sum = current_sum + Fraction(1, x)
                if new_sum > current_sum:
                    selected.append(x)
                    selected_set.add(x)
                    current_sum = new_sum
                    improved = True
                    break  # Restart after one improvement
        
        if improved:
            continue
        
        # Try removing one element and adding another (swap)
        improved = False
        for i, rem in enumerate(selected):
            # Try removing rem
            temp_set = set(selected)
            temp_set.discard(rem)
            temp_list = [x for x in selected if x != rem]
            
            for x in range(1, n + 1):
                if x in temp_set:
                    continue
                if not _conflicts_with(x, temp_set):
                    # Check if this swap improves
                    new_sum = current_sum - Fraction(1, rem) + Fraction(1, x)
                    if new_sum > current_sum:
                        # Perform swap
                        selected.remove(rem)
                        selected.append(x)
                        selected_set.discard(rem)
                        selected_set.add(x)
                        current_sum = new_sum
                        improved = True
                        break
            if improved:
                break
        
        if improved:
            continue
        
        # Try removing one element (if it frees up more valuable additions)
        improved = False
        for rem in selected:
            temp_set = set(selected)
            temp_set.discard(rem)
            temp_list = [x for x in selected if x != rem]
            
            # Try adding multiple elements after removal
            added = 0
            new_sum = current_sum - Fraction(1, rem)
            new_selected = list(temp_list)
            new_selected_set = set(temp_list)
            
            for x in range(1, n + 1):
                if x in new_selected_set:
                    continue
                if not _conflicts_with(x, new_selected_set):
                    new_selected.append(x)
                    new_selected_set.add(x)
                    new_sum += Fraction(1, x)
                    added += 1
            
            if new_sum > current_sum:
                selected = new_selected
                selected_set = new_selected_set
                current_sum = new_sum
                improved = True
                break
        
        if not improved:
            # No improvement possible with single moves
            break
        
        # Track best
        if current_sum > best_sum:
            best_sum = current_sum
            best_selected = list(selected)
    
    return best_selected

def _multi_start_greedy(n=256, deadline=None):
    """Try different greedy orderings and local search from each."""
    best_selected = None
    best_sum = Fraction(0, 1)
    
    # Standard greedy (increasing order)
    selected = _greedy_initial(n)
    selected = _local_search(selected, n, max_iter=3000, deadline=deadline)
    s = _reciprocal_sum(selected)
    if s > best_sum:
        best_sum = s
        best_selected = selected
    
    # Try greedy with some randomization
    import random
    random.seed(42)
    
    for trial in range(20):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb the greedy: skip some small numbers occasionally
        selected = []
        selected_set = set()
        for x in range(1, n + 1):
            # With small probability, skip x even if it fits
            if random.random() < 0.05 and x > 1:
                continue
            if not _conflicts_with(x, selected_set):
                selected.append(x)
                selected_set.add(x)
        
        selected = _local_search(selected, n, max_iter=2000, deadline=deadline)
        s = _reciprocal_sum(selected)
        if s > best_sum:
            best_sum = s
            best_selected = selected
    
    return best_selected

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s limit
    
    # Get initial solution via multi-start greedy + local search
    result = _multi_start_greedy(256, deadline)
    
    if result is None or len(result) == 0:
        # Fallback: simple greedy
        result = _greedy_initial(256)
    
    # Final verification
    if not _is_valid_3ap_free(result):
        # Fix by removing conflicting elements
        result_set = set(result)
        changed = True
        while changed:
            changed = False
            for a in list(result_set):
                for b in list(result_set):
                    if a == b:
                        continue
                    c = 2 * b - a
                    if c in result_set and c != a and c != b:
                        # Remove the largest to preserve small numbers
                        to_remove = max(a, b, c)
                        result_set.discard(to_remove)
                        changed = True
        result = list(result_set)
    
    return result
# EVOLVE-BLOCK-END
