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
    """Check if adding x to selected_set creates a 3-term AP. O(|S|) using set lookups."""
    for a in selected_set:
        # x as third term: (a, b, x) => 2b - a = x => b = (a + x) / 2
        # x as first term: (x, a, b) => 2a - b = x => b = 2a - x
        # x as middle: (a, x, b) => a + b = 2x => b = 2x - a
        b1 = (a + x) // 2
        if (a + x) % 2 == 0 and b1 in selected_set and b1 != a and b1 != x:
            return True
        b2 = 2 * a - x
        if b2 in selected_set and b2 != a and b2 != x:
            return True
        b3 = 2 * x - a
        if b3 in selected_set and b3 != a and b3 != x:
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
    """Local search: try adding, removing, or swapping elements to improve reciprocal sum.
    Uses float arithmetic for speed; exact Fraction only at end."""
    selected_set = set(selected)
    current_sum = sum(1.0 / x for x in selected)
    best_selected = list(selected)
    best_sum = current_sum
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # Try adding a single element (smallest x first for max gain)
        for x in range(1, n + 1):
            if x in selected_set:
                continue
            if not _conflicts_with(x, selected_set):
                selected.append(x)
                selected_set.add(x)
                current_sum += 1.0 / x
                improved = True
                break
        
        if improved:
            if current_sum > best_sum:
                best_sum = current_sum
                best_selected = list(selected)
            continue
        
        # Try swap: remove one, add one
        improved = False
        for rem in selected:
            temp_set = selected_set.copy()
            temp_set.discard(rem)
            best_gain = 0.0
            best_x = None
            for x in range(1, n + 1):
                if x in temp_set:
                    continue
                if not _conflicts_with(x, temp_set):
                    gain = 1.0 / x - 1.0 / rem
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
            if best_x is not None:
                selected.remove(rem)
                selected.append(best_x)
                selected_set.discard(rem)
                selected_set.add(best_x)
                current_sum += best_gain
                improved = True
                break
        
        if improved:
            if current_sum > best_sum:
                best_sum = current_sum
                best_selected = list(selected)
            continue
        
        # Try remove-one + greedy-fill: remove an element, then greedily add as many as possible
        improved = False
        for rem in selected:
            temp_set = selected_set.copy()
            temp_set.discard(rem)
            new_selected = [x for x in selected if x != rem]
            new_set = set(new_selected)
            new_sum = current_sum - 1.0 / rem
            
            # Greedily add smallest available numbers that don't conflict
            for x in range(1, n + 1):
                if x in new_set:
                    continue
                if not _conflicts_with(x, new_set):
                    new_selected.append(x)
                    new_set.add(x)
                    new_sum += 1.0 / x
            
            if new_sum > current_sum + 1e-12:
                selected = new_selected
                selected_set = new_set
                current_sum = new_sum
                improved = True
                break
        
        if not improved:
            break
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_selected = list(selected)
    
    return best_selected

def _multi_start_greedy(n=256, deadline=None):
    """Try different greedy orderings and local search from each.
    Uses float comparison for speed."""
    best_selected = None
    best_sum = 0.0
    
    # Standard greedy (increasing order)
    selected = _greedy_initial(n)
    selected = _local_search(selected, n, max_iter=8000, deadline=deadline)
    s = sum(1.0 / x for x in selected)
    if s > best_sum:
        best_sum = s
        best_selected = selected
    
    # Try greedy with some randomization
    import random
    random.seed(42)
    
    for trial in range(40):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb the greedy: skip some small numbers occasionally
        selected = []
        selected_set = set()
        for x in range(1, n + 1):
            if random.random() < 0.05 and x > 1:
                continue
            if not _conflicts_with(x, selected_set):
                selected.append(x)
                selected_set.add(x)
        
        selected = _local_search(selected, n, max_iter=5000, deadline=deadline)
        s = sum(1.0 / x for x in selected)
        if s > best_sum:
            best_sum = s
            best_selected = selected
    
    return best_selected

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
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
