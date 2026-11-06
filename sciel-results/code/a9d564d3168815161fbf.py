import time
import random

def _has_4ap(arr):
    """Check if sorted array arr contains a nonconstant 4-term AP."""
    s = set(arr)
    n = len(arr)
    for i in range(n):
        a = arr[i]
        for j in range(i + 1, n):
            d = arr[j] - a
            if d <= 0:
                continue
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if b in s and c in s and e in s:
                return True
    return False

def _can_add(x, current):
    """Check if adding x to current set creates a 4-AP."""
    # x is 4th term: x-3d, x-2d, x-d must be in current
    # x is 3rd term: x-2d, x-d, x+d must be in current
    # x is 2nd term: x-d, x+d, x+2d must be in current
    # x is 1st term: x+d, x+2d, x+3d must be in current
    max_d = (x - 1) // 3 + 1 if x > 1 else 0
    for d in range(1, max_d + 1):
        if (x - 3 * d) in current and (x - 2 * d) in current and (x - d) in current:
            return False
        if (x - 2 * d) in current and (x - d) in current and (x + d) in current:
            return False
        if (x - d) in current and (x + d) in current and (x + 2 * d) in current:
            return False
        if (x + d) in current and (x + 2 * d) in current and (x + 3 * d) in current:
            return False
    return True

def _ternary_01_set(n=512):
    """Numbers in [1, n] with base-3 digits only 0 and 1."""
    result = []
    for x in range(1, n + 1):
        temp = x
        valid = True
        while temp > 0:
            if temp % 3 == 2:
                valid = False
                break
            temp //= 3
        if valid:
            result.append(x)
    return result

def _greedy_construct(n=512):
    """Greedy construction: add numbers in order if they don't create a 4-AP."""
    selected = set()
    for x in range(1, n + 1):
        if _can_add(x, selected):
            selected.add(x)
    return sorted(selected)

def _local_search(arr, deadline, max_rounds=500):
    """Local search with swaps to improve 4-AP-free set."""
    current = set(arr)
    current_list = sorted(current)
    n = 512
    all_elements = set(range(1, n + 1))
    not_in_set = all_elements - current
    
    best_size = len(current_list)
    best_set = set(current)
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        # Try swaps: remove one element, add another
        for a in list(current_list):
            if time.time() > deadline:
                break
            
            # Try removing a and adding each candidate
            trial_set = current - {a}
            
            # Try adding candidates from not_in_set
            for b in sorted(not_in_set):
                if b == a:
                    continue
                if _can_add(b, trial_set):
                    # Check if this swap actually improves (it should since we're adding)
                    # But we need to verify no 4-AP in the full set
                    new_set = trial_set | {b}
                    if len(new_set) > best_size:
                        # Verify
                        new_list = sorted(new_set)
                        if not _has_4ap(new_list):
                            current = new_set
                            current_list = new_list
                            not_in_set = all_elements - current
                            best_size = len(current_list)
                            best_set = set(current)
                            improved = True
                            break
        if not improved:
            # Try single additions
            for b in sorted(not_in_set):
                if time.time() > deadline:
                    break
                if _can_add(b, current):
                    current.add(b)
                    current_list = sorted(current)
                    not_in_set = all_elements - current
                    best_size = len(current_list)
                    best_set = set(current)
                    improved = True
            if not improved:
                break
    
    return sorted(best_set)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s limit
    
    # Phase 1: Base-3 construction (guaranteed 4-AP-free)
    base3_set = _ternary_01_set(512)
    base3_set_sorted = sorted(base3_set)
    
    # Verify
    if _has_4ap(base3_set_sorted):
        # Fallback to greedy if something went wrong
        base3_set_sorted = _greedy_construct(512)
    
    best_result = base3_set_sorted
    best_size = len(best_result)
    
    # Phase 2: Greedy improvement from base-3 set
    if time.time() < deadline - 5:
        current = set(base3_set_sorted)
        for x in range(1, 513):
            if time.time() > deadline - 5:
                break
            if x in current:
                continue
            if _can_add(x, current):
                current.add(x)
        
        greedy_result = sorted(current)
        if not _has_4ap(greedy_result) and len(greedy_result) > best_size:
            best_result = greedy_result
            best_size = len(greedy_result)
    
    # Phase 3: Local search with swaps
    if time.time() < deadline - 2:
        ls_result = _local_search(best_result, deadline)
        if not _has_4ap(ls_result) and len(ls_result) > best_size:
            best_result = ls_result
            best_size = len(ls_result)
    
    # Final verification
    if _has_4ap(best_result):
        # Fallback to base-3 set
        best_result = base3_set_sorted
    
    return best_result
