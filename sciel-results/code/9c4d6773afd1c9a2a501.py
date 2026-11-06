import time
import random
import sys

N = 256
MAX_SUM = 2 * N  # Maximum possible sum is 2*N

def _make_sum_array(sel):
    """Create a boolean array where sums[s] = True if s is a sum of two elements in sel."""
    sums = bytearray(MAX_SUM + 1)
    for i in range(len(sel)):
        a = sel[i]
        for j in range(i, len(sel)):
            s = a + sel[j]
            sums[s] = 1
    return sums

def _is_maximal_fast(sel, sums_arr):
    """
    Fast maximality check using bytearray for sums.
    Returns True if sel is inclusion-maximal (no missing element can be added).
    """
    sel_set = set(sel)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        
        # Condition 1: 2*x is already a sum
        if sums_arr[2 * x]:
            continue
        
        # Condition 2: x + a is a sum for some a in sel
        blocked = False
        for a in sel:
            if sums_arr[x + a]:
                blocked = True
                break
        
        if not blocked:
            return False
            
    return True

def _greedy_construct_fast(order):
    """Greedy construction with bytearray sum tracking."""
    selected = []
    sums_arr = bytearray(MAX_SUM + 1)
    for x in order:
        if sums_arr[2 * x]:
            continue
        can_add = True
        for a in selected:
            if sums_arr[x + a]:
                can_add = False
                break
        if can_add:
            selected.append(x)
            sums_arr[2 * x] = 1
            for a in selected:
                sums_arr[x + a] = 1
    return selected, sums_arr

def _local_search_reduce_fast(sel, sums_arr, deadline):
    """
    Try to remove elements from sel while keeping it maximal.
    Uses bytearray for fast sum recomputation.
    """
    current_sel = list(sel)
    current_sums = sums_arr
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        current_sel_set = set(current_sel)
        
        # Try removing each element
        for i in range(len(current_sel)):
            if time.time() > deadline:
                return current_sel
            
            x_to_remove = current_sel[i]
            new_sel = current_sel[:i] + current_sel[i+1:]
            
            # Recompute sums for the new set
            new_sums = _make_sum_array(new_sel)
            
            if _is_maximal_fast(new_sel, new_sums):
                current_sel = new_sel
                current_sums = new_sums
                improved = True
                break
    return current_sel

def _try_remove_multiple(sel, sums_arr, deadline):
    """
    Try to remove multiple elements at once for faster reduction.
    """
    current_sel = list(sel)
    current_sums = sums_arr
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        n = len(current_sel)
        
        # Try all pairs of removals
        for i in range(n):
            if time.time() > deadline:
                return current_sel
            for j in range(i + 1, n):
                new_sel = current_sel[:i] + current_sel[i+1:j] + current_sel[j+1:]
                new_sums = _make_sum_array(new_sel)
                if _is_maximal_fast(new_sel, new_sums):
                    current_sel = new_sel
                    current_sums = new_sums
                    improved = True
                    break
            if improved:
                break
    return current_sel

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin for 60s limit
    
    best_set = None
    best_size = float('inf')
    
    # 1. Generate initial candidates using different greedy strategies
    candidates = []
    
    # Ascending
    sel, sums = _greedy_construct_fast(list(range(1, N + 1)))
    if _is_maximal_fast(sel, sums):
        candidates.append((sel, sums))
        
    # Descending
    sel, sums = _greedy_construct_fast(list(range(N, 0, -1)))
    if _is_maximal_fast(sel, sums):
        candidates.append((sel, sums))
        
    # Randomized Greedy - can do more trials now due to speedup
    rng = random.Random(12345)
    num_initial_trials = 500
    for _ in range(num_initial_trials):
        if time.time() > deadline - 15:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        sel, sums = _greedy_construct_fast(order)
        if _is_maximal_fast(sel, sums):
            candidates.append((sel, sums))
            
    # 2. Local search on each candidate
    for sel, sums in candidates:
        if time.time() > deadline - 3:
            break
            
        # Update best if this initial set is already better
        if len(sel) < best_size:
            best_size = len(sel)
            best_set = list(sel)
            
        # Run local search to minimize size
        optimized_sel = _local_search_reduce_fast(sel, sums, deadline)
        
        if len(optimized_sel) < best_size:
            best_size = len(optimized_sel)
            best_set = optimized_sel
            
            # Try multi-removal on the improved set
            if time.time() < deadline - 5:
                further_optimized = _try_remove_multiple(optimized_sel, _make_sum_array(optimized_sel), deadline)
                if len(further_optimized) < best_size:
                    best_size = len(further_optimized)
                    best_set = further_optimized
            
    # 3. If no valid set found (should not happen), fallback
    if best_set is None:
        sel, sums = _greedy_construct_fast(list(range(1, N + 1)))
        if _is_maximal_fast(sel, sums):
            best_set = sel
        else:
            # Force maximality by adding elements
            sel_set = set(sel)
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for x in range(1, N + 1):
                    if x in sel_set:
                        continue
                    if not sums_arr[2*x] if 'sums_arr' in dir() else True:
                        pass
                    ok = True
                    if sums_arr[2*x]:
                        ok = False
                    else:
                        for a in sel:
                            if sums_arr[x+a]:
                                ok = False
                                break
                    if ok:
                        sel.append(x)
                        sel_set.add(x)
                        sums_arr[2*x] = 1
                        for a in sel:
                            sums_arr[x+a] = 1
                        changed = True
                        break
            best_set = sel
            
    return best_set
