import time
import itertools
import random
from functools import lru_cache

# Precompute all triple sums for each pair (i,j) with i<j, store sums
# For a set S, we need:
# 1. x does not divide any triple sum from S
# 2. For each a in S, a does not divide x + u + v for any u,v in S\{a}

def _can_add(x, selected, triple_sums=None):
    """Check if x can be added to selected without violating the constraint.
    selected: list of selected integers
    triple_sums: optional precomputed dict mapping (i,j) -> list of sums with k
    """
    n = len(selected)
    
    # Check 1: x must not divide sum of any 3 distinct elements in selected
    if n >= 3:
        for i in range(n):
            for j in range(i+1, n):
                s_ij = selected[i] + selected[j]
                for k in range(j+1, n):
                    if (s_ij + selected[k]) % x == 0:
                        return False
    
    # Check 2: For each a in selected, a must not divide x + u + v for any u, v in selected \ {a}
    for a_idx, a in enumerate(selected):
        others = [selected[i] for i in range(n) if i != a_idx]
        m = len(others)
        for i in range(m):
            for j in range(i+1, m):
                if (x + others[i] + others[j]) % a == 0:
                    return False
    return True

def _can_add_fast(x, selected, precomputed):
    """Faster check using precomputed data structures."""
    n = len(selected)
    
    # Check 1: x must not divide sum of any 3 distinct elements in selected
    if n >= 3:
        # Use the precomputed triple sums
        triple_sums = precomputed['triple_sums']
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if triple_sums[(i,j,k)] % x == 0:
                        return False
    
    # Check 2: For each a in selected, a must not divide x + u + v for any u, v in selected \ {a}
    for a_idx, a in enumerate(selected):
        others = [selected[i] for i in range(n) if i != a_idx]
        m = len(others)
        for i in range(m):
            for j in range(i+1, m):
                if (x + others[i] + others[j]) % a == 0:
                    return False
    return True

def _greedy(order, deadline):
    """Run greedy in the given order."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _greedy_with_lookahead(order, deadline, look_ahead=1):
    """Greedy with limited lookahead: when adding x, also check that we don't block too many future additions."""
    selected = []
    remaining = list(order)
    
    while remaining:
        if time.time() > deadline:
            break
        
        best_x = None
        best_score = -1
        
        for idx, x in enumerate(remaining):
            if not _can_add(x, selected):
                continue
            
            # Score: prefer larger x, but also consider how many elements it blocks
            # Quick heuristic: count how many remaining elements would be blocked
            if look_ahead > 0:
                blocked = 0
                for y in remaining:
                    if y == x:
                        continue
                    if not _can_add(y, selected + [x]):
                        blocked += 1
                score = x - blocked * 0.5
            else:
                score = x
            
            if score > best_score:
                best_score = score
                best_x = x
                best_idx = idx
        
        if best_x is None:
            break
        
        selected.append(best_x)
        remaining.remove(best_x)
    
    return selected

def _greedy_with_priority(selected, deadline, max_iterations=200):
    """Greedy that at each step picks the element that is 'safest' to add."""
    in_set = set(selected)
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        best_x = None
        best_score = -1
        # Score each candidate by how few conflicts it would create
        for x in range(192, 0, -1):
            if x in in_set:
                continue
            if not _can_add(x, selected):
                continue
            # Heuristic: prefer larger numbers (they're less likely to divide sums)
            score = x
            if score > best_score:
                best_score = score
                best_x = x
        if best_x is None:
            break
        selected.append(best_x)
        in_set.add(best_x)
    return selected

def _compute_conflict_count(x, selected):
    """Count how many potential additions would be blocked by adding x."""
    blocked = 0
    for y in range(1, 193):
        if y in selected or y == x:
            continue
        if not _can_add(y, selected + [x]):
            blocked += 1
    return blocked

def _local_search(selected, deadline):
    """Try to improve the set by swapping or adding elements with better heuristics."""
    selected = list(selected)
    improved = True
    no_improve_count = 0
    max_no_improve = 5
    
    while improved and time.time() < deadline - 2:
        improved = False
        no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            break
            
        # Try adding any element not in selected
        in_set = set(selected)
        for x in range(192, 0, -1):
            if time.time() > deadline:
                return selected
            if x not in in_set and _can_add(x, selected):
                selected.append(x)
                in_set.add(x)
                improved = True
                no_improve_count = 0
                break
        
        # Try single swap: remove one, add one
        if not improved:
            best_swap_gain = 0
            best_swap_new_set = None
            best_swap_remove_idx = -1
            best_swap_add_x = -1
            
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                candidate = selected[i]
                selected_without = selected[:i] + selected[i+1:]
                
                for x in range(192, 0, -1):
                    if x in in_set:
                        continue
                    if not _can_add(x, selected_without):
                        continue
                    
                    new_set = selected_without + [x]
                    new_in_set = set(new_set)
                    
                    # Count how many additional elements can be added
                    extra_count = 0
                    for y in range(192, 0, -1):
                        if time.time() > deadline:
                            break
                        if y not in new_in_set and _can_add(y, new_set):
                            extra_count += 1
                            if extra_count >= 3:
                                break
                    
                    # Also try to add immediately after swap
                    immediate_add = 0
                    for y in range(192, 0, -1):
                        if y not in new_in_set and _can_add(y, new_set):
                            immediate_add += 1
                            break
                    
                    gain = extra_count + immediate_add
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_new_set = new_set
                        best_swap_remove_idx = i
                        best_swap_add_x = x
            
            if best_swap_new_set is not None and best_swap_gain > 0:
                selected = best_swap_new_set
                in_set = set(selected)
                improved = True
                no_improve_count = 0
        
        # Try double swap: remove two, add two
        if not improved and len(selected) >= 2:
            best_double_swap_gain = 0
            best_double_swap_new_set = None
            
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(selected)):
                    if time.time() > deadline:
                        break
                    selected_without = selected[:i] + selected[i+1:j] + selected[j+1:]
                    
                    for x in range(192, 0, -1):
                        if x in in_set:
                            continue
                        if not _can_add(x, selected_without):
                            continue
                        set_with_x = selected_without + [x]
                        
                        for y in range(192, 0, -1):
                            if y == x or y in in_set:
                                continue
                            if not _can_add(y, set_with_x):
                                continue
                            
                            new_set = set_with_x + [y]
                            new_in_set = set(new_set)
                            
                            extra_count = 0
                            for z in range(192, 0, -1):
                                if time.time() > deadline:
                                    break
                                if z not in new_in_set and _can_add(z, new_set):
                                    extra_count += 1
                                    if extra_count >= 2:
                                        break
                            
                            gain = extra_count
                            if gain > best_double_swap_gain:
                                best_double_swap_gain = gain
                                best_double_swap_new_set = new_set
                    
                    if best_double_swap_gain >= 2:
                        break
                if best_double_swap_gain >= 2:
                    break
            
            if best_double_swap_new_set is not None and best_double_swap_gain > 0:
                selected = best_double_swap_new_set
                in_set = set(selected)
                improved = True
                no_improve_count = 0
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin (90s limit, leave 35s buffer)
    
    best = []
    best_size = 0
    
    # Strategy 1: Greedy descending
    result = _greedy(range(192, 0, -1), deadline)
    if len(result) > best_size:
        best = result
        best_size = len(result)
    
    # Strategy 2: Greedy ascending
    if time.time() < deadline - 10:
        result = _greedy(range(1, 193), deadline)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Strategy 3: Random shuffles with more iterations
    random.seed(42)
    num_random = 30
    for i in range(num_random):
        if time.time() > deadline - 15:
            break
        order = list(range(1, 193))
        random.shuffle(order)
        result = _greedy(order, deadline)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Strategy 4: Greedy with lookahead from different starting points
    if time.time() < deadline - 15:
        for start_size in range(0, 6):
            if time.time() > deadline - 15:
                break
            initial = list(range(192, 192 - start_size, -1))
            result = _greedy_with_priority(initial, deadline)
            if len(result) > best_size:
                best = result
                best_size = len(result)
    
    # Strategy 5: Multiple local search passes on top candidates
    if time.time() < deadline - 20:
        best = _local_search(best, deadline)
        best_size = len(best)
    
    # Strategy 6: Try to improve with more aggressive local search
    if time.time() < deadline - 5:
        # Create multiple variants and try local search on each
        variants = [best]
        
        # Variant: remove smallest element and re-greedy
        if len(best) > 1:
            variant = best[1:]  # Remove smallest (first element if sorted)
            if _can_add(best[0], variant):
                pass  # Already valid
            else:
                result = _greedy(range(192, 0, -1), deadline)
                if len(result) > best_size:
                    variants.append(result)
        
        # Variant: remove largest element and re-greedy
        if len(best) > 1:
            variant = best[:-1]  # Remove largest
            result = _greedy(range(192, 0, -1), deadline)
            if len(result) > best_size:
                variants.append(result)
        
        # Run local search on each variant
        for v in variants:
            if time.time() > deadline - 3:
                break
            v_result = _local_search(v, deadline)
            if len(v_result) > best_size:
                best = v_result
                best_size = len(best)
    
    # Final polish: try to add any remaining elements
    if time.time() < deadline - 1:
        in_set = set(best)
        for x in range(192, 0, -1):
            if time.time() > deadline:
                break
            if x not in in_set and _can_add(x, best):
                best.append(x)
                in_set.add(x)
    
    return best
