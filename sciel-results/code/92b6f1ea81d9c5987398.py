"""B_3[1] set construction: maximize cardinality with unique unordered triple sums."""
import time
import random
from itertools import combinations_with_replacement

def _new_sums_involving(x, S):
    """Compute all triple sums that include x, given existing sorted set S."""
    n = len(S)
    sums = [3 * x]
    for s in S:
        sums.append(2 * x + s)
        sums.append(x + 2 * s)
    for i in range(n):
        for j in range(i + 1, n):
            sums.append(x + S[i] + S[j])
    return sums

def _check_collision(new_sums, existing_sums_set):
    """Check if any new sum collides with existing sums."""
    for s in new_sums:
        if s in existing_sums_set:
            return True
    return False

def _greedy_build(start=1, end=4096, step=1, deadline=None):
    """Greedy construction scanning from start to end with given step."""
    selected = []
    existing_sums_set = set()
    
    for x in range(start, end + 1, step):
        if deadline and time.time() > deadline:
            break
        new_sums = _new_sums_involving(x, selected)
        if not _check_collision(new_sums, existing_sums_set):
            selected.append(x)
            for s in new_sums:
                existing_sums_set.add(s)
    
    return selected, existing_sums_set

def _greedy_with_heuristic(start=1, end=4096, deadline=None):
    """Greedy with heuristic: try candidates that minimize future conflicts."""
    selected = []
    existing_sums_set = set()
    candidates = list(range(start, end + 1))
    
    while candidates:
        if deadline and time.time() > deadline:
            break
        
        random.shuffle(candidates)
        
        for x in candidates:
            new_sums = _new_sums_involving(x, selected)
            if not _check_collision(new_sums, existing_sums_set):
                selected.append(x)
                for s in new_sums:
                    existing_sums_set.add(s)
                candidates.remove(x)
                break
    
    return selected, existing_sums_set

def _local_search_improve(selected, existing_sums_set, time_limit=30):
    """Try to improve the set via local search: swap elements."""
    if time.time() > time_limit:
        return selected
    
    best = list(selected)
    best_sums = set(existing_sums_set)
    best_size = len(best)
    
    for _ in range(5000):
        if time.time() > time_limit:
            break
        
        if len(best) >= 80:
            break
        
        found = False
        for x in range(1, 4097):
            if x in best:
                continue
            new_sums = _new_sums_involving(x, best)
            if not _check_collision(new_sums, best_sums):
                best.append(x)
                for s in new_sums:
                    best_sums.add(s)
                best_size = len(best)
                found = True
                break
        
        if found:
            continue
        
        break
    
    return best

def _backtrack_search(selected, existing_sums_set, target_size, deadline):
    """Backtracking search to find a valid set of target_size elements."""
    if len(selected) >= target_size:
        return list(selected)
    if time.time() > deadline:
        return list(selected)
    
    n = len(selected)
    # Try to add elements, backtracking if stuck
    for x in range(1, 4097):
        if time.time() > deadline:
            break
        if x in selected:
            continue
        new_sums = _new_sums_involving(x, selected)
        if not _check_collision(new_sums, existing_sums_set):
            selected.append(x)
            new_set = set(existing_sums_set)
            for s in new_sums:
                new_set.add(s)
            
            result = _backtrack_search(selected, new_set, target_size, deadline)
            if len(result) >= target_size:
                return result
            
            selected.pop()
    
    return list(selected)

def _greedy_with_backtrack(start=1, end=4096, deadline=None, target_size=20):
    """Greedy with backtracking to escape local minima."""
    selected = []
    existing_sums_set = set()
    remaining = set(range(start, end + 1))
    
    while remaining and len(selected) < target_size:
        if deadline and time.time() > deadline:
            break
        
        # Try candidates in order, with backtracking
        found = False
        candidates = sorted(remaining)
        
        for x in candidates:
            if deadline and time.time() > deadline:
                break
            new_sums = _new_sums_involving(x, selected)
            if not _check_collision(new_sums, existing_sums_set):
                selected.append(x)
                for s in new_sums:
                    existing_sums_set.add(s)
                remaining.remove(x)
                found = True
                break
        
        if not found:
            # Backtrack: remove last element and try alternatives
            if selected:
                removed = selected.pop()
                # Remove sums involving removed
                n = len(selected)
                to_remove = set()
                to_remove.add(3 * removed)
                for s in selected:
                    to_remove.add(2 * removed + s)
                    to_remove.add(removed + 2 * s)
                for i in range(n):
                    for j in range(i + 1, n):
                        to_remove.add(removed + selected[i] + selected[j])
                existing_sums_set -= to_remove
                remaining.add(removed)
                # Mark that we've tried this position
                if x > end:
                    break
            else:
                break
    
    return selected, existing_sums_set

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    best_result = []
    best_size = 0
    
    # Strategy 1: Linear greedy from 1
    selected, sums_set = _greedy_build(1, 4096, 1, deadline)
    if len(selected) > best_size:
        best_size = len(selected)
        best_result = list(selected)
    
    if time.time() > deadline:
        return best_result
    
    # Strategy 2: Try different starting points and steps
    for start in [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]:
        if time.time() > deadline:
            break
        for step in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]:
            if time.time() > deadline:
                break
            selected, sums_set = _greedy_build(start, 4096, step, deadline)
            if len(selected) > best_size:
                best_size = len(selected)
                best_result = list(selected)
    
    if time.time() > deadline:
        return best_result
    
    # Strategy 3: Local search - try to add more elements
    current = list(best_result)
    current_set = set(current)
    
    current_sums = set()
    n_curr = len(current)
    for i in range(n_curr):
        for j in range(i, n_curr):
            for k in range(j, n_curr):
                current_sums.add(current[i] + current[j] + current[k])
    
    for x in range(1, 4097):
        if time.time() > deadline:
            break
        if x in current_set:
            continue
        
        new_sums = _new_sums_involving(x, current)
        
        has_collision = False
        for s in new_sums:
            if s in current_sums:
                has_collision = True
                break
        
        if not has_collision:
            current.append(x)
            current_set.add(x)
            for s in new_sums:
                current_sums.add(s)
            best_size = len(current)
    
    # Strategy 4: Backtracking search to find larger sets
    if len(current) < 80 and time.time() < deadline - 10:
        # Try backtracking from a smaller base
        for base_size in [max(1, len(current) - 3), max(1, len(current) - 2), len(current)]:
            if time.time() > deadline - 5:
                break
            base = current[:base_size]
            base_set = set(base)
            base_sums = set()
            n_base = len(base)
            for i in range(n_base):
                for j in range(i, n_base):
                    for k in range(j, n_base):
                        base_sums.add(base[i] + base[j] + base[k])
            
            result, _ = _greedy_with_backtrack(1, 4096, deadline - 5, min(80, len(current) + 5))
            if len(result) > best_size:
                best_size = len(result)
                current = list(result)
                current_set = set(current)
    
    # Strategy 5: Final local search
    if len(current) < 80 and time.time() < deadline - 3:
        current_sums = set()
        n_curr = len(current)
        for i in range(n_curr):
            for j in range(i, n_curr):
                for k in range(j, n_curr):
                    current_sums.add(current[i] + current[j] + current[k])
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for x in range(1, 4097):
                if time.time() > deadline - 2:
                    break
                if x in current_set:
                    continue
                
                new_sums = _new_sums_involving(x, current)
                
                has_collision = False
                for s in new_sums:
                    if s in current_sums:
                        has_collision = True
                        break
                
                if not has_collision:
                    current.append(x)
                    current_set.add(x)
                    for s in new_sums:
                        current_sums.add(s)
                    best_size = len(current)
                    improved = True
    
    best_result = current
    return best_result
