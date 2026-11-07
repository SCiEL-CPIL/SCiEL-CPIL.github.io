"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _check_divisibility_conflict(new_sums, existing_sums):
    """Check if any new sum conflicts with any existing sum via divisibility.
    new_sums: list of new sums to add
    existing_sums: set of existing sums
    Returns True if conflict found."""
    for ns in new_sums:
        for es in existing_sums:
            if ns == es:
                return True
            if ns % es == 0 or es % ns == 0:
                return True
    # Also check new sums against each other
    for i in range(len(new_sums)):
        for j in range(i + 1, len(new_sums)):
            a, b = new_sums[i], new_sums[j]
            if a == b:
                return True
            if a % b == 0 or b % a == 0:
                return True
    return False

def _backtrack_search(arr, candidate_list, sums_set, max_size, deadline):
    """Backtracking search to find a valid set of size max_size.
    arr: current set (list)
    candidate_list: list of candidates to try (sorted)
    sums_set: set of all current subset sums
    Returns (valid_set, new_sums_set) or (None, None)"""
    if time.time() > deadline:
        return None, None
    
    if len(arr) >= max_size:
        return arr[:], sums_set.copy()
    
    for idx, cand in enumerate(candidate_list):
        if time.time() > deadline:
            return None, None
        
        # Compute new subset sums: cand alone, and cand + each existing sum
        new_sums = [cand]
        for s in sums_set:
            new_sums.append(s + cand)
        
        # Check for conflicts
        all_sums = sums_set | set(new_sums)
        if len(all_sums) != len(sums_set) + len(new_sums):
            # Duplicate sums - invalid
            continue
        
        # Check divisibility
        conflict = False
        for ns in new_sums:
            for es in sums_set:
                if ns % es == 0 or es % ns == 0:
                    conflict = True
                    break
            if conflict:
                break
        if conflict:
            continue
        # Check new sums against each other
        for i in range(len(new_sums)):
            for j in range(i + 1, len(new_sums)):
                a, b = new_sums[i], new_sums[j]
                if a == b or a % b == 0 or b % a == 0:
                    conflict = True
                    break
            if conflict:
                break
        if conflict:
            continue
        
        # Valid addition - recurse
        new_arr = arr + [cand]
        new_sums_set = sums_set | set(new_sums)
        remaining_candidates = candidate_list[idx + 1:]
        
        result, result_sums = _backtrack_search(new_arr, remaining_candidates, new_sums_set, max_size, deadline)
        if result is not None:
            return result, result_sums
    
    return None, None

def _greedy_with_backtrack(candidates, target_size, deadline):
    """Try to build a set of target_size using greedy + limited backtracking."""
    candidates_sorted = sorted(candidates, reverse=True)  # Try larger first
    arr = []
    sums_set = set()
    
    for cand in candidates_sorted:
        if time.time() > deadline:
            break
        if len(arr) >= target_size:
            break
        
        new_sums = [cand] + [s + cand for s in sums_set]
        
        # Check duplicates
        if len(set(new_sums) | sums_set) != len(sums_set) + len(new_sums):
            continue
        
        # Check divisibility
        conflict = False
        for ns in new_sums:
            for es in sums_set:
                if ns % es == 0 or es % ns == 0:
                    conflict = True
                    break
            if conflict:
                break
        if conflict:
            continue
        for i in range(len(new_sums)):
            for j in range(i + 1, len(new_sums)):
                a, b = new_sums[i], new_sums[j]
                if a == b or a % b == 0 or b % a == 0:
                    conflict = True
                    break
            if conflict:
                break
        if conflict:
            continue
        
        arr.append(cand)
        sums_set.update(new_sums)
    
    return arr, sums_set

def _local_search_improve(arr, sums_set, deadline, max_iterations=1000):
    """Try to improve the set by swapping elements to enable growth."""
    current_arr = arr[:]
    current_sums = sums_set.copy()
    current_size = len(current_arr)
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Try to add a random element
        added = False
        for _ in range(20):
            if time.time() > deadline:
                break
            cand = random.randint(1, 256)
            if cand in current_arr:
                continue
            
            new_sums = [cand] + [s + cand for s in current_sums]
            
            if len(set(new_sums) | current_sums) != len(current_sums) + len(new_sums):
                continue
            
            conflict = False
            for ns in new_sums:
                for es in current_sums:
                    if ns % es == 0 or es % ns == 0:
                        conflict = True
                        break
                if conflict:
                    break
            if conflict:
                continue
            for i in range(len(new_sums)):
                for j in range(i + 1, len(new_sums)):
                    a, b = new_sums[i], new_sums[j]
                    if a == b or a % b == 0 or b % a == 0:
                        conflict = True
                        break
                if conflict:
                    break
            if conflict:
                continue
            
            current_arr.append(cand)
            current_sums.update(new_sums)
            current_size = len(current_arr)
            added = True
            break
        
        if added:
            continue
        
        # Try swapping one element
        if len(current_arr) > 0:
            i = random.randint(0, len(current_arr) - 1)
            old_elem = current_arr[i]
            
            for _ in range(10):
                if time.time() > deadline:
                    break
                new_elem = random.randint(1, 256)
                if new_elem in current_arr:
                    continue
                
                # Remove old_elem, add new_elem
                trial_arr = current_arr[:i] + current_arr[i+1:] + [new_elem]
                
                # Recompute sums from scratch (expensive but correct)
                trial_sums = set()
                n = len(trial_arr)
                for mask in range(1, 1 << n):
                    s = 0
                    for k in range(n):
                        if mask & (1 << k):
                            s += trial_arr[k]
                    trial_sums.add(s)
                
                # Check uniqueness (already guaranteed by set)
                # Check divisibility
                sums_list = sorted(trial_sums)
                conflict = False
                for a in range(len(sums_list)):
                    for b in range(a + 1, len(sums_list)):
                        if sums_list[b] % sums_list[a] == 0:
                            conflict = True
                            break
                    if conflict:
                        break
                
                if not conflict:
                    current_arr = trial_arr
                    current_sums = trial_sums
                    break
    
    return current_arr, current_sums, current_size

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10 second safety margin
    
    best_arr = [31, 32, 34]
    best_size = 3
    
    random.seed(42)
    
    # Phase 1: Greedy construction with different orderings
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        arr, sums_set = _greedy_with_backtrack(candidates, 15, deadline)
        
        if len(arr) > best_size:
            best_arr = arr[:]
            best_size = len(arr)
            if best_size >= 15:
                break
    
    # Phase 2: Local search to improve
    if best_size < 15 and time.time() < deadline:
        # Recompute sums for best_arr
        best_sums = set()
        n = len(best_arr)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += best_arr[i]
            best_sums.add(s)
        
        improved_arr, improved_sums, improved_size = _local_search_improve(
            best_arr, best_sums, deadline, max_iterations=500
        )
        
        if improved_size > best_size:
            best_arr = improved_arr
            best_size = improved_size
    
    # Phase 3: Try backtracking from scratch with target sizes
    for target in range(best_size + 1, min(best_size + 4, 16)):
        if time.time() > deadline:
            break
        
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        # Sort by some heuristic - maybe by "rarity" of divisibility relations
        candidates.sort(reverse=True)
        
        result, result_sums = _backtrack_search([], candidates, set(), target, deadline)
        if result is not None and len(result) > best_size:
            best_arr = result
            best_size = len(result)
    
    # Final validation
    if best_size >= 4:
        n = len(best_arr)
        sum_count = {}
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += best_arr[i]
            if s in sum_count:
                # Duplicate sum - invalid
                best_arr = [31, 32, 34]
                best_size = 3
                break
            sum_count[s] = 1
        else:
            sums_list = sorted(sum_count.keys())
            valid = True
            for i in range(len(sums_list)):
                for j in range(i + 1, len(sums_list)):
                    if sums_list[j] % sums_list[i] == 0:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                best_arr = [31, 32, 34]
                best_size = 3
    
    # Ensure proper types
    result = []
    for x in best_arr:
        if isinstance(x, int) and not isinstance(x, bool):
            result.append(x)
    
    return list(result) if result else [31, 32, 34]
# EVOLVE-BLOCK-END
