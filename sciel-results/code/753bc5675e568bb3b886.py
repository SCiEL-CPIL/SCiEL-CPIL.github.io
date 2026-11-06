"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit
    
    # Prescribed elements
    prescribed = [1, 2, 5, 11]
    N = 284
    prescribed_set = set(prescribed)
    
    def init_sums(s):
        sums = set()
        for i, x in enumerate(s):
            for y in s[i:]:
                sums.add(x + y)
        return sums
    
    def can_add(s, sums, x):
        """Check if x can be added to Sidon set s with existing sums."""
        for y in s:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True
    
    def add_element(s, sums, x):
        """Add x to s and update sums. Assumes can_add is true."""
        s.append(x)
        for y in s[:-1]:
            sums.add(x + y)
        sums.add(2 * x)
    
    # Strategy 1: Fast greedy baseline (from inspiration 1)
    s1 = list(prescribed)
    sums1 = init_sums(s1)
    for x in range(12, N + 1):
        new = {x + y for y in s1} | {2 * x}
        if len(new) == len(s1) + 1 and not new & sums1:
            s1.append(x)
            sums1 |= new
    
    best_set = s1[:]
    best_size = len(best_set)
    
    if best_size == 20:
        return best_set
    
    # Strategy 2: Randomized greedy restarts (from current program)
    random.seed(42)
    
    for restart in range(200):
        if time.time() > DEADLINE:
            break
        
        candidates = list(range(12, N + 1))
        random.shuffle(candidates)
        
        s, sums = list(prescribed), init_sums(prescribed)
        for x in candidates:
            if can_add(s, sums, x):
                add_element(s, sums, x)
        
        if len(s) > best_size:
            best_set = s[:]
            best_size = len(best_set)
            if best_size == 20:
                return best_set
    
    if best_size == 20:
        return best_set
    
    # Strategy 3: Structured backtracking with constraint propagation
    # We need to find 16 more elements from [12, 284]
    # Use MRV (Minimum Remaining Values) heuristic
    
    all_candidates = list(range(12, N + 1))
    target_size = 20
    needed = target_size - len(prescribed)
    
    def backtrack_with_pruning(current_set, current_sums, remaining_indices, depth):
        """
        Backtracking search with pruning.
        current_set: list of elements so far
        current_sums: set of pairwise sums
        remaining_indices: list of candidate indices to consider
        depth: how many more elements we need
        """
        if depth == 0:
            return current_set[:]
        
        if not remaining_indices:
            return None
        
        if time.time() > DEADLINE:
            return None
        
        # Pruning: if not enough candidates left
        if len(remaining_indices) < depth:
            return None
        
        # MRV heuristic: try candidates that have the most conflicts first
        # (they are more constrained, so fail faster if they don't work)
        scored = []
        for idx in remaining_indices:
            x = all_candidates[idx]
            if x in current_set:
                continue
            conflict_count = 0
            for y in current_set:
                if x + y in current_sums:
                    conflict_count += 1
            if 2 * x in current_sums:
                conflict_count += 1
            # We want to try low-conflict candidates first (they're more likely to succeed)
            scored.append((conflict_count, idx))
        
        scored.sort(key=lambda t: t[0])
        
        # Try candidates in order of least conflict
        tried = set()
        for conflict_count, idx in scored:
            if len(tried) >= min(10, len(remaining_indices)):
                break  # Limit branching factor for speed
            
            if idx in tried:
                continue
            tried.add(idx)
            
            x = all_candidates[idx]
            if x in current_set:
                continue
            
            if not can_add(current_set, current_sums, x):
                continue
            
            # Add x
            new_set = current_set + [x]
            new_sums = set(current_sums)
            for y in current_set:
                new_sums.add(x + y)
            new_sums.add(2 * x)
            
            # Recurse
            result = backtrack_with_pruning(new_set, new_sums, remaining_indices, depth - 1)
            if result:
                return result
        
        return None
    
    # Try backtracking from the current best set as a warm start
    # Remove some elements from best_set to create a smaller starting set
    for remove_count in range(0, min(4, best_size - len(prescribed))):
        if time.time() > DEADLINE:
            break
        
        # Remove random elements from best_set (non-prescribed)
        non_prescribed_in_best = [x for x in best_set if x not in prescribed_set]
        
        for trial in range(50):
            if time.time() > DEADLINE:
                break
            
            if remove_count == 0:
                start_set = best_set[:]
            else:
                to_remove = random.sample(non_prescribed_in_best, min(remove_count, len(non_prescribed_in_best)))
                start_set = [x for x in best_set if x not in to_remove]
            
            start_sums = init_sums(start_set)
            needed = target_size - len(start_set)
            
            if needed <= 0:
                if len(start_set) == target_size:
                    return start_set
                continue
            
            # Get remaining candidates (not in start_set)
            start_set_set = set(start_set)
            remaining_candidates = [x for x in all_candidates if x not in start_set_set]
            
            result = backtrack_with_pruning(start_set, start_sums, list(range(len(remaining_candidates))), needed)
            
            if result and len(result) > best_size:
                best_set = result[:]
                best_size = len(best_set)
                if best_size == 20:
                    return best_set
    
    # Strategy 4: Local search as fallback
    def local_search(current_best, current_size):
        best_set = current_best[:]
        best_size = current_size
        s = best_set[:]
        sums = init_sums(s)
        
        all_candidates = [x for x in range(12, N + 1) if x not in prescribed_set]
        
        while time.time() < DEADLINE:
            k = random.choice([1, 2, 3])
            
            current_non_prescribed = [x for x in s if x not in prescribed_set]
            
            if len(current_non_prescribed) < k:
                break
            
            to_remove = random.sample(current_non_prescribed, k)
            
            s_trial = [x for x in s if x not in to_remove]
            sums_trial = init_sums(s_trial)
            
            available = [x for x in all_candidates if x not in s_trial]
            
            if len(available) < k:
                continue
                
            random.shuffle(available)
            
            added_count = 0
            for x in available:
                if added_count >= k:
                    break
                if can_add(s_trial, sums_trial, x):
                    add_element(s_trial, sums_trial, x)
                    added_count += 1
            
            if len(s_trial) > best_size:
                best_set = s_trial[:]
                best_size = len(best_set)
                s = s_trial[:]
                sums = sums_trial
                if best_size == 20:
                    return (best_set, best_size)
        
        return (best_set, best_size)
    
    best_set, best_size = local_search(best_set, best_size)
    
    return best_set

# EVOLVE-BLOCK-END
