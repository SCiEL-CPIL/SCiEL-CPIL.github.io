import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s margin before 150s kill
    
    # Prescribed elements
    prescribed = [1, 2, 5, 11]
    
    # Initial sums from prescribed elements
    def compute_sums(s):
        sums = set()
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                sums.add(s[i] + s[j])
        return sums
    
    initial_sums = compute_sums(prescribed)
    best_set = prescribed[:]
    best_size = len(prescribed)
    
    def is_valid_addition(x, current_set, used_sums):
        # Check if x can be added to current_set
        for y in current_set:
            if x + y in used_sums:
                return False
        if 2 * x in used_sums:
            return False
        return True
    
    def get_new_sums(x, current_set):
        new_sums = set()
        for y in current_set:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        return new_sums
    
    # Strategy 1: Greedy construction with multiple random orderings
    # Strategy 2: Backtracking search
    
    # First, try a greedy approach with different orderings
    def greedy_construct(order):
        current_set = prescribed[:]
        used_sums = initial_sums.copy()
        
        for x in order:
            if time.time() > deadline:
                break
            if x in current_set:
                continue
            if is_valid_addition(x, current_set, used_sums):
                current_set.append(x)
                new_sums = get_new_sums(x, current_set)
                used_sums |= new_sums
        
        return current_set
    
    # Try greedy with different random orderings
    candidates = list(range(12, 285))
    
    # First, filter candidates that are immediately invalid
    valid_candidates = []
    for x in candidates:
        if is_valid_addition(x, prescribed, initial_sums):
            valid_candidates.append(x)
    
    # Try several random orderings
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        random.shuffle(valid_candidates)
        result = greedy_construct(valid_candidates)
        
        if len(result) > best_size:
            best_set = result[:]
            best_size = len(result)
    
    # Now try backtracking search to find larger sets
    # Use a more efficient backtracking with better pruning
    
    def backtrack_search(current_set, used_sums, candidates, depth):
        nonlocal best_set, best_size, deadline
        
        if time.time() > deadline:
            return False
        
        if len(current_set) > best_size:
            best_set = current_set[:]
            best_size = len(current_set)
            
        if len(current_set) == 20:
            return True
            
        if not candidates:
            return False
            
        # Pruning: if remaining candidates < needed elements
        needed = 20 - len(current_set)
        if len(candidates) < needed:
            return False
        
        # Try to add each candidate
        for i, x in enumerate(candidates):
            if time.time() > deadline:
                return False
                
            if not is_valid_addition(x, current_set, used_sums):
                continue
                
            new_sums = get_new_sums(x, current_set)
            new_set = current_set + [x]
            new_used_sums = used_sums | new_sums
            
            # Filter candidates for next step
            next_candidates = []
            for y in candidates[i+1:]:
                if is_valid_addition(y, new_set, new_used_sums):
                    next_candidates.append(y)
                    
            if backtrack_search(new_set, new_used_sums, next_candidates, depth + 1):
                return True
                
        return False
    
    # Run backtracking from the initial set
    # But first, let's try to build up to a larger set using greedy, then backtrack from there
    
    # Try multiple starting points for backtracking
    for attempt in range(10):
        if time.time() > deadline:
            break
        
        random.shuffle(valid_candidates)
        # Build a partial set using greedy
        partial_set = prescribed[:]
        used_sums = initial_sums.copy()
        
        # Add some elements greedily
        added = 0
        for x in valid_candidates:
            if added >= 8:  # Build up to about 12 elements total
                break
            if is_valid_addition(x, partial_set, used_sums):
                partial_set.append(x)
                used_sums |= get_new_sums(x, partial_set)
                added += 1
        
        # Now backtrack from this partial set
        remaining_candidates = [x for x in valid_candidates if x not in partial_set]
        # Sort remaining candidates
        remaining_candidates.sort()
        
        # Filter remaining candidates that are valid with partial_set
        filtered_remaining = []
        for x in remaining_candidates:
            if is_valid_addition(x, partial_set, used_sums):
                filtered_remaining.append(x)
        
        if filtered_remaining:
            backtrack_search(partial_set, used_sums, filtered_remaining, 0)
    
    return best_set
