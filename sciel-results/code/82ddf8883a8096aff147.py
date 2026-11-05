import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    required = {1, 2, 5, 11}
    max_val = 284
    
    best_size = len(required)
    best_set = list(required)
    
    # Precompute: for each candidate x, what sums does it create with required elements?
    # We'll use a different approach: randomized greedy with multiple restarts
    
    # Strategy: Try many random orderings of candidates, greedily add if possible
    # Then do local search (swap) to improve
    
    candidates = list(range(1, max_val + 1))
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Randomize the order of non-required candidates
        non_required = [x for x in candidates if x not in required]
        random.shuffle(non_required)
        
        # Start with required elements
        current = list(required)
        current_sums = set()
        for i in range(len(current)):
            for j in range(i, len(current)):
                current_sums.add(current[i] + current[j])
        
        # Greedily add elements in randomized order
        for x in non_required:
            if time.time() > deadline:
                break
            if _can_add_fast(current, x, current_sums):
                current.append(x)
                for y in current[:-1]:
                    current_sums.add(x + y)
                current_sums.add(2 * x)
        
        # Local search: try to swap elements to get more
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_set = set(current)
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                # Try removing current[i] and adding a different element
                removed = current[i]
                temp_current = [x for j, x in enumerate(current) if j != i]
                temp_sums = _get_sums(temp_current)
                
                # Try adding elements not in current
                for x in non_required:
                    if x in current_set or x == removed:
                        continue
                    if _can_add_fast(temp_current, x, temp_sums):
                        # Found an improvement
                        current = temp_current + [x]
                        current_sums = _get_sums(current)
                        current_set = set(current)
                        improved = True
                        break
                if improved:
                    break
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = sorted(current)
            
            # Early termination if we reach 20
            if best_size >= 20:
                break
    
    return sorted(best_set)

def _can_add_fast(seq, x, sums_set):
    """Check if x can be added to seq given existing sums."""
    for y in seq:
        s = x + y
        if s in sums_set:
            return False
    if 2 * x in sums_set:
        return False
    return True

def _get_sums(seq):
    """Compute all pairwise sums of seq."""
    sums = set()
    n = len(seq)
    for i in range(n):
        for j in range(i, n):
            sums.add(seq[i] + seq[j])
    return sums
