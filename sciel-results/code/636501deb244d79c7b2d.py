import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    prescribed = [1, 2, 5, 11]
    all_elements = list(range(1, 285))
    # Candidates are elements not in prescribed
    candidates = [x for x in all_elements if x not in prescribed]
    
    best_set = prescribed.copy()
    
    # Try to find a set of size 20, then 19, etc.
    for target_size in range(20, len(prescribed), -1):
        if time.time() > deadline:
            break
            
        found = search_for_size(target_size, prescribed, candidates, start_time, deadline)
        if found:
            best_set = found
            # Since we iterate from high to low, the first found is the max
            # We can break if we found 20, as that's the theoretical max
            if target_size == 20:
                break
                
    return best_set

def search_for_size(target_size, initial_set, candidates, start_time, deadline):
    # We need to pick (target_size - len(initial_set)) elements from candidates
    num_to_pick = target_size - len(initial_set)
    
    # Shuffle candidates to avoid bias
    # We can use a local random seed or just shuffle
    cand_copy = candidates.copy()
    random.shuffle(cand_copy)
    
    current_set = initial_set.copy()
    current_sums = set()
    for i, x in enumerate(current_set):
        for y in current_set[i:]:
            current_sums.add(x + y)
            
    def backtrack(index, current_set, current_sums, chosen):
        if time.time() > deadline:
            return None
            
        if len(chosen) == num_to_pick:
            return current_set.copy()
            
        # Pruning: if not enough candidates left
        if len(cand_copy) - index < num_to_pick - len(chosen):
            return None
            
        for i in range(index, len(cand_copy)):
            x = cand_copy[i]
            
            # Check if x can be added
            new_sums = {x + a for a in current_set} | {2 * x}
            if len(new_sums) == len(current_set) + 1 and not new_sums & current_sums:
                # Add x
                new_set = current_set + [x]
                new_sums_set = current_sums | new_sums
                new_chosen = chosen + [x]
                
                result = backtrack(i + 1, new_set, new_sums_set, new_chosen)
                if result:
                    return result
                    
        return None

    # To increase robustness, we might want to run multiple random shuffles if the first fails quickly
    # But backtracking with pruning is usually strong.
    # Let's add a restart mechanism if it doesn't finish quickly?
    # For now, one deep search per target size.
    
    # If the single shuffle fails, we might miss a solution.
    # Let's run a few restarts if time permits.
    attempts = 0
    max_attempts = 100 # Limit attempts to avoid infinite loop if pruning is weak
    
    while attempts < max_attempts and time.time() < deadline:
        random.shuffle(cand_copy)
        result = backtrack(0, initial_set.copy(), 
                           {x+y for i,x in enumerate(initial_set) for y in initial_set[i:]}, 
                           [])
        if result:
            return result
        attempts += 1
        
    return None
