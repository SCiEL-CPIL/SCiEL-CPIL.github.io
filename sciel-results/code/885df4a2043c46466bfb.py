import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    fixed = {1, 2, 5, 11}
    best_set = list(fixed)
    best_size = len(fixed)
    
    # We will try to find a set of size 20, then 19, etc.
    # Or just run a single search that tries to maximize size.
    
    def search(S, sums, target_size, depth_limit=None):
        nonlocal best_set, best_size, deadline
        
        if time.time() > deadline:
            return None
            
        if len(S) == target_size:
            return S
            
        # If we are deeper than current best, update best?
        # The search is for a specific target size.
        # If we find a full solution, we return it.
        
        candidates = []
        # To speed up, we can iterate through 1..284
        # But we can also just pick from a pre-generated list of all numbers not in S
        # For simplicity, iterate 1..284
        
        # Optimization: Sort candidates? 
        # Random order is better for finding *a* solution.
        # Let's generate list of valid candidates first.
        
        valid_candidates = []
        for x in range(1, 285):
            if x in S:
                continue
            
            # Check if x can be added
            # 1. 2x not in sums
            if 2*x in sums:
                continue
            
            # 2. x+y not in sums for all y in S
            ok = True
            for y in S:
                if x + y in sums:
                    ok = False
                    break
            
            if ok:
                valid_candidates.append(x)
                
        if not valid_candidates:
            return None
            
        # Shuffle to randomize search path
        random.shuffle(valid_candidates)
        
        for x in valid_candidates:
            new_S = S | {x}
            new_sums = set(sums)
            # Add new sums
            new_sums.add(2*x)
            for y in S:
                new_sums.add(x + y)
                
            # Recurse
            result = search(new_S, new_sums, target_size)
            if result is not None:
                return result
                
        return None

    # Try to find size 20
    # We might need multiple restarts if the first one fails
    # But backtracking is complete, so if it's solvable, it will find it (eventually).
    # However, it might be slow.
    # We can limit the time for each target size.
    
    # Let's try a simpler approach: 
    # Run the backtracking for target_size = 20.
    # If it doesn't finish in 50s, we might be stuck in a bad branch.
    # Actually, backtracking is deterministic if we don't shuffle, but we do shuffle.
    # So each run is different.
    
    # Strategy: 
    # 1. Try to build size 20.
    # 2. If time runs out, return best found (which might be less than 20 if we track partials, but search returns None on failure).
    # To track partials, we can modify search to update best_set if len(S) > best_size.
    
    def search_with_tracking(S, sums, target_size):
        nonlocal best_set, best_size, deadline
        
        if time.time() > deadline:
            return None
            
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            
        if len(S) == target_size:
            return S
            
        valid_candidates = []
        for x in range(1, 285):
            if x in S:
                continue
            
            if 2*x in sums:
                continue
            
            ok = True
            for y in S:
                if x + y in sums:
                    ok = False
                    break
            
            if ok:
                valid_candidates.append(x)
                
        if not valid_candidates:
            return None
            
        random.shuffle(valid_candidates)
        
        for x in valid_candidates:
            new_S = S | {x}
            new_sums = set(sums)
            new_sums.add(2*x)
            for y in S:
                new_sums.add(x + y)
                
            result = search_with_tracking(new_S, new_sums, target_size)
            if result is not None:
                return result
                
        return None

    # Set a seed for reproducibility of the random shuffle? 
    # No, we want diversity. But for debugging, maybe.
    # The evaluator runs it once.
    
    # Try to find a set of size 20
    # If it fails, try 19, etc? 
    # Actually, if we can't find 20, we might still have found a large set in best_set.
    # But search_with_tracking updates best_set as it goes deeper.
    # So if it backtracks all the way, best_set will be the max depth reached.
    
    # We should probably try multiple targets or just run it and see.
    # Since 20 is the max, we aim for 20.
    
    # To increase chances, we can restart the search a few times if it fails quickly?
    # Or just let it run.
    
    # Let's run it for target_size 20.
    # If it takes too long, it will hit deadline and return None, but best_set will be updated.
    
    search_with_tracking(fixed, set([1+1, 1+2, 1+5, 1+11, 2+2, 2+5, 2+11, 5+5, 5+11, 11+11]), 20)
    
    # If we didn't find 20, best_set might be size < 20.
    # We can try to improve it with local search? 
    # Or just return best_set.
    
    return best_set
