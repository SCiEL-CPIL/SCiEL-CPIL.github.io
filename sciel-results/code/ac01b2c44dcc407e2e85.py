import time
import itertools
import random

def construct():
    start_time = time.time()
    time_limit = 85.0  # Leave 5s buffer
    best_result = []
    
    # Helper to check if adding x is valid
    def can_add(x, selected, existing_sums):
        # New sums involving x:
        # 1. x+x+x
        # 2. x+x+y for y in selected
        # 3. x+y+z for y <= z in selected
        
        new_sums = set()
        
        # 1. 3x
        s = 3 * x
        if s in existing_sums:
            return False
        new_sums.add(s)
        
        # 2. 2x + y
        for y in selected:
            s = 2 * x + y
            if s in existing_sums or s in new_sums:
                return False
            new_sums.add(s)
            
        # 3. x + y + z
        # y <= z from selected
        for y, z in itertools.combinations_with_replacement(selected, 2):
            s = x + y + z
            if s in existing_sums or s in new_sums:
                return False
            new_sums.add(s)
            
        return True
    
    def greedy_search(candidate_order):
        """Run greedy search with given candidate ordering."""
        selected = []
        existing_sums = set()
        
        for x in candidate_order:
            if time.time() - start_time > time_limit:
                break
            
            if can_add(x, selected, existing_sums):
                selected.append(x)
                existing_sums.add(3 * x)
                for y in selected[:-1]:
                    existing_sums.add(2 * x + y)
                for y, z in itertools.combinations_with_replacement(selected[:-1], 2):
                    existing_sums.add(x + y + z)
        
        return selected
    
    # Strategy 1: Deterministic greedy (original)
    result1 = greedy_search(range(1, 4097))
    if len(result1) > len(best_result):
        best_result = result1
    
    # Strategy 2: Randomized restarts with different orderings
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5.0:
        attempts = 0
        while time.time() - start_time < time_limit - 2.0:
            attempts += 1
            # Create a shuffled candidate list
            candidates = list(range(1, 4097))
            random.shuffle(candidates)
            
            result = greedy_search(candidates)
            if len(result) > len(best_result):
                best_result = result
            
            # Early exit if we reach a high cardinality
            if len(best_result) >= 40:
                break
    
    # Strategy 3: Try with some bias towards smaller numbers first
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5.0 and len(best_result) < 40:
        for _ in range(3):
            if time.time() - start_time > time_limit - 2.0:
                break
            # Mix of small numbers and random
            candidates = list(range(1, 200)) + list(range(200, 4097))
            random.shuffle(candidates[199:])  # Keep small numbers in front
            result = greedy_search(candidates)
            if len(result) > len(best_result):
                best_result = result
    
    return best_result
