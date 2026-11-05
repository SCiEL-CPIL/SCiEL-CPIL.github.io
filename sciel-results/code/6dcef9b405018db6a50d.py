import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    fixed = {1, 2, 5, 11}
    best_set = list(fixed)
    best_size = len(fixed)
    
    # Pre-calculate initial sums
    init_sums = set()
    fixed_list = list(fixed)
    for i in range(len(fixed_list)):
        for j in range(i, len(fixed_list)):
            init_sums.add(fixed_list[i] + fixed_list[j])
            
    def get_sums(s):
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.add(s[i] + s[j])
        return sums

    def is_valid_addition(x, current_set, current_sums):
        # Check if x can be added to current_set
        # Sums involving x: x + y for y in current_set, and x + x
        for y in current_set:
            if x + y in current_sums:
                return False
        if x + x in current_sums:
            return False
        return True

    def add_to_set(x, current_set, current_sums):
        for y in current_set:
            current_sums.add(x + y)
        current_sums.add(x + x)
        current_set.append(x)
        current_set.sort() # Keep sorted for consistency, though not strictly needed for sum check

    def remove_from_set(x, current_set, current_sums):
        # Remove sums involving x
        for y in current_set:
            if y != x:
                current_sums.discard(x + y)
        current_sums.discard(x + x)
        current_set.remove(x)

    # Randomized Local Search / Destroy and Repair
    # We maintain a current state and try to improve it.
    
    # Initial greedy fill
    current_set = list(fixed)
    current_sums = get_sums(current_set)
    
    # Greedy add random candidates
    candidates = [x for x in range(1, 285) if x not in fixed]
    random.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline:
            break
        if is_valid_addition(x, current_set, current_sums):
            add_to_set(x, current_set, current_sums)
            
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = list(current_set)
        
    # Local search loop
    # Strategy: Randomly remove k elements, then try to re-add elements to maximize size.
    # We prioritize adding elements that allow the set to grow.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Destroy
        # Remove 1 to 3 random non-fixed elements
        removable = [x for x in current_set if x not in fixed]
        if not removable:
            break
            
        num_remove = min(len(removable), random.randint(1, 3))
        to_remove = random.sample(removable, num_remove)
        
        for x in to_remove:
            remove_from_set(x, current_set, current_sums)
            
        # Repair: Try to add elements back or new ones
        # We want to maximize size. 
        # Let's try to add all possible elements in random order.
        # But we want to recover the size and potentially exceed it.
        
        # Create a list of candidates: the removed ones + all other non-fixed candidates not in set
        # To be efficient, we can just iterate through all 1..284 not in current_set
        # But that's 280 iterations. Fast enough.
        
        # Shuffle candidates to break symmetry
        all_candidates = [x for x in range(1, 285) if x not in current_set]
        random.shuffle(all_candidates)
        
        # Prioritize the removed elements? Maybe.
        # Let's just shuffle everything.
        
        for x in all_candidates:
            if time.time() > deadline:
                break
            if is_valid_addition(x, current_set, current_sums):
                add_to_set(x, current_set, current_sums)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
            # If we hit the theoretical max (20), we can stop
            if best_size >= 20:
                break
                
    return best_set
