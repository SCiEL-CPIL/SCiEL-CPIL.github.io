import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 4096
    best_set = []
    best_size = 0
    
    # Helper to check if c can be added to S given used_sums
    # Returns True if valid, False otherwise
    def can_add(S, c, used_sums):
        # Generate new sums
        new_sums = []
        
        # 1. 3c
        s = 3 * c
        if s in used_sums:
            return False
        new_sums.append(s)
        
        # 2. 2c + x
        for x in S:
            s = 2 * c + x
            if s in used_sums:
                return False
            new_sums.append(s)
            
        # 3. c + x + y
        len_S = len(S)
        for i in range(len_S):
            x = S[i]
            for j in range(i, len_S):
                y = S[j]
                s = c + x + y
                if s in used_sums:
                    return False
                new_sums.append(s)
                
        # Check internal duplicates in new_sums
        # Since S is sorted, and c is fixed, are there duplicates?
        # 3c is unique.
        # 2c+x is unique for distinct x.
        # c+x+y: x+y is unique for unordered pairs (x,y) from S?
        # Not necessarily. 1+4 = 2+3.
        # So we must check.
        if len(new_sums) != len(set(new_sums)):
            return False
            
        return True

    def add_to_state(S, c, used_sums):
        new_sums = [3 * c]
        for x in S:
            new_sums.append(2 * c + x)
        len_S = len(S)
        for i in range(len_S):
            x = S[i]
            for j in range(i, len_S):
                y = S[j]
                new_sums.append(c + x + y)
        used_sums.update(new_sums)
        S.append(c)

    def remove_from_state(S, c, used_sums):
        # S must have c as the last element
        S.pop()
        new_sums = [3 * c]
        for x in S:
            new_sums.append(2 * c + x)
        len_S = len(S)
        for i in range(len_S):
            x = S[i]
            for j in range(i, len_S):
                y = S[j]
                new_sums.append(c + x + y)
        for s in new_sums:
            used_sums.discard(s)

    # 1. Deterministic Greedy Baseline
    S = []
    used_sums = set()
    for c in range(1, n + 1):
        if time.time() > deadline:
            break
        if can_add(S, c, used_sums):
            add_to_state(S, c, used_sums)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
                
    # 2. Randomized Backtracking Search
    # We try to extend the greedy solution or start fresh with randomization.
    # Strategy: Restart with random candidates, use backtracking to fix conflicts.
    
    max_restarts = 500
    restarts = 0
    
    while time.time() < deadline and restarts < max_restarts:
        restarts += 1
        S = []
        used_sums = set()
        
        # Shuffle candidates to randomize the order of selection
        # To speed up, we might bias towards smaller numbers initially, 
        # but random is fine for exploration.
        # Let's try a mix: first few greedy, then random backtracking.
        
        # We'll use a recursive-like iterative backtracking
        # State: S, used_sums, index in candidate list
        # To keep it simple and fast, we'll do a limited depth backtracking.
        
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        # We'll try to build a set. If we get stuck, we backtrack.
        # Instead of full recursion, we'll use a stack.
        # Stack entries: (current_S, current_used_sums, candidate_index)
        # This is memory heavy. 
        # Alternative: Randomized Greedy with "undo" of last few steps.
        
        # Let's try a simpler approach: Randomized Greedy with multiple passes.
        # If we can't add the next candidate, skip it.
        # To improve, we can randomly shuffle and try to add.
        
        # Better: Use the greedy result as a seed, then try to swap elements.
        # But for now, let's do randomized greedy with backtracking on the last 5 elements.
        
        # Simplified: Just run randomized greedy multiple times.
        # The previous program did this but was slow.
        # Let's optimize the random greedy.
        
        for c in candidates:
            if time.time() > deadline:
                break
            if can_add(S, c, used_sums):
                add_to_state(S, c, used_sums)
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S[:]
                    
    return best_set
