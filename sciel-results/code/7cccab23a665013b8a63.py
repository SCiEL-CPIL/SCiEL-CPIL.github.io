import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 4096
    best_set = []
    best_len = 0
    
    def is_valid_set(S):
        """Check if set S has all unique unordered triple sums."""
        S = list(S)
        k = len(S)
        sums = set()
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    s = S[i] + S[j] + S[l]
                    if s in sums:
                        return False
                    sums.add(s)
        return True
    
    def get_new_sums(x, S):
        """Get all new triple sums when adding x to set S."""
        new_sums = set()
        new_sums.add(3 * x)
        for y in S:
            new_sums.add(2 * x + y)
        for i in range(len(S)):
            for j in range(i, len(S)):
                new_sums.add(x + S[i] + S[j])
        return new_sums
    
    def can_add(x, S, existing_sums):
        """Check if x can be added to S without collision."""
        new_sums = get_new_sums(x, S)
        if new_sums & existing_sums:
            return False, None
        # Check internal uniqueness
        expected = 1 + len(S) + len(S) * (len(S) + 1) // 2
        if len(new_sums) != expected:
            return False, None
        return True, new_sums
    
    # Strategy 1: Power of 3 base
    power3_set = [3**i for i in range(8) if 3**i <= n]
    # power3_set = [1, 3, 9, 27, 81, 243, 729, 2187]
    
    # Verify power of 3 set is valid
    if is_valid_set(power3_set):
        best_set = list(power3_set)
        best_len = len(power3_set)
    
    # Strategy 2: Greedy smallest-first from scratch
    S_greedy = []
    sums_greedy = set()
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        valid, new_sums = can_add(x, S_greedy, sums_greedy)
        if valid:
            S_greedy.append(x)
            sums_greedy.update(new_sums)
    
    if len(S_greedy) > best_len:
        best_set = S_greedy
        best_len = len(S_greedy)
    
    # Strategy 3: Randomized greedy with multiple seeds
    for seed in range(200):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        S = []
        existing_sums = set()
        
        for x in candidates:
            if time.time() > deadline:
                break
            valid, new_sums = can_add(x, S, existing_sums)
            if valid:
                S.append(x)
                existing_sums.update(new_sums)
        
        if len(S) > best_len:
            best_set = S
            best_len = len(S)
    
    # Strategy 4: Hill climbing - try to improve current best
    # Try to add elements or swap to increase size
    if best_len < 80:
        S = list(best_set)
        existing_sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                for k in range(j, len(S)):
                    existing_sums.add(S[i] + S[j] + S[k])
        
        # Try to add elements
        for x in range(1, n + 1):
            if time.time() > deadline:
                break
            if x in S:
                continue
            valid, new_sums = can_add(x, S, existing_sums)
            if valid:
                S.append(x)
                existing_sums.update(new_sums)
        
        if len(S) > best_len:
            best_set = S
            best_len = len(S)
        
        # Try swaps: remove one element, add a different one
        for _ in range(500):
            if time.time() > deadline:
                break
            
            S_copy = list(best_set)
            existing_sums_copy = set()
            for i in range(len(S_copy)):
                for j in range(i, len(S_copy)):
                    for k in range(j, len(S_copy)):
                        existing_sums_copy.add(S_copy[i] + S_copy[j] + S_copy[k])
            
            # Try removing a random element and adding a new one
            if len(S_copy) == 0:
                break
            
            idx_to_remove = random.randint(0, len(S_copy) - 1)
            removed = S_copy[idx_to_remove]
            S_copy.pop(idx_to_remove)
            
            # Recompute sums after removal
            existing_sums_copy = set()
            for i in range(len(S_copy)):
                for j in range(i, len(S_copy)):
                    for k in range(j, len(S_copy)):
                        existing_sums_copy.add(S_copy[i] + S_copy[j] + S_copy[k])
            
            # Try to add a new element
            added = False
            for x in range(1, n + 1):
                if time.time() > deadline:
                    break
                if x in S_copy or x == removed:
                    continue
                valid, new_sums = can_add(x, S_copy, existing_sums_copy)
                if valid:
                    S_copy.append(x)
                    existing_sums_copy.update(new_sums)
                    added = True
                    break
            
            if added and len(S_copy) > best_len:
                best_set = S_copy
                best_len = len(S_copy)
    
    # Final validation
    if best_set and not is_valid_set(best_set):
        # Fallback to power of 3
        best_set = [3**i for i in range(8) if 3**i <= n]
    
    return best_set
