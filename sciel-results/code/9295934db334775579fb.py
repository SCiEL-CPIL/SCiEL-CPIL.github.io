import time
import random

P = {'n': 128}

def construct():
    n = P['n']
    deadline = time.time() + 80  # 80s out of 90s
    
    # Initial set: {63, ..., 128} size 66
    S = set(range(63, n + 1))
    best_size = len(S)
    best_set = S.copy()
    
    # Helper to check validity
    def is_valid(s):
        if len(s) < 3:
            return True
        s_list = list(s)
        s_set = set(s_list)
        # Check all triples
        # Optimization: only check if pair sums are in s
        # Precompute sums?
        # For n=128, len(s) ~ 60. Triples ~ 30k. Sums check is O(1).
        # 30k * 60 iterations per check is too slow if called often.
        # But we only call on candidates.
        
        # Faster check:
        # For each a, b in s:
        #   sum_ab = a + b
        #   if sum_ab in s:
        #     # Check if there is c in s \ {a,b} such that a+c in s and b+c in s
        #     # This is expensive.
        
        # Alternative: Iterate triples.
        # With size 66, C(66,3) = 45k.
        # 45k checks per validity call.
        # If we do 1000 calls, 45M ops. Might be tight in Python.
        
        # Let's use a bitset or boolean array for membership.
        present = [False] * (n + 1)
        for x in s:
            present[x] = True
            
        for i in range(len(s_list)):
            a = s_list[i]
            for j in range(i + 1, len(s_list)):
                b = s_list[j]
                sum_ab = a + b
                if sum_ab > n or not present[sum_ab]:
                    continue
                # sum_ab is in S. Now check for c
                for k in range(len(s_list)):
                    if k == i or k == j:
                        continue
                    c = s_list[k]
                    sum_ac = a + c
                    sum_bc = b + c
                    if sum_ac <= n and sum_bc <= n:
                        if present[sum_ac] and present[sum_bc]:
                            return False
        return True

    # Greedy add/remove
    current_set = S.copy()
    
    # Try to add elements from 1..62
    candidates = list(range(1, 63))
    random.shuffle(candidates)
    
    for _ in range(100): # Iterations
        if time.time() > deadline:
            break
            
        improved = False
        # Try adding
        for x in candidates:
            if x in current_set:
                continue
            new_set = current_set | {x}
            if is_valid(new_set):
                current_set = new_set
                improved = True
                break
        
        # Try removing to allow adding? (Local search)
        if not improved:
            # Try swapping
            # Remove one, add two?
            # Simpler: try removing each element and see if we can add more
            for x in list(current_set):
                if time.time() > deadline:
                    break
                trial_set = current_set - {x}
                # Try to add from candidates not in trial_set
                added = False
                for y in candidates:
                    if y in trial_set:
                        continue
                    new_set = trial_set | {y}
                    if is_valid(new_set):
                        if len(new_set) > len(current_set):
                            current_set = new_set
                            improved = True
                            break
                        else:
                            # Same size, maybe keep for diversity?
                            pass
                if improved:
                    break
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
        
        if not improved:
            # Stuck, maybe restart or random shuffle
            # For now, just break if no improvement in a few steps
            pass

    return list(best_set)
