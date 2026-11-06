import time
import sys
import random

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin for 60s limit
    max_val = 4096
    
    # Helper to check if adding x is valid
    def can_add(S, used_sums, x):
        k = len(S)
        # Generate new sums
        new_sums = set()
        
        # 1. x + x + x
        s1 = 3 * x
        if s1 in used_sums or s1 in new_sums:
            return False
        new_sums.add(s1)
        
        # 2. x + x + s
        for s in S:
            s2 = 2 * x + s
            if s2 in used_sums or s2 in new_sums:
                return False
            new_sums.add(s2)
            
        # 3. x + s_i + s_j
        for i in range(k):
            for j in range(i, k):
                s3 = x + S[i] + S[j]
                if s3 in used_sums or s3 in new_sums:
                    return False
                new_sums.add(s3)
                
        return True

    def add_to_state(S, used_sums, x):
        k = len(S)
        S_k = S[:-1] # Assuming x was just appended
        sums_to_add = []
        sums_to_add.append(3 * x)
        for s in S_k:
            sums_to_add.append(2 * x + s)
        for i in range(len(S_k)):
            for j in range(i, len(S_k)):
                sums_to_add.append(x + S_k[i] + S_k[j])
        for s_val in sums_to_add:
            used_sums.add(s_val)
        return sums_to_add

    def remove_from_state(S, used_sums, x, sums_to_remove):
        for s_val in sums_to_remove:
            used_sums.remove(s_val)
            
    def get_sums_for(x, S):
        """Calculate the sums contributed by x given the rest of the set S (excluding x)."""
        sums = [3*x]
        for s in S:
            sums.append(2*x + s)
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.append(x + S[i] + S[j])
        return sums

    # Fast Greedy first to get a good initial bound
    S = []
    used_sums = set()
    
    for x in range(1, max_val + 1):
        if can_add(S, used_sums, x):
            S.append(x)
            add_to_state(S, used_sums, x)
            
    best_set = list(S)
    best_len = len(S)
    
    # Local Search
    # We maintain S and used_sums.
    # Moves:
    # 1. Add x (if valid)
    # 2. Remove y, Add x1, Add x2 (if valid) -> Size +1
    # 3. Swap y with x (if valid) -> Size 0, better structure?
    
    # To speed up "Add x", we can keep a list of candidates or just scan.
    # Scanning 1..4096 is ~4096 * O(N^2). If N=16, N^2=256. 4096*256 ~ 1M ops.
    # In Python, this might be 0.1s per scan. We have 55s. So ~500 scans.
    # But we do better: we only scan if we didn't add anything in the previous step?
    # Or we can interleave.
    
    # Let's try a loop that attempts improvements.
    # We will use a "temperature" or just random moves.
    
    # Optimization: Instead of scanning all 1..4096 for every attempt, 
    # we can keep a list of "promising" candidates or just randomize the start point.
    # But for "Add", we need to find ANY valid x.
    # If we just scan 1..4096, it's deterministic. 
    # To explore, we can shuffle the range or use random sampling for the "Add" phase too?
    # If we miss a valid x because we didn't scan it, we might not grow.
    # So for "Grow", we MUST scan all valid candidates or use a very high probability of hitting one.
    # Given the density, scanning is safer for growth.
    
    # For "2-for-1", we can be more random.
    
    no_improvement_count = 0
    max_no_improvement = 50 # Restart strategy?
    
    while time.time() - start_time < DEADLINE:
        improved = False
        
        # Phase 1: Try to add a single element
        # We scan from 1 to max_val. 
        # To avoid bias towards small numbers in later iterations (if we always start at 1),
        # we can rotate the start point or just accept that small numbers are easier to fit?
        # Actually, small numbers are harder to fit because they create smaller sums which might collide?
        # No, larger numbers create larger sums. The range is 1..4096.
        # Sums range up to 3*4096 = 12288.
        
        # Let's just scan 1..4096.
        # To make it non-deterministic and explore, we can start at a random offset?
        # No, if we miss a valid x at index 50 because we started at 100, we fail to grow.
        # So we must check all.
        
        # However, checking all 4096 every time is expensive if we do it many times.
        # We can break early if we find one.
        
        added = False
        # Randomize the scan order? 
        # If we randomize, we might find a "worse" x (one that blocks future growth) 
        # vs a "better" x. 
        # Greedy usually picks the smallest valid x. 
        # Let's stick to smallest valid x for stability, but allow swaps later.
        
        for x in range(1, max_val + 1):
            if x in S: # Optimization: check if already in set? S is small, linear search or set?
                continue
            if can_add(S, used_sums, x):
                S.append(x)
                add_to_state(S, used_sums, x)
                best_set = list(S)
                best_len = len(S)
                improved = True
                added = True
                break # Try to add only one at a time to keep it simple? 
                      # Or continue adding?
                      # If we continue, we might fill up quickly.
                      # Let's break to allow other moves to be considered.
        
        if not added:
            # Phase 2: Try 2-for-1 move
            # Pick a random element to remove
            if len(S) > 2:
                idx_remove = random.randint(0, len(S) - 1)
                y = S[idx_remove]
                
                # Remove y
                sums_y = get_sums_for(y, [s for i, s in enumerate(S) if i != idx_remove])
                # Wait, get_sums_for needs the set WITHOUT y.
                S_temp = S[:idx_remove] + S[idx_remove+1:]
                sums_y = get_sums_for(y, S_temp)
                
                # Remove from used_sums
                for s_val in sums_y:
                    used_sums.remove(s_val)
                S.pop(idx_remove)
                
                # Now try to add two elements
                # We need to find x1, x2 such that:
                # 1. can_add(S, used_sums, x1)
                # 2. can_add(S + [x1], used_sums + sums(x1), x2)
                
                # We can sample x1 from 1..4096
                # If we find x1, we sample x2.
                
                success = False
                # Try a few random x1s
                for _ in range(10): # Limit attempts to save time
                    x1 = random.randint(1, max_val)
                    if x1 in S:
                        continue
                    if can_add(S, used_sums, x1):
                        # Tentatively add x1
                        sums_x1 = get_sums_for(x1, S)
                        S.append(x1)
                        for s_val in sums_x1:
                            used_sums.add(s_val)
                        
                        # Now try to add x2
                        for _ in range(10):
                            x2 = random.randint(1, max_val)
                            if x2 in S:
                                continue
                            if x2 == x1:
                                continue
                            if can_add(S, used_sums, x2):
                                # Success! Add x2
                                sums_x2 = get_sums_for(x2, S)
                                S.append(x2)
                                for s_val in sums_x2:
                                    used_sums.add(s_val)
                                success = True
                                improved = True
                                best_set = list(S)
                                best_len = len(S)
                                break
                            
                        if success:
                            break
                        else:
                            # Rollback x1
                            S.pop()
                            for s_val in sums_x1:
                                used_sums.remove(s_val)
                
                if not success:
                    # Rollback removal of y
                    S.append(y)
                    for s_val in sums_y:
                        used_sums.add(s_val)
                    # Note: S order might change, but used_sums is correct.
                    # S is a list, order doesn't matter for validity, only content.
                    # But for determinism in future steps, it's fine.
                    
            # Phase 3: Random Swap (to escape local minima)
            # Swap y with x
            if len(S) > 0 and random.random() < 0.2: # 20% chance to swap if no growth
                idx_remove = random.randint(0, len(S) - 1)
                y = S[idx_remove]
                S_temp = S[:idx_remove] + S[idx_remove+1:]
                sums_y = get_sums_for(y, S_temp)
                
                for s_val in sums_y:
                    used_sums.remove(s_val)
                S.pop(idx_remove)
                
                swapped = False
                for _ in range(10):
                    x = random.randint(1, max_val)
                    if x in S:
                        continue
                    if can_add(S, used_sums, x):
                        S.append(x)
                        sums_x = get_sums_for(x, S)
                        for s_val in sums_x:
                            used_sums.add(s_val)
                        swapped = True
                        break
                
                if not swapped:
                    # Rollback
                    S.append(y)
                    for s_val in sums_y:
                        used_sums.add(s_val)
                else:
                    # Swapped successfully. 
                    # This doesn't improve size, but might allow future growth.
                    # We don't count this as "improved" for size, but it changes state.
                    pass
        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > 100:
                # Restart with a different random seed or strategy?
                # For now, just continue. The random swaps should help.
                pass
        else:
            no_improvement_count = 0
            
    return best_set
