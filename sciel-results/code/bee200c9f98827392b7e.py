import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Candidate pool: [128, 256]
    candidates = list(range(128, 257))
    
    best_set = []
    best_size = 0
    
    def get_sums(S):
        sums = set()
        for x in S:
            new_sums = {x + s for s in sums}
            new_sums.add(x)
            sums |= new_sums
        return sums

    def can_add(sums, x):
        # Check if adding x to set with sums 'sums' is valid
        # New sums are {x} U {x+s for s in sums}
        # We need to check:
        # 1. No duplicate sums (x not in sums, x+s not in sums, x+s1 == x+s2)
        # 2. No divisibility between new sums and old sums
        # 3. No divisibility between new sums
        
        new_sums = set()
        new_sums.add(x)
        for s in sums:
            new_sums.add(x + s)
            
        # Check duplicates with old sums
        if new_sums & sums:
            return None
            
        # Check divisibility
        # We can optimize by checking only new sums against old sums and other new sums
        # For efficiency, we'll check all pairs in (sums U new_sums) but that's O(N^2).
        # Better: Check new sums against (sums U new_sums).
        
        all_sums = sums | new_sums
        # To speed up, we can sort and check, but for small sets (size < 10, sums < 1000), O(N^2) is fine.
        # Max sums for size 10 is 2^10 - 1 = 1023.
        # 1023^2 ~ 1M checks. Might be slow if done millions of times.
        
        # Optimization:
        # If a | b, then b >= 2a.
        # We can iterate through new_sums and check against all_sums.
        
        for n in new_sums:
            for a in all_sums:
                if a == n:
                    continue
                if a > n:
                    if a % n == 0:
                        return None
                else: # a < n
                    if n % a == 0:
                        return None
        return all_sums

    def try_greedy(order):
        nonlocal best_set, best_size
        current_set = []
        current_sums = set()
        
        for x in order:
            new_sums = can_add(current_sums, x)
            if new_sums is not None:
                current_set.append(x)
                current_sums = new_sums
                
        if len(current_set) > best_size:
            best_set = list(current_set)
            best_size = len(best_set)
            # Optional: print or log
            # print(f"Found size {best_size}: {best_set}")
            
        return len(current_set)

    # Strategy 1: Randomized Greedy
    random.seed(42)
    for i in range(2000):
        if time.time() > deadline - 2.0:
            break
        random.shuffle(candidates)
        try_greedy(candidates)
        
    # Strategy 2: Ordered Greedy (Asc, Desc)
    try_greedy(sorted(candidates))
    try_greedy(sorted(candidates, reverse=True))
    
    # Strategy 3: Local Search / Swap
    # If we have a best set, try swapping elements to see if we can add more.
    # This is complex. Let's stick to more random restarts with different biases.
    
    # Bias towards larger numbers?
    # Bias towards smaller numbers?
    
    # Let's try a "simulated annealing" like approach or just more random trials.
    # To make random trials more effective, we can prioritize candidates that are "far" from existing sums in terms of divisibility.
    # But simple random shuffle works surprisingly well for these "large number" problems.
    
    # Let's add a check: if we get stuck early, restart.
    
    # Final verification
    if best_set:
        final_sums = get_sums(best_set)
        # Verify validity
        valid = True
        sums_list = sorted(final_sums)
        for i in range(len(sums_list)):
            for j in range(i+1, len(sums_list)):
                if sums_list[j] % sums_list[i] == 0:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            best_set = [128, 129, 131] # Fallback

    return best_set
