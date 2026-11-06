import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin before 60s limit
    
    # Helper to check if a set of integers is a valid B3 set
    # Optimized: Use a set of sums.
    def is_valid_set(S):
        if not S:
            return True
        sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    val = S[i] + S[j] + S[k]
                    if val in sums:
                        return False
                    sums.add(val)
        return True

    def check_add(S, x, sums_set):
        """Check if adding x to S preserves unique triple sums."""
        n = len(S)
        new_sums = set()
        
        # (x,x,x)
        val = 3 * x
        if val in sums_set or val in new_sums:
            return False, None
        new_sums.add(val)
        
        # (x,x,s) for each s in S
        for s in S:
            val = 2 * x + s
            if val in sums_set or val in new_sums:
                return False, None
            new_sums.add(val)
        
        # (x,s,t) for s <= t in S
        for i in range(n):
            for j in range(i, n):
                val = x + S[i] + S[j]
                if val in sums_set or val in new_sums:
                    return False, None
                new_sums.add(val)
        
        return True, new_sums

    def greedy_forward(start_val=1, end_val=4096, step=1):
        S = []
        sums_set = set()
        for x in range(start_val, min(end_val + 1, 4097), step):
            valid, new_sums = check_add(S, x, sums_set)
            if valid:
                S.append(x)
                sums_set.update(new_sums)
        return S, sums_set

    def randomized_greedy(seed_offset=0):
        S = []
        sums_set = set()
        candidates = list(range(1, 4097))
        random.seed(42 + seed_offset)
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if x in S:
                continue
            valid, new_sums = check_add(S, x, sums_set)
            if valid:
                S.append(x)
                sums_set.update(new_sums)
        return S, sums_set

    best_S = []
    best_sums = set()
    best_size = 0

    # Phase 1: Get a good initial solution using greedy
    # Try a few different greedy strategies
    for start in [1, 2, 3, 10, 100]:
        if time.time() > deadline - 20.0:
            break
        S, sums = greedy_forward(start, 4096, 1)
        if len(S) > best_size:
            best_S = S
            best_sums = sums
            best_size = len(S)
            
    # Also try randomized greedy to find a diverse start
    for i in range(5):
        if time.time() > deadline - 20.0:
            break
        S, sums = randomized_greedy(i)
        if len(S) > best_size:
            best_S = S
            best_sums = sums
            best_size = len(S)

    # Phase 2: Local Search with 1-opt and 2-opt moves
    # We want to maximize the size of the set.
    # 1-opt: Add one element.
    # 2-opt: Remove two, add two.
    
    S = best_S
    sums_set = best_sums
    S_set = set(S)
    
    # Pre-compute all possible sums for current S? No, sums_set is maintained.
    # For 2-opt, we need to check validity of S \ {u, w} U {v, z}.
    # This is expensive. We will only try 2-opt if 1-opt fails.
    
    stall_count = 0
    max_stalls = 100 # If we can't improve for 100 random tries, restart or give up
    
    while time.time() < deadline - 2.0:
        if stall_count > max_stalls:
            # Try a different randomized greedy start
            stall_count = 0
            max_stalls += 50 # Be more patient?
            # Or just break if we've been running for a while
            if len(S) < 20: # If we are really stuck at a small size, restart
                 for i in range(5):
                     if time.time() > deadline - 2.0: break
                     S_new, sums_new = randomized_greedy(random.randint(0, 1000))
                     if len(S_new) > len(S):
                         S = S_new
                         sums_set = sums_new
                         S_set = set(S)
                         stall_count = 0
                 continue
            else:
                 # We are stuck at a local maximum. 
                 # Try a random restart with a bias towards different structures
                 # e.g. start with larger numbers
                 candidates = list(range(100, 4097))
                 random.shuffle(candidates)
                 S_new = []
                 sums_new = set()
                 for x in candidates:
                     if time.time() > deadline - 2.0: break
                     valid, ns = check_add(S_new, x, sums_new)
                     if valid:
                         S_new.append(x)
                         sums_new.update(ns)
                 if len(S_new) > len(S):
                     S = S_new
                     sums_set = sums_new
                     S_set = set(S)
                     stall_count = 0
                 else:
                     break # Give up if we can't improve
        
        # Try 1-opt: Add a random element
        improved = False
        # Sample candidates not in S
        # To speed up, we can just pick random numbers from 1 to 4096
        for _ in range(100): # Try 100 random additions
            if time.time() > deadline - 2.0:
                break
            x = random.randint(1, 4096)
            if x in S_set:
                continue
            valid, new_sums = check_add(S, x, sums_set)
            if valid:
                S.append(x)
                sums_set.update(new_sums)
                S_set.add(x)
                improved = True
                break
        
        if improved:
            stall_count = 0
            continue
        else:
            stall_count += 1
            
            # Try 2-opt: Remove 2, Add 2
            # This is expensive. Only try if we have space in time.
            if len(S) >= 2 and random.random() < 0.1: # 10% chance to try 2-opt
                # Pick two elements to remove
                u, w = random.sample(S, 2)
                # We need to find v, z such that (S - {u, w}) + {v, z} is valid.
                # This is hard to search directly.
                # Instead, let's try to "swap" one element at a time to reach a state where we can add?
                # Or simply, try to replace u with v. If valid, do it.
                # If we can't add, maybe we can swap to a "better" configuration?
                # But we only care about size. So swaps that keep size same are only useful if they enable future additions.
                # This is complex. Let's stick to 1-opt and restarts for now, as 2-opt validity check is O(N^3) per candidate.
                pass

    # Final verification
    if not is_valid_set(S):
        # Fallback to a known small valid set
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        
    return S

# EVOLVE-BLOCK-END
