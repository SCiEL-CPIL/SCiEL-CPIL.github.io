import time
import random

def construct():
    start = time.time()
    deadline = start + 80  # Safety margin before 90s kill
    
    # Incremental DP for subset sums by cardinality
    # dp[c] is a set of achievable sums using exactly c elements
    def compute_sums(s):
        dp = [set() for _ in range(len(s) + 1)]
        dp[0].add(0)
        for x in s:
            # Iterate backwards to avoid using x multiple times in the same step
            for c in range(len(s) - 1, -1, -1):
                # New sums of size c+1 are old sums of size c plus x
                dp[c + 1].update(sum_val + x for sum_val in dp[c])
        return dp

    def is_valid(s):
        dp = compute_sums(s)
        all_sums = set()
        for c in range(len(s) + 1):
            if not all_sums.isdisjoint(dp[c]):
                return False
            all_sums.update(dp[c])
        return True

    # Greedy construction
    # We want to find the largest set of distinct integers in 1..256
    # such that subset sums of different cardinalities are disjoint.
    
    # Strategy: 
    # 1. Start with empty set.
    # 2. Try to add the smallest possible integer that maintains validity.
    # 3. If stuck, backtrack or try random restarts with different initial seeds?
    # Given the 90s limit, a greedy approach with some backtracking or randomized restarts is best.
    
    # Let's try a greedy approach first.
    # To maximize size, we generally want smaller numbers to allow more numbers to fit in the sum range.
    # However, the constraint is about disjointness of sums across cardinalities.
    
    # A known valid set is powers of 2: {1, 2, 4, 8, 16, 32, 64, 128} (size 8).
    # Can we do better?
    # With size 9, the max sum is at least 1+2+...+9 = 45. Min sum for c=1 is 1.
    # The condition is that sum(c) != sum(c') for c != c'.
    
    # Let's implement a randomized greedy search.
    # We will try to build a set of size K.
    # Start with K=8 (known valid). Try to extend to 9, 10, etc.
    
    best_set = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Function to try to extend a given set to a larger size
    def try_extend(s, target_size):
        s = sorted(s)
        current_size = len(s)
        if current_size >= target_size:
            return s
        
        # Greedily add elements
        # We can try adding elements in increasing order
        # If we get stuck, we might need to backtrack, but for now let's try a simple greedy
        # and see if we can find a valid set of size target_size.
        
        # Actually, a better strategy is to use local search on a set of fixed size K.
        # But the task is to maximize size.
        
        # Let's try to find a valid set of size K by random search.
        # We'll use a heuristic: start with small numbers, check validity.
        
        pass # Placeholder, we'll implement the main search loop below

    # Main search: Randomized local search to maximize size
    # We'll maintain a best set found so far.
    # We'll try to increase the size by 1 using a greedy addition, and if that fails,
    # we'll try random restarts or local search on the current best set to find a "better" 
    # set that might be extendable.
    
    # Actually, let's just do a greedy construction with backtracking limited by time.
    # Or, even simpler: Randomized greedy.
    
    # Let's try to construct a set of size K greedily.
    # Start with K=1, try to add the smallest valid number.
    # If we reach a dead end, backtrack? 
    # Given the time limit, let's try a few random restarts of a greedy builder.
    
    def greedy_build(seed=0):
        rng = random.Random(seed)
        s = []
        # Try to add elements
        # We want to maximize size.
        # Let's try adding elements from a shuffled list of candidates?
        # Or just try to add the smallest available valid element?
        
        # Strategy: 
        # 1. Start with empty set.
        # 2. Generate a list of candidate elements (1..256).
        # 3. Shuffle them? No, smaller elements are generally better for packing.
        # 4. Try to add the smallest element that is valid.
        # 5. If no element can be added, stop.
        
        s = []
        # We'll iterate through potential additions.
        # To avoid getting stuck in a local minimum where we pick 1, 2, 4... and get stuck at 8,
        # we can try different starting points or randomization.
        
        # Let's try a greedy approach that prioritizes smaller numbers but allows some randomness
        # to escape bad local minima.
        
        # Actually, the powers of 2 work. 
        # {1, 2, 4, 8, 16, 32, 64, 128} is size 8.
        # Can we add 256? 
        # Sums for 128: 128. Sums for 256: 256.
        # But we need to check ALL subset sums.
        
        # Let's just run a greedy builder that tries to add the smallest valid element.
        # If it fails to grow beyond 8, we try random restarts.
        
        current_s = []
        # We'll try to add elements.
        # To be efficient, we'll precompute sums incrementally.
        
        # Let's use a simple loop:
        # Try to add x from 1 to 256.
        # If valid, add it and restart the loop from 1 (or continue?).
        # Restarting from 1 allows us to fill in gaps if we skipped a small number.
        
        # However, adding 1 first is usually good.
        
        # Let's try this:
        # 1. Start with empty set.
        # 2. While we can add an element:
        #    a. Find the smallest x in 1..256 not in set such that set+{x} is valid.
        #    b. If found, add it.
        #    c. If not found, break.
        
        # This is O(256 * 2^k) in the worst case, but k is small (<= 10 or so).
        # 2^10 = 1024. 256 * 1024 * 10 = ~2.6 million ops. This is fast enough.
        
        s = []
        while True:
            added = False
            for x in range(1, 257):
                if x in s:
                    continue
                new_s = s + [x]
                if is_valid(new_s):
                    s = new_s
                    added = True
                    break
            if not added:
                break
        return s

    # Run greedy build with a few random seeds? 
    # The greedy build above is deterministic (always picks smallest x).
    # To find larger sets, we might need to pick non-smallest x sometimes.
    # Let's modify greedy_build to accept a parameter for randomness.
    
    def greedy_build_random(seed=0):
        rng = random.Random(seed)
        s = []
        # We'll try to add elements.
        # Instead of always picking the smallest, we pick a random valid element?
        # Or we pick from a range?
        
        # Let's try: 
        # 1. Start with empty set.
        # 2. While we can add an element:
        #    a. Find ALL valid x in 1..256 not in s.
        #    b. If none, break.
        #    c. Pick one randomly (or with some bias towards smaller ones).
        
        s = []
        while True:
            valid_candidates = []
            for x in range(1, 257):
                if x in s:
                    continue
                new_s = s + [x]
                if is_valid(new_s):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            # Bias towards smaller numbers to keep sums small
            # We can weight the choice by 1/x or just pick the smallest with some probability
            # Let's just pick the smallest 50% of the time, random otherwise.
            if rng.random() < 0.5:
                x = min(valid_candidates)
            else:
                x = rng.choice(valid_candidates)
            
            s.append(x)
        
        return s

    best_set = greedy_build()
    best_size = len(best_set)
    
    # Try random restarts to see if we can do better
    num_restarts = 1000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        s = greedy_build_random(i)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            # If we found a larger set, we can try to extend it further greedily
            # (though greedy_build_random already does its best)
    
    # Final check and return
    if not is_valid(best_set):
        # Fallback to powers of 2
        best_set = [1, 2, 4, 8, 16, 32, 64, 128]
        
    return [int(x) for x in best_set]
