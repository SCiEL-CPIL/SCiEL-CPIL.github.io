import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0  # 5s margin for safety
    
    best_set = set()
    best_size = 0
    
    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            if is_valid(S):
                best_set = set(S)
                best_size = len(S)
                
    def get_bitset(S):
        reachable = 1
        for x in S:
            reachable |= (reachable << x)
        return reachable

    def is_valid(S):
        reachable = get_bitset(S)
        return ((reachable >> 200) & 1) == 0

    def find_witness(S):
        # Returns a subset T of S summing to 200, or None
        # Using DP with backtracking
        elements = list(S)
        random.shuffle(elements) # Randomize to find different witnesses
        
        # dp[s] = (prev_sum, element) that led to s
        # We only need to store for s up to 200
        dp = [None] * 201
        dp[0] = (0, 0) # Base case
        
        reachable = [False] * 201
        reachable[0] = True
        
        for x in elements:
            # Iterate backwards to avoid using x multiple times in this step
            # But since we are just checking existence, and we want *a* subset,
            # standard 0/1 knapsack DP logic applies.
            # However, we want to reconstruct, so we need to know which element was added.
            # Standard approach:
            # new_reachable[s] = reachable[s] or reachable[s-x]
            # If reachable[s-x] is true and reachable[s] was false, then x is part of a solution for s.
            
            # To be safe and simple, let's use a 2D-like structure or just re-run DP?
            # No, 1D DP with careful backtracking is tricky if we overwrite.
            # Let's use a dictionary or list of (prev_sum, el) for each sum s.
            
            # Actually, since N=100 and Target=200, we can just do:
            # dp[s] = list of elements in the subset? No, too slow.
            
            # Let's stick to the bitset for validity, and a separate reconstruction for witness.
            # Reconstruction:
            # We can store `parent[s]` = (element, prev_sum)
            # But we must ensure we don't use the same element twice.
            # The 1D DP `reachable[s] |= reachable[s-x]` doesn't track history well for 0/1.
            
            # Correct 1D DP for 0/1 knapsack reconstruction:
            # We need to iterate s from Target down to x.
            
            prev = [None] * 201
            prev[0] = (-1, 0) # Sentinel: no element, sum 0
            
            # We need to know if a sum is reachable.
            # Let's just run the DP and store the "last element added" for each sum.
            # If we update dp[s] from dp[s-x], we record x.
            
            # Initialize dp for this run
            # dp[s] is True if reachable
            # path[s] is the element that completed the sum s, and the previous sum
            
            dp = [False] * 201
            dp[0] = True
            path = [None] * 201 # path[s] = (x, s-x)
            
            for x in elements:
                for s in range(200, x - 1, -1):
                    if dp[s - x] and not dp[s]:
                        dp[s] = True
                        path[s] = (x, s - x)
                        
            if dp[200]:
                # Reconstruct
                T = []
                s = 200
                while s > 0:
                    x, prev_s = path[s]
                    T.append(x)
                    s = prev_s
                return T
            return None

    def improve(S):
        # Try to add elements back
        S_curr = set(S)
        base_reach = get_bitset(S_curr)
        
        # Sort candidates to try? Or random?
        # Trying all possible additions
        changed = True
        while changed:
            changed = False
            # Iterate over all numbers 1..100
            # To make it deterministic or random, we can shuffle the list of missing numbers
            missing = [x for x in range(1, 101) if x not in S_curr]
            random.shuffle(missing)
            
            for x in missing:
                new_reach = base_reach | (base_reach << x)
                if ((new_reach >> 200) & 1) == 0:
                    S_curr.add(x)
                    base_reach = new_reach
                    changed = True
        return S_curr

    # 1. Deterministic Greedies (from original)
    # Large to Small
    s = []
    reachable = 1
    for x in range(100, 0, -1):
        new_reach = reachable | (reachable << x)
        if not ((new_reach >> 200) & 1):
            s.append(x)
            reachable = new_reach
    update_best(set(s))
    
    # Small to Large
    s = []
    reachable = 1
    for x in range(1, 101):
        new_reach = reachable | (reachable << x)
        if not ((new_reach >> 200) & 1):
            s.append(x)
            reachable = new_reach
    update_best(set(s))
    
    # 2. Subtractive Greedy Trials
    # We will run multiple trials.
    # Each trial: Start with full set, remove elements from witnesses until valid.
    # Then improve.
    
    trials = 0
    while time.time() < deadline:
        trials += 1
        S = set(range(1, 101))
        
        # Subtractive Phase
        while True:
            T = find_witness(S)
            if T is None:
                break
            
            # Choose element to remove
            # Strategy: Random from T
            # Or maybe weighted?
            # Let's stick to random for diversity
            x_to_remove = random.choice(T)
            S.discard(x_to_remove)
            
            # Safety check: if S becomes empty, break (shouldn't happen for target 200 with N=100)
            if not S:
                break
        
        # Improvement Phase
        S_improved = improve(S)
        update_best(S_improved)
        
        # Optional: Try different removal strategies in subsequent trials?
        # We can alternate between random and max/min removal.
        # But random is a good baseline.
        
    return list(best_set)
