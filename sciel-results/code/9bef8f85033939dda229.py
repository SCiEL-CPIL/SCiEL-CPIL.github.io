"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    best_set = []
    best_len = 0
    
    def can_add(S, sums, x):
        """
        Check if adding x to S preserves unique triple sums.
        S is a list of integers.
        sums is a set of all current triple sums.
        Returns True if valid, False otherwise.
        """
        new_sums = []
        
        # 1. (x, x, x)
        new_sums.append(3 * x)
        
        # 2. (x, x, s) for s in S
        for s in S:
            new_sums.append(2 * x + s)
            
        # 3. (x, s1, s2) for s1, s2 in S, s1 <= s2
        n = len(S)
        for i in range(n):
            s1 = S[i]
            for j in range(i, n):
                s2 = S[j]
                new_sums.append(x + s1 + s2)
                
        # Check for duplicates among new_sums
        if len(new_sums) != len(set(new_sums)):
            return False
            
        # Check if any new_sum is already in sums
        for ns in new_sums:
            if ns in sums:
                return False
                
        return True

    def greedy_search(skip_prob):
        S = []
        sums = set()
        
        # Iterate through all candidates
        for x in range(1, 4097):
            # Check time
            if time.time() > deadline:
                return S, sums
            
            # Random skip
            if random.random() < skip_prob:
                continue
                
            if can_add(S, sums, x):
                S.append(x)
                # Update sums
                # We need to add all new sums to the set
                # 1. 3*x
                sums.add(3 * x)
                # 2. 2*x + s
                for s in S[:-1]: # S[:-1] is S before adding x? No, S now has x.
                    # Wait, S was appended. So S[:-1] is the old S.
                    sums.add(2 * x + s)
                # 3. x + s1 + s2
                old_S = S[:-1]
                n = len(old_S)
                for i in range(n):
                    s1 = old_S[i]
                    for j in range(i, n):
                        s2 = old_S[j]
                        sums.add(x + s1 + s2)
                        
        return S, sums

    # Run standard greedy first
    S, sums = greedy_search(0.0)
    if len(S) > best_len:
        best_len = len(S)
        best_set = S
        
    # Run randomized greeds
    # We can vary the skip probability
    # Also, we can try different random seeds implicitly by just calling random
    
    # How many restarts?
    # Each restart takes roughly (4096 * avg_len^2) operations.
    # If avg_len is 30, 4096 * 900 ~ 3.6M ops.
    # Python can do ~10M-50M simple ops/sec.
    # So ~0.1s per restart.
    # We have 55s. So ~500 restarts possible.
    
    while time.time() < deadline - 2.0: # 2s margin for final return
        # Random skip probability between 0.0 and 0.2
        # Higher skip prob might allow "jumping" over bad local minima?
        # Or maybe we should try to bias towards larger numbers?
        # Let's stick to uniform skip for now.
        p = random.uniform(0.0, 0.15)
        S, sums = greedy_search(p)
        if len(S) > best_len:
            best_len = len(S)
            best_set = S
            # If we find a very good one, we might want to refine it?
            # But greedy is already fairly good.
            
    return best_set
# EVOLVE-BLOCK-END
