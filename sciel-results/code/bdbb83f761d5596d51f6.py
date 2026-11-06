import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 4096
    best_set = []
    best_len = 0
    
    # Helper to check if x can be added to current set S with current sums
    # Returns True if valid, and if so, the list of new sums to add
    def check_and_get_new_sums(S, sums, x):
        new_sums = []
        n = len(S)
        
        # 1. 3x
        s_3x = 3 * x
        if s_3x in sums:
            return False, None
        new_sums.append(s_3x)
        
        # 2. 2x + s_i
        for s_i in S:
            s_val = 2 * x + s_i
            if s_val in sums or s_val in new_sums: # Check against existing and new
                # Note: new_sums is a list, so 'in' is O(k). 
                # For efficiency, we can use a set for new_sums too, but k is small.
                # However, we must also ensure new_sums elements are unique among themselves.
                # Since we check against 'sums' (existing) and we just appended, 
                # we need to check if s_val was already in new_sums.
                # Let's use a set for new_sums to make checks O(1).
                pass 
        
        # Let's optimize: use a set for new_sums
        new_sums_set = set()
        
        # 1. 3x
        s_3x = 3 * x
        if s_3x in sums:
            return False, None
        new_sums_set.add(s_3x)
        
        # 2. 2x + s_i
        for s_i in S:
            s_val = 2 * x + s_i
            if s_val in sums:
                return False, None
            if s_val in new_sums_set:
                return False, None
            new_sums_set.add(s_val)
            
        # 3. x + s_i + s_j (i <= j)
        for i in range(n):
            s_i = S[i]
            for j in range(i, n):
                s_j = S[j]
                s_val = x + s_i + s_j
                if s_val in sums:
                    return False, None
                if s_val in new_sums_set:
                    return False, None
                new_sums_set.add(s_val)
                
        return True, new_sums_set

    def run_greedy(order):
        S = []
        sums = set()
        
        for x in order:
            valid, new_sums = check_and_get_new_sums(S, sums, x)
            if valid:
                S.append(x)
                sums.update(new_sums)
                
        return S

    # Strategy 1: Pure Greedy with Natural Order
    # This is fast and gives a baseline.
    S_init = run_greedy(range(1, N + 1))
    if len(S_init) > best_len:
        best_set = S_init
        best_len = len(S_init)
        
    # Strategy 2: Randomized Greedy with Multiple Restarts
    # We try to find a larger set by shuffling the order.
    # Since the constraint is global, the order matters.
    # Larger numbers tend to have fewer collisions with small numbers? 
    # Actually, small numbers collide with each other more often in sums?
    # Let's try different random seeds and biases.
    
    random.seed(42)
    
    # Time budget for randomized search
    # Each greedy pass is O(N * K^2) where K is current size. 
    # N=4096, K~30-50. 4096 * 2500 ~ 10^7 operations. 
    # Python might take 1-2 seconds per pass.
    # We have ~50 seconds. So we can do ~20-30 passes.
    
    while time.time() < deadline - 5.0:
        # Create a random order
        # Sometimes biasing towards larger numbers helps avoid early collisions?
        # Or just uniform random.
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S_try = run_greedy(order)
        
        if len(S_try) > best_len:
            best_set = S_try
            best_len = len(S_try)
            
    # Strategy 3: Local Improvement on Best Set
    # Try to add more elements to the best set found so far.
    if best_len > 0:
        S_cur = list(best_set)
        sums_cur = set()
        # Recompute sums for current best set
        n = len(S_cur)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    sums_cur.add(S_cur[i] + S_cur[j] + S_cur[k])
                    
        S_cur_set = set(S_cur)
        
        # Try to add any remaining number
        for x in range(1, N + 1):
            if time.time() > deadline - 2.0:
                break
            if x in S_cur_set:
                continue
            valid, new_sums = check_and_get_new_sums(S_cur, sums_cur, x)
            if valid:
                S_cur.append(x)
                S_cur_set.add(x)
                sums_cur.update(new_sums)
                if len(S_cur) > best_len:
                    best_set = S_cur
                    best_len = len(S_cur)
                    
    return best_set
