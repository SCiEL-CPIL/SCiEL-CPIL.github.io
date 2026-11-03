import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_set = []
    best_len = 0
    
    # Helper to check if we can add k to current set S with existing sums
    def can_add(S, sums, k):
        # Check sums of k with all elements in S
        # And k+k
        # If any sum is in sums, return False
        # Also need to check if k+k is in sums? Yes.
        # And if k+s is in sums for any s in S.
        # Note: sums contains all a+b for a,b in S.
        # We need to check if k+s is in sums for s in S.
        # And if k+k is in sums.
        # If not, we can add.
        # The new sums to add would be k+s for s in S and k+k.
        # We just need to check for conflicts.
        
        # Conflict if (k + s) is in sums for any s in S
        # OR if (k + k) is in sums
        # Note: sums is a set of integers.
        
        for s in S:
            if (k + s) in sums:
                return False, sums
        if (k + k) in sums:
            return False, sums
        
        # If we add k, what are the new sums?
        # New sums: {k+s for s in S} U {k+k}
        # We need to ensure these new sums don't conflict with each other?
        # No, if they were in sums, we would have returned False.
        # Do they conflict with each other?
        # k+s1 == k+s2 implies s1==s2, so distinct.
        # k+s == k+k implies s==k, but k not in S yet.
        # So new sums are distinct and not in old sums.
        # So we can safely add.
        
        new_sums = sums.copy()
        for s in S:
            new_sums.add(k + s)
        new_sums.add(k + k)
        
        return True, new_sums

    # Greedy with randomization
    # We will run multiple iterations of randomized greedy
    
    max_iters = 20000
    iter_count = 0
    
    while iter_count < max_iters and time.time() < deadline - 2.0:
        S = []
        sums = set()
        
        # Random order of candidates
        candidates = list(range(N))
        np.random.shuffle(candidates)
        
        for k in candidates:
            ok, new_sums = can_add(S, sums, k)
            if ok:
                S.append(k)
                sums = new_sums
                
        if len(S) > best_len:
            best_len = len(S)
            best_set = S
            # If we reach a theoretical max or high value, we can stop
            # Max Sidon size in [0, 49] is approx sqrt(50) ~ 7.
            # If we find 8, it's great.
            if best_len >= 8:
                # We can break or continue to see if 9 is possible
                # 9 is unlikely.
                pass
        iter_count += 1
        
    # Construct vector
    h = np.zeros(N)
    if best_len > 0:
        for idx in best_set:
            h[idx] = 1.0 / best_len
    else:
        h[0] = 1.0
        
    return list(h)
