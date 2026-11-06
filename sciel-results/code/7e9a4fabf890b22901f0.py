import time
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin for 60s limit
    
    # 1. Initialize with a safe skeleton
    # Powers of 4 ensure unique sums for triples of skeleton elements
    # 4^0=1, 4^1=4, 4^2=16, 4^3=64, 4^4=256, 4^5=1024
    # 4^6=4096 is the limit, but usually we want room for insertion.
    # Let's include 1024. 4096 itself might be too restrictive or just a boundary.
    # If we use 1, 4, 16, 64, 256, 1024, 4096, that's 7 elements.
    # Let's stick to < 4096 to allow insertion near the top if needed, or just up to 4096.
    # The problem says <= 4096.
    
    S = [1, 4, 16, 64, 256, 1024]
    # Pre-compute sums for the skeleton
    sums = set()
    for t in combinations_with_replacement(S, 3):
        sums.add(sum(t))
        
    best_set = list(S)
    best_size = len(S)
    
    # Candidates to try
    # We can try all numbers from 1 to 4096
    # To be efficient, we might want to prioritize numbers in gaps or just go 1..4096
    # Going 1..4096 is simple and covers everything.
    
    candidates = list(range(1, 4097))
    
    # Optimization: Sort candidates? 
    # Usually, smaller numbers are "harder" to place if they conflict, 
    # but placing them early constrains the rest. 
    # Placing larger numbers first might leave more room for small ones?
    # The skeleton is already fixed. 
    # Let's try increasing order first.
    
    # We can run this greedy pass multiple times with different starting skeletons 
    # or candidate orders if time permits, but the skeleton is strong.
    # Let's just do one thorough pass.
    
    for c in candidates:
        if time.time() > deadline:
            break
            
        # If c is already in S, skip
        if c in S:
            continue
            
        # Check if c can be added
        # New sums involving c:
        # 1. 3*c
        # 2. 2*c + s (for s in S)
        # 3. c + 2*s (for s in S)
        # 4. c + s1 + s2 (for s1, s2 in S, s1 <= s2)
        
        new_sums = []
        
        # Type 1
        s1 = 3 * c
        new_sums.append(s1)
        
        # Type 2 & 3
        for s in S:
            new_sums.append(2 * c + s)
            new_sums.append(c + 2 * s)
            
        # Type 4
        # combinations_with_replacement(S, 2)
        for s1, s2 in combinations_with_replacement(S, 2):
            new_sums.append(c + s1 + s2)
            
        # Check for internal collisions in new_sums
        if len(new_sums) != len(set(new_sums)):
            continue
            
        # Check for collisions with existing sums
        # If any new sum is in existing sums, collision
        # We can do this efficiently by iterating
        valid = True
        for ns in new_sums:
            if ns in sums:
                valid = False
                break
                
        if valid:
            S.append(c)
            sums.update(new_sums)
            
    return S

# The above is a single pass. 
# Can we do better? 
# The "Knowledge" mentioned a score of 1.778. 
# If the seed gets ~10, and this gets ~18, that's a 1.8x improvement.
# Is 18 achievable? 
# $N^{1/3}$ for 4096 is 16. 
# With a good skeleton, we might push it to 17 or 18.
# 
# To squeeze more, we can try multiple random restarts or different skeleton bases.
# But powers of 4 are optimal for base-4 separation.
# What if we use a denser skeleton? No, base must be > 3 to avoid carries.
# So base 4 is the tightest.
#
# Let's add a second phase: 
# If time remains, try to find a better set using local search or different candidate orders.
# However, the greedy insertion on the skeleton is likely near-optimal for this constraint.
#
# One improvement: 
# The order of insertion matters. 
# If we insert 2, it might block 3, 5, etc.
# If we insert 3 first, it might block 2.
# Since we iterate 1..4096, we are biased towards small numbers.
# Small numbers have more "neighbors" in the sum space.
# Maybe iterating in a random order or larger-first order yields a larger set?
# Let's try a few different orders in the remaining time.
