import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55

    def is_valid(S):
        """Check if any subset of S sums to 200. S is a list/set of distinct values in [1,100]."""
        if not S:
            return True
        S = sorted(S)
        # If min element > 100, impossible but we're bounded
        # Quick checks for small subset sizes when min is large
        min_val = S[0]
        max_subset_size = min(len(S), 200 // min_val + 1) if min_val > 0 else len(S)
        
        # Use DP for subset sum up to 200
        # dp[s] = True if sum s is achievable
        dp = [False] * 201
        dp[0] = True
        for x in S:
            # Process in reverse to avoid using same element twice
            for s in range(200, x - 1, -1):
                if dp[s - x]:
                    dp[s] = True
        return not dp[200]

    # Base valid solution: {66, 67, ..., 100} size 35
    base = list(range(66, 101))
    best_set = base
    best_size = 35

    # Verify base
    if not is_valid(base):
        # Fallback: {67,...,100}
        base = list(range(67, 101))
        best_set = base
        best_size = 34

    # Try to find size 36
    # Strategy: For each candidate to remove from base and two to add from [1,65]
    # Or: try adding one element to a modified base
    # More efficient: try all single additions to subsets of base of size 35
    
    # Actually, let's try: for each x in 1..65, check if {66..100} minus some elements plus x works
    # Simpler: for each pair (remove y from base, add x from 1..65), check validity
    
    smalls = list(range(1, 66))
    
    # First try: can we just add one element to base and remove one?
    for x in smalls:
        if time.time() > deadline:
            break
        # Try removing each y from base and adding x
        for y in base:
            if time.time() > deadline:
                break
            cand = set(base)
            cand.discard(y)
            cand.add(x)
            if is_valid(list(cand)):
                best_size = 35  # Still 35, same size
                # Try to find size 36
                # Now try to add another small element
                for x2 in smalls:
                    if x2 == x:
                        continue
                    if time.time() > deadline:
                        break
                    cand2 = set(cand)
                    cand2.add(x2)
                    if is_valid(list(cand2)):
                        best_size = 36
                        best_set = list(cand2)
                        # Try size 37
                        for x3 in smalls:
                            if x3 == x or x3 == x2:
                                continue
                            if time.time() > deadline:
                                break
                            cand3 = set(cand2)
                            cand3.add(x3)
                            if is_valid(list(cand3)):
                                best_size = 37
                                best_set = list(cand3)
                                return best_set
                        return best_set
    
    # Random search for size 36+
    for _ in range(2000):
        if time.time() > deadline:
            break
        # Generate random set biased toward large numbers
        cand_set = set()
        # Always include many large numbers
        for i in range(70, 101):
            cand_set.add(i)
        # Add some from 1-69
        extra_needed = 36 - len(cand_set)
        if extra_needed > 0:
            pool = [i for i in range(1, 70) if i not in cand_set]
            random.shuffle(pool)
            cand_set.update(pool[:extra_needed])
        
        if len(cand_set) >= 36 and is_valid(list(cand_set)):
            if len(cand_set) > best_size:
                best_size = len(cand_set)
                best_set = list(cand_set)
    
    return best_set
