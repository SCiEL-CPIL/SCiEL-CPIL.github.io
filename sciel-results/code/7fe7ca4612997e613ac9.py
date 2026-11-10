import time
import random

def check_validity(S):
    """
    Checks if no element in S is the arithmetic mean of any subset 
    of at least two OTHER elements in S.
    Returns True if valid, False otherwise.
    """
    n = len(S)
    if n < 3:
        return True
        
    for a in S:
        others = [x for x in S if x != a]
        m = len(others)
        
        # Max possible sum is m * max(others)
        # We need to check subsets of size k >= 2
        # dp[k] is a bitset where bit j is 1 if sum j is achievable with k elements
        
        # Optimization: We can stop early if we find a violation
        # Also, we only care about sums up to k * a.
        # Max k is m. Max sum to check is m * a.
        
        # Bitset DP
        # dp[k] is an integer representing a bitset of sums.
        # We need to be careful with the size.
        # Max sum of 'others' is sum(others).
        # But we only check if sum == k * a.
        
        # Let's use a list of ints for bitsets.
        # dp[0] = 1 (bit 0 set)
        dp = [0] * (m + 1)
        dp[0] = 1
        
        current_max_k = 0
        for x in others:
            # Iterate backwards to avoid using the same element twice
            for k in range(current_max_k, -1, -1):
                if dp[k] != 0:
                    # Shift and OR
                    dp[k+1] |= (dp[k] << x)
            current_max_k += 1
            
        # Check for violations
        # For each k from 2 to m, check if sum = k * a is in dp[k]
        for k in range(2, m + 1):
            target_sum = k * a
            if dp[k] & (1 << target_sum):
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    best_set = []
    
    # Heuristics for ordering
    # 1. Ascending
    # 2. Descending
    # 3. Random (multiple times)
    # 4. "Spread out" (e.g., step by 2, then fill)
    
    orderings = []
    
    # Ascending
    orderings.append(list(range(1, 513)))
    
    # Descending
    orderings.append(list(range(512, 0, -1)))
    
    # Random orderings
    random.seed(42)
    for _ in range(5):
        order = list(range(1, 513))
        random.shuffle(order)
        orderings.append(order)
        
    # "Sparse" start: Powers of 2, then others
    powers = [2**i for i in range(1, 10) if 2**i <= 512]
    rest = [x for x in range(1, 513) if x not in powers]
    orderings.append(powers + rest)
    
    for order in orderings:
        if time.time() > deadline:
            break
            
        current_set = []
        for x in order:
            if time.time() > deadline:
                break
                
            # Check if adding x is valid
            # We can do a fast check:
            # 1. Is x the mean of a subset of current_set?
            # 2. Does x cause any a in current_set to become a mean?
            
            # For simplicity and robustness, use full check on current_set + [x]
            # But that's O(N^2) per step. 
            # With N=512, and set size ~50, it's fast.
            
            candidate_set = current_set + [x]
            if check_validity(candidate_set):
                current_set = candidate_set
        
        if len(current_set) > len(best_set):
            best_set = current_set
            
    return best_set
