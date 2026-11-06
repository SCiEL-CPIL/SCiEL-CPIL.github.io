import time
import itertools

def _is_valid_mean_free(selected, candidate):
    """Check if adding candidate to selected keeps the set mean-free.
    
    For each element a in selected ∪ {candidate}, check that no subset of
    at least 2 OTHER elements has sum = len(subset) * a.
    """
    full = selected + [candidate]
    n = len(full)
    
    for a in full:
        others = [x for x in full if x != a]
        m = len(others)
        
        # For each subset size k >= 2, check if any k-subset sums to k*a
        # Use DP with bitsets: dp[s] = set of achievable sums using up to s elements
        # But more efficiently: for each k, check if k*a is achievable by exactly k elements
        
        # Since m can be up to 47, we need an efficient check
        # For a given k, we check if there exists a k-subset of others summing to k*a
        # This is a subset-sum problem with exactly k elements
        
        # Use DP: dp[i][j] = set of sums achievable using j elements from first i of others
        # To save memory, use rolling array
        # But m is at most 47 and values up to 512, so max sum is 47*512 = 24064
        # That's too large for a full DP table per k per a
        
        # Alternative: for small k, enumerate combinations
        # For larger k, use meet-in-the-middle or just check via DP
        
        # Let's use a bitset approach per k:
        # bitset[j] is a Python int where bit s is set if sum s is achievable with j elements
        # Start: bitset[0] = 1 (sum 0 with 0 elements)
        # For each element e in others:
        #   for j from current_count down to 1:
        #     bitset[j] |= (bitset[j-1] << e)
        
        # Max sum we care about: for each k, we only need to check if k*a is achievable
        # k*a <= 47 * 512 = 24064, so we need bits up to 24064
        # Python ints can handle this efficiently
        
        target_max = max(k * a for k in range(2, m + 1)) if m >= 2 else 0
        mask = (1 << (target_max + 1)) - 1 if target_max >= 0 else 0
        
        bitsets = [0] * (m + 1)
        bitsets[0] = 1  # sum 0 with 0 elements
        
        count = 0
        for e in others:
            # Update in reverse to avoid using e more than once
            for j in range(count + 1, 0, -1):
                bitsets[j] |= (bitsets[j-1] << e) & mask
            count += 1
        
        for k in range(2, m + 1):
            target = k * a
            if target <= target_max and (bitsets[k] >> target) & 1:
                return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s safety margin
    
    selected = []
    
    for x in range(1, 513):
        if time.time() > deadline:
            break
        
        # Try adding x to selected
        if _is_valid_mean_free(selected, x):
            selected.append(x)
    
    return selected
