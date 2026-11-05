import time
import random

def construct():
    # Deadline guard: 55 seconds to leave margin for GC/overhead
    deadline = time.time() + 55.0
    best_set = []
    
    def is_valid(S):
        n = len(S)
        if n == 0:
            return True
        
        # For each x in S, check if x is the arithmetic mean of any subset of S\{x} of size >= 2
        for i in range(n):
            x = S[i]
            
            # We need to check subsets of S\{x}
            # Let A = S\{x}
            # We use bitset DP: bits[k] has bit s set if sum s is achievable with k elements from A
            # Max sum is bounded by 512 * 48 = 24576
            # We initialize bits[0] = 1 (sum 0 with 0 elements)
            
            bits = [0] * n  # bits[0] to bits[n-1]
            bits[0] = 1
            
            count = 0
            # Iterate through all elements except x
            for j in range(n):
                if j == i:
                    continue
                v = S[j]
                count += 1
                # Update bits in reverse to ensure each element is used at most once
                # Max k for current count is count
                for k in range(count, 0, -1):
                    bits[k] |= (bits[k-1] << v)
            
            # Check if x is the mean of any subset of size k >= 2
            # Condition: sum(subset) = k * x
            # Max possible sum for k elements is limited, but we just check the bit
            # Min k is 2, Max k is count
            for k in range(2, count + 1):
                target_sum = k * x
                # If target_sum is too large, the bit won't be set, but we can break early if we know max sum
                # Max sum of any k elements in S is at most k * 512
                # Min sum is at least k * 1
                # We just check the bit
                if (bits[k] >> target_sum) & 1:
                    return False
        
        return True

    # 1. Try deterministic heuristics
    heuristics = [
        list(range(1, 513)),       # Ascending
        list(range(512, 0, -1)),   # Descending
        list(range(1, 513, 2)),    # Odds
        list(range(2, 513, 2)),    # Evens
        list(range(512, 0, -2)),   # Descending Evens
    ]
    
    for h in heuristics:
        if time.time() > deadline:
            break
        S = []
        for x in h:
            if len(S) >= 48:
                break
            if is_valid(S + [x]):
                S.append(x)
        
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) == 48:
                return best_set

    # 2. Randomized Greedy Search
    # We can afford many trials because is_valid is fast (bitset DP)
    while time.time() < deadline:
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        S = []
        for x in candidates:
            if len(S) >= 48:
                break
            # Check validity before adding
            # Optimization: Check if x is mean of S, and if adding x breaks S elements
            # is_valid(S + [x]) does both.
            if is_valid(S + [x]):
                S.append(x)
        
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) == 48:
                return best_set
        
        # Optional: Local Search / Repair
        # If we didn't reach 48, try to swap elements to allow more additions
        # This is complex, so we rely on multiple random restarts.
        # Given the speed, 1000s of restarts are possible.

    return best_set
