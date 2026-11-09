import time
import random

def construct():
    deadline = time.time() + 55.0  # Safety margin
    best_set = []
    
    def is_valid_addition(current_set, x):
        # Check if x can be added to current_set
        # 1. Check if x is mean of any subset of current_set (size >= 2)
        # 2. Check if any a in current_set is mean of subset of (current_set - {a}) + {x} (size >= 2)
        
        n = len(current_set)
        if n < 2:
            # If n < 2, condition "subset of at least two OTHER elements" is vacuously true for the new element?
            # If n=0, x is alone. No subset of size >= 2. Valid.
            # If n=1, x and a. Subsets of size >= 2 from "others" (which has size 1) don't exist. Valid.
            return True
            
        # Check 1: Is x the mean of any subset of current_set?
        # We need sum(subset) == k * x for some k >= 2.
        # DP on current_set to find reachable sums for each k.
        max_sum = sum(current_set)
        # We only care about sums up to max_sum.
        # dp[k] is a bitset of reachable sums.
        dp = [0] * (n + 1)
        dp[0] = 1 # sum 0 reachable with 0 elements
        
        for a in current_set:
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k+1] |= (dp[k] << a)
                    
        for k in range(2, n + 1):
            # Check if k*x is a reachable sum
            if k * x <= max_sum:
                if (dp[k] >> (k * x)) & 1:
                    return False
            else:
                break # Sums only increase with k if x is positive? No, k increases, target increases.
                # Actually if k*x > max_sum, larger k will also have larger targets, but max_sum is fixed.
                # So we can break.

        # Check 2: For each a in current_set, is a the mean of a subset of (current_set \ {a}) U {x}?
        # Let S_a = current_set \ {a}.
        # We need to check if a is the mean of a subset T of S_a U {x} with |T| >= 2.
        # T can contain x or not.
        # Case 2a: T does not contain x. Then T is a subset of S_a.
        #   This was already valid in current_set? 
        #   Wait, the condition is "No element ... mean of ... OTHER elements".
        #   In current_set, a was not the mean of any subset of current_set \ {a}.
        #   So Case 2a is already satisfied by the invariant of current_set.
        #   We only need to check subsets that INCLUDE x.
        
        # Case 2b: T contains x.
        #   T = {x} U T', where T' is a subset of S_a.
        #   |T| = 1 + |T'| >= 2 => |T'| >= 1.
        #   Mean condition: a = (x + sum(T')) / (1 + |T'|)
        #   => (1 + |T'|) * a = x + sum(T')
        #   => sum(T') - |T'| * a = (1 + |T'|) * a - x - |T'| * a ... wait.
        #   (1 + k) * a = x + sum(T')  where k = |T'|.
        #   sum(T') = (k + 1) * a - x.
        
        # So for each a in current_set:
        #   We need to know if there exists k >= 1 and a subset T' of S_a of size k
        #   such that sum(T') = (k + 1) * a - x.
        
        # This requires a DP for each a? That would be O(N^2) DPs.
        # N is small (<= 48). N^2 is ~2300.
        # Each DP is O(N * Sum). Sum ~ 25000.
        # 2300 * 48 * 25000 is too slow in Python.
        
        # Alternative:
        # We can just check the full set validity for the new set using a single optimized routine?
        # Or, we can precompute something?
        
        # Let's try a simpler approach: Just run the full validity check on the new set.
        # But optimize it.
        
        pass

    # The "Check 2" logic above is complex to optimize incrementally.
    # Given N is small, maybe a full check with bitsets is fast enough?
    
    def is_valid_set(S):
        n = len(S)
        if n < 3: return True
        
        # For each a in S, check if a is mean of subset of S \ {a}
        for i, a in enumerate(S):
            others = S[:i] + S[i+1:]
            m = len(others)
            if m < 2: continue
            
            # DP for others
            max_sum = sum(others)
            dp = [0] * (m + 1)
            dp[0] = 1
            for x in others:
                for k in range(m - 1, -1, -1):
                    if dp[k]:
                        dp[k+1] |= (dp[k] << x)
                        
            for k in range(2, m + 1):
                target = k * a
                if target > max_sum: break
                if (dp[k] >> target) & 1:
                    return False
        return True

    # Greedy with randomization
    # Try to find a set of size 48.
    # Strategy: Pick numbers from a pool.
    # Large numbers are less likely to be averages of smaller ones?
    # Actually, averages of large numbers are large.
    # If we pick all large numbers, their averages are large.
    # Example: 500, 501, 502. Mean of 500, 502 is 501. Conflict.
    # So we need to avoid arithmetic progressions.
    
    # Let's try a greedy approach with a specific order.
    # Order: Randomly shuffle 1..512, then try to add.
    # Or: Sort by some heuristic?
    
    # Since we have time, let's do a few random restarts.
    
    best_size = 0
    best_result = []
    
    # Pre-shuffled lists for different seeds
    for seed in range(10):
        if time.time() > deadline: break
        
        random.seed(seed)
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        current_set = []
        for x in candidates:
            if len(current_set) >= 48:
                break
            new_set = current_set + [x]
            if is_valid_set(new_set):
                current_set = new_set
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_result = current_set
            if best_size == 48:
                break
                
    return best_result
