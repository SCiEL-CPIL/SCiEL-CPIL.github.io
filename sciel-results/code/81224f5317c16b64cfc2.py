import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    max_val = 256
    max_size = 60
    
    # dp[k] is a bitset of sums
    # union_dp is the bitset of all sums across all cardinalities
    
    def is_valid_addition(dp, union_dp, x, n_current):
        # Check if adding x to a set of size n_current is valid
        # New sums for size k are N_k = dp[k-1] << x
        # We need N_k to be disjoint from dp[m] for all m != k.
        # Since dp[m] are pairwise disjoint (valid state), union_dp is their union.
        # The forbidden region for N_k is union_dp \ dp[k].
        # So we check if (N_k & union_dp) has any bits not in dp[k].
        
        n_new = n_current + 1
        for k in range(1, n_new + 1):
            n_k = dp[k-1] << x
            if n_k == 0:
                continue
            # Check against existing sums of different cardinalities
            # Intersection with union_dp gives all collisions with ANY existing sum.
            # We allow collisions with dp[k] (same cardinality).
            # So we mask out dp[k] from the intersection and see if anything remains.
            collisions = n_k & union_dp
            # If there are collisions, they must all be in dp[k]
            # If (collisions & ~dp[k]) is non-zero, there is a collision with a different cardinality.
            bad = collisions & ~dp[k]
            if bad != 0:
                return False
        return True

    best_size = 0
    best_set = []
    
    # We will try a few strategies
    # 1. Greedy increasing
    # 2. Greedy decreasing
    # 3. Randomized greedy with optimized check
    
    for attempt in range(200):  # More attempts due to speedup
        if time.time() > deadline:
            break
        
        if attempt % 3 == 0:
            candidates = list(range(1, max_val + 1))
        elif attempt % 3 == 1:
            candidates = list(range(max_val, 0, -1))
        else:
            candidates = list(range(1, max_val + 1))
            random.shuffle(candidates)
            
        dp = [0] * (max_size + 1)
        dp[0] = 1
        union_dp = 1
        current_set = []
        used = [False] * (max_val + 1)
        
        for c in candidates:
            if len(current_set) >= max_size:
                break
            if used[c]:
                continue
            
            if is_valid_addition(dp, union_dp, c, len(current_set)):
                # Add c
                current_set.append(c)
                used[c] = True
                # Update dp
                new_union = union_dp
                for k in range(len(current_set), 0, -1):
                    new_sums = dp[k-1] << c
                    dp[k] |= new_sums
                    new_union |= new_sums
                union_dp = new_union
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
        if best_size == max_size:
            break
            
    return best_set
