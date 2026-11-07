import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    max_val = 256
    max_size = 60
    
    # dp[k] is a bitset of sums reachable with k elements
    # We use Python ints for bitsets
    
    def check_and_add(dp, S, x):
        """
        Check if x can be added to S given current dp.
        Returns (valid, new_dp, new_S)
        """
        # We need to check collision for each k
        # new_sums for k+1 is (dp[k] << x)
        # This must not intersect with dp[j] for j != k+1
        
        # Optimization: Pre-compute the union of all dp[j] for j != target?
        # No, target depends on k.
        
        # Let's just loop.
        # Note: dp[k] for k > len(S) is 0.
        # We only need to check k from 0 to len(S).
        
        new_dp = dp[:]  # Shallow copy, but we will modify in place or create new?
        # Actually, we can just check validity first, then update.
        
        valid = True
        # We need to check if (dp[k] << x) intersects with any dp[j] where j != k+1.
        # Let's pre-compute the "forbidden" bits for each target cardinality k+1.
        # Forbidden for target t is OR of dp[j] for all j != t.
        
        # Since len(S) is small (<= 60), we can compute this on the fly or pre-compute.
        # Pre-computing `forbidden[t]` for t in 0..len(S)+1 might be useful.
        # forbidden[t] = (OR of all dp[j] for j in 0..len(S)+1) ^ dp[t]
        # Wait, dp[t] might be 0.
        
        # Let's compute `all_sums` = OR of all dp[j].
        # Then for target t, forbidden is `all_sums & ~dp[t]`.
        # But we are adding to dp[t] (which becomes dp[t] | (dp[t-1] << x)).
        # The new sums are `dp[t-1] << x`.
        # These new sums must not intersect with `all_sums & ~dp[t]`?
        # No, they must not intersect with `dp[j]` for any `j != t`.
        # So yes, `new_sums & (all_sums & ~dp[t])` must be 0.
        # Note: `dp[t]` contains existing sums of cardinality t. New sums of cardinality t are allowed to collide with existing sums of cardinality t.
        
        all_sums = 0
        for j in range(len(S) + 1):
            all_sums |= dp[j]
        
        # Check each k
        for k in range(len(S), -1, -1):
            if dp[k] == 0:
                continue
            new_sums = dp[k] << x
            target_t = k + 1
            # Forbidden region is all_sums excluding dp[target_t]
            # But wait, if target_t > len(S), dp[target_t] is 0.
            # So forbidden is all_sums if target_t > len(S), else all_sums ^ dp[target_t] (assuming no overlap between dp buckets? No, they can overlap in value, but they are distinct buckets. The bitset `all_sums` is the union of values. `all_sums & ~dp[target_t]` removes the bits that are already in bucket target_t. Any bit in `new_sums` that is in `all_sums` but NOT in `dp[target_t]` is a collision with another bucket.)
            
            forbidden = all_sums
            if target_t <= len(S):
                forbidden ^= dp[target_t] # Remove bits that are in the target bucket
            
            if new_sums & forbidden:
                valid = False
                break
        
        if not valid:
            return False, None, None
        
        # Update dp
        # We must update in reverse order to avoid using the new values in the same step
        new_dp = dp[:]
        for k in range(len(S), -1, -1):
            if dp[k]:
                new_dp[k + 1] |= (dp[k] << x)
                
        new_S = S + [x]
        return True, new_dp, new_S

    best_set = []
    best_len = 0
    
    # DFS with Backtracking
    # We will try to build a set.
    # Heuristic: Try candidates in descending order.
    # Candidates: 1..256
    
    # To avoid deep recursion stack overflow or slow recursion, 
    # we can use an iterative approach or just increase recursion limit.
    sys.setrecursionlimit(10000)
    
    def dfs(current_S, current_dp, remaining_candidates):
        nonlocal best_set, best_len, deadline
        
        if time.time() > deadline:
            return
        
        if len(current_S) > best_len:
            best_set = current_S[:]
            best_len = len(current_S)
            
        if len(current_S) == max_size:
            return
            
        # Heuristic: Sort remaining candidates?
        # We want to pick "good" candidates.
        # Larger numbers tend to separate sums better?
        # Let's try descending order from the remaining set.
        
        # Optimization: If we have many candidates, checking all is expensive.
        # We can limit the branching factor?
        # Or just iterate through them.
        
        # To speed up, we can filter candidates that are "obviously" bad?
        # Hard to say.
        
        # Let's iterate through candidates in descending order.
        # We can't just sort `remaining_candidates` every time (O(N log N)).
        # We can pass a sorted list and a set of used?
        # Or just iterate 1..256 and check if in current_S?
        # That's O(256) per node. With depth 60, that's manageable if branching is low.
        
        # But branching can be high.
        # Let's try a "Greedy with Backtracking" where we only backtrack a few levels deep?
        # Or use a beam search?
        
        # Let's try simple DFS with descending order.
        # We need to be careful about time.
        
        # Generate list of candidates not in current_S
        # Since max_val is 256, we can just loop 256..1
        used_set = set(current_S)
        
        # We can prune if the number of remaining candidates is less than (max_size - len(current_S))
        # But we don't know if we can add them all.
        
        for x in range(max_val, 0, -1):
            if x in used_set:
                continue
                
            # Check validity
            valid, new_dp, new_S = check_and_add(current_dp, current_S, x)
            if valid:
                dfs(new_S, new_dp, remaining_candidates) # remaining_candidates not really used if we loop 1..256
                
                if time.time() > deadline:
                    return
                if best_len == max_size:
                    return

    # Initial call
    # Start with empty set
    init_dp = [0] * (max_size + 1)
    init_dp[0] = 1
    dfs([], init_dp, None)
    
    return [int(x) for x in best_set]
