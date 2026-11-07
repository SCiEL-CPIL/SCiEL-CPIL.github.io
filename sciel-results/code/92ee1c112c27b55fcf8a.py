import time
import random

def construct():
        start_time = time.time()
        deadline = start_time + 55.0  # Safety margin
        
        max_val = 256
        max_size = 60
        # Max sum is 60 * 256 = 15360. Bitset size is small.
        
        def try_order(order):
            """
            Greedy construction with pre-check collision detection.
            """
            # dp[k] is a bitset of sums reachable with k elements
            dp = [0] * (max_size + 1)
            dp[0] = 1  # Sum 0
            
            S = []
            
            for x in order:
                if len(S) >= max_size:
                    break
                if time.time() > deadline:
                    break
                
                # Check if x can be added
                # For each k from len(S) down to 0:
                # New sums for k+1 are (dp[k] << x)
                # These must not intersect with dp[j] for any j != k+1
                # Note: dp[j] for j > len(S) is 0, so we only check j in [0, len(S)]
                
                valid = True
                # We need to check against the *current* state of dp, 
                # because adding x to k affects k+1, which might collide with existing k+1? 
                # No, collision with same k+1 is allowed.
                # Collision with other j is forbidden.
                
                # Optimization: 
                # The new sums are disjoint from each other for different k? 
                # No, we are adding x to all existing sums.
                # The sets (dp[k] << x) for different k might overlap?
                # If they overlap, that's fine (same cardinality k+1? No, different k).
                # Wait.
                # If we add x, the new sums for cardinality k+1 come from old cardinality k.
                # The new sums for cardinality k+2 come from old cardinality k+1.
                # These are distinct cardinalities.
                # So we must check (dp[k] << x) against dp[j] for all j != k+1.
                
                # Also, do the new sums for k+1 collide with new sums for k+2?
                # No, they are in different dp buckets.
                
                # So the check is:
                for k in range(len(S), -1, -1):
                    if dp[k] == 0:
                        continue
                    
                    new_sums = dp[k] << x
                    target_k = k + 1
                    
                    # Check collision with all existing buckets j != target_k
                    # j ranges from 0 to len(S) (since dp[j] is 0 for j > len(S))
                    for j in range(len(S) + 1):
                        if j == target_k:
                            continue
                        if dp[j] != 0 and (new_sums & dp[j]):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Update dp in reverse order
                    for k in range(len(S), -1, -1):
                        if dp[k]:
                            dp[k + 1] |= (dp[k] << x)
                    S.append(x)
            
            return S

        best_set = []
        
        # Strategy 1: Descending
        res = try_order(range(max_val, 0, -1))
        if len(res) > len(best_set):
            best_set = res
            
        # Strategy 2: Ascending
        if time.time() < deadline:
            res = try_order(range(1, max_val + 1))
            if len(res) > len(best_set):
                best_set = res

        # Strategy 3: Randomized Descending (Shuffle top 50%, keep bottom 50% sorted desc?)
        # Or just full shuffle with bias
        for attempt in range(100):
            if time.time() > deadline - 2.0:
                break
            
            random.seed(attempt * 999 + 1)
            # Bias towards larger numbers
            candidates = list(range(1, max_val + 1))
            # Shuffle, but maybe sort by value with some noise?
            # Simple shuffle is okay, but descending is stronger.
            # Let's try: 80% chance pick from top half, 20% from bottom?
            # Easier: Shuffle, then sort by a key that is value - random_noise
            # Or just shuffle.
            random.shuffle(candidates)
            
            res = try_order(candidates)
            if len(res) > len(best_set):
                best_set = res
        
        return [int(x) for x in best_set]
