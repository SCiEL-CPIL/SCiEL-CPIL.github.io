import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 58

    def try_greedy_build(order):
        S = []
        n = 0
        # dp[k] is a bitset (Python int) where bit s is set iff sum s is achievable
        # by a subset of cardinality k from S.
        # Max possible sum: 60 * 256 = 15360. We use arbitrary precision ints.
        dp = [0] * 61
        dp[0] = 1  # bit 0 set: sum 0 with cardinality 0
        
        # all_sums is a bitset of all sums achievable by any subset of S.
        all_sums = 1  # bit 0 set
        
        for x in order:
            if x in S:
                continue
            
            # Check if x can be added
            # New dp': dp'[k] = dp[k] | (dp[k-1] << x)
            # Condition: for all k != j, dp'[k] & dp'[j] == 0
            # Since dp[k] & dp[j] == 0 for k != j (invariant), we need:
            # (dp[k] | (dp[k-1] << x)) & (dp[j] | (dp[j-1] << x)) == 0 for k != j
            # This expands to:
            # dp[k] & dp[j] == 0 (true)
            # dp[k] & (dp[j-1] << x) == 0
            # (dp[k-1] << x) & dp[j] == 0
            # (dp[k-1] << x) & (dp[j-1] << x) == 0  => dp[k-1] & dp[j-1] == 0 (true if k-1 != j-1)
            
            # So we need to check for all k, j with k != j:
            # 1. dp[k] & (dp[j-1] << x) == 0
            # 2. (dp[k-1] << x) & dp[j] == 0
            
            # A simpler sufficient and necessary condition derived from the invariant:
            # Let new_sums_k = dp[k-1] << x.
            # We need new_sums_k to be disjoint from (all_sums \ dp[k]).
            # Because dp[k] already contains valid sums for cardinality k.
            # If new_sums_k has any bit set that is in all_sums but not in dp[k], it's a conflict.
            
            valid = True
            for k in range(1, n + 2):
                if k - 1 >= len(dp):
                    continue
                ns = dp[k-1] << x
                # Check conflict
                if ns & (all_sums & ~dp[k]):
                    valid = False
                    break
            
            if valid:
                # Commit x
                S.append(x)
                n += 1
                
                # Update dp
                # dp[k] = dp[k] | (dp[k-1] << x)
                # Must do in reverse order to avoid using updated dp[k-1]
                # Or use a copy.
                # Since n is small, copy is fine.
                old_dp = dp[:]
                for k in range(n, 0, -1):
                    dp[k] = old_dp[k] | (old_dp[k-1] << x)
                # dp[0] remains 1
                
                # Update all_sums
                # all_sums = OR of all dp[k]
                # Or all_sums |= (old_all_sums << x)? No, that's not right.
                # all_sums' = all_sums | (OR of new_sums)
                # new_sums = (OR of old_dp) << x = all_sums << x
                # So all_sums |= (all_sums << x)
                all_sums |= (all_sums << x)
                
        return S

    def local_improve(S):
        """Try to add more elements to S by trying all remaining elements."""
        if not S:
            return S
        
        S = list(S)
        n = len(S)
        
        # Rebuild dp from S
        dp = [0] * 61
        dp[0] = 1
        all_sums = 1
        
        for x in S:
            old_dp = dp[:]
            for k in range(n, 0, -1):
                dp[k] = old_dp[k] | (old_dp[k-1] << x)
            all_sums |= (all_sums << x)
        
        # Try adding each remaining element
        in_S = set(S)
        for x in range(1, 257):
            if x in in_S:
                continue
            
            valid = True
            for k in range(1, n + 2):
                if k - 1 >= len(dp):
                    continue
                ns = dp[k-1] << x
                if ns & (all_sums & ~dp[k]):
                    valid = False
                    break
            
            if valid:
                S.append(x)
                n += 1
                in_S.add(x)
                old_dp = dp[:]
                for k in range(n, 0, -1):
                    dp[k] = old_dp[k] | (old_dp[k-1] << x)
                all_sums |= (all_sums << x)
        
        return S

    best_set = []
    
    # Generate diverse orders
    orders = []
    
    # 1. Sorted ascending and descending
    orders.append(list(range(1, 257)))
    orders.append(list(range(256, 0, -1)))
    
    # 2. Alternating small/large
    small = list(range(1, 129))
    large = list(range(256, 127, -1))
    alt = []
    for i in range(128):
        alt.append(small[i])
        alt.append(large[i])
    orders.append(alt)
    
    # 3. Random with different seeds
    rng = random.Random(42)
    for trial in range(500):
        if time.time() > deadline - 2:
            break
        o = list(range(1, 257))
        rng.shuffle(o)
        orders.append(o)
    
    # 4. Biased random: weight towards certain ranges
    for trial in range(200):
        if time.time() > deadline - 2:
            break
        o = list(range(1, 257))
        # Create a biased order: pick from ranges with different probabilities
        biased = []
        for _ in range(256):
            r = rng.random()
            if r < 0.3:
                biased.append(rng.randint(1, 64))
            elif r < 0.6:
                biased.append(rng.randint(65, 128))
            elif r < 0.9:
                biased.append(rng.randint(129, 192))
            else:
                biased.append(rng.randint(193, 256))
        # Remove duplicates, keep unique
        seen = set()
        unique_biased = []
        for v in biased:
            if v not in seen:
                seen.add(v)
                unique_biased.append(v)
        # Add any missing values
        for v in range(1, 257):
            if v not in seen:
                unique_biased.append(v)
        orders.append(unique_biased)
        
    # 5. Spaced orders (e.g., multiples of 4, then 3, etc.)
    # This might help with sum separation.
    for step in [2, 3, 4, 5, 8]:
        o = []
        for i in range(1, 257, step):
            o.append(i)
        # Fill in the rest
        for v in range(1, 257):
            if v not in o:
                o.append(v)
        orders.append(o)
        
    # 6. Descending spaced
    for step in [2, 3, 4, 5, 8]:
        o = []
        for i in range(256, 0, -step):
            o.append(i)
        for v in range(256, 0, -1):
            if v not in o:
                o.append(v)
        orders.append(o)

    for o in orders:
        if time.time() > deadline:
            break
        S = try_greedy_build(o)
        if len(S) > len(best_set):
            best_set = S
            # If we found a very good set, we can tighten the deadline check
            if len(S) >= 15:
                deadline = start_time + 58
    
    # Local improvement pass
    if best_set:
        improved = local_improve(best_set)
        if len(improved) > len(best_set):
            best_set = improved

    return [int(x) for x in best_set]
