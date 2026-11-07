import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 58

    def try_greedy_build(order):
        S = []
        n = 0
        # DP[k] is a set of sums
        dp = [set() for _ in range(1)] # Start with n=0, dp[0]={0}
        dp[0].add(0)
        
        # Union of all sums
        all_sums = {0}
        # Track which sums belong to which cardinality for fast lookup
        # sum_to_cardinalities: dict from sum value -> set of cardinalities that produce it
        sum_to_card = {0: {0}}
        
        for x in order:
            if x in S:
                continue
            
            # Check if x can be added
            valid = True
            
            # For each cardinality k from 1 to n+1, compute new sums
            # and check for conflicts
            for k in range(1, n + 2):
                if k - 1 >= len(dp):
                    continue
                prev_sums = dp[k-1]
                if not prev_sums:
                    continue
                
                # Get existing sums for cardinality k
                if k < len(dp):
                    existing_k = dp[k]
                else:
                    existing_k = set()
                
                # Check each new sum for conflict
                for s in prev_sums:
                    new_sum = s + x
                    # If new_sum is in all_sums, check if it's already in existing_k
                    if new_sum in all_sums:
                        if new_sum not in existing_k:
                            # Conflict: new_sum was produced by a different cardinality
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                # Commit x
                S.append(x)
                n += 1
                # Update dp in reverse order to avoid using updated values
                # We need to create new dp since sizes change
                old_dp = dp
                new_dp = [set() for _ in range(n + 1)]
                for k in range(n + 1):
                    if k < len(old_dp):
                        new_dp[k] = set(old_dp[k])
                    if k > 0 and k - 1 < len(old_dp):
                        for s in old_dp[k-1]:
                            new_dp[k].add(s + x)
                dp = new_dp
                
                # Update all_sums and sum_to_card
                for k in range(n + 1):
                    for s in dp[k]:
                        if s not in all_sums:
                            all_sums.add(s)
                            sum_to_card[s] = {k}
                        else:
                            sum_to_card[s].add(k)

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
    
    for o in orders:
        if time.time() > deadline:
            break
        S = try_greedy_build(o)
        if len(S) > len(best_set):
            best_set = S
            # If we found a very good set, we can tighten the deadline check
            if len(S) >= 15:
                deadline = start_time + 58
    
    return [int(x) for x in best_set]
