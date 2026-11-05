import time

def construct():
    target = 4979
    n = 5040
    
    # Generate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors, order doesn't strictly matter for 0/1 knapsack min count
    # but sorting might help with pruning if we were doing branch and bound.
    # For DP, we just iterate.
    
    # dp[s] = list of divisors summing to s with min count
    # Using a dict to only keep reachable states
    dp = {0: []}
    
    for d in divisors:
        # Iterate over a copy of keys to avoid updating current state
        # and using it again in the same step (0/1 property)
        current_states = list(dp.items())
        for s, lst in current_states:
            new_s = s + d
            if new_s <= target:
                new_lst = lst + [d]
                # Check if we have a better solution for new_s
                if new_s not in dp or len(new_lst) < len(dp[new_s]):
                    dp[new_s] = new_lst
                # If we found a solution for target, we can't stop immediately
                # because a later divisor might allow a shorter sum?
                # No, we are adding divisors one by one.
                # Wait, the DP formulation above is:
                # "For each divisor d, consider adding it to any previous sum."
                # This correctly handles 0/1 knapsack.
                # Does finding a solution for target early allow pruning?
                # Not really, because a different combination of remaining divisors
                # might yield a smaller count?
                # Actually, since we are minimizing count, and all divisors are positive,
                # if we find a solution with count K, we can prune any state 
                # that already has count >= K? No, because adding more divisors 
                # increases count.
                # However, if we have a state s with count c, and we have a target solution 
                # with count C, if c >= C, we don't need to extend s?
                # Yes, because adding any positive divisor increases count.
                # So we can prune states with count >= best_found.
                
                if new_s == target:
                    # We found a candidate. 
                    # We can't break the outer loop immediately because 
                    # maybe a different combination of *other* divisors 
                    # (not including d) yields a smaller count?
                    # No, the DP processes divisors one by one.
                    # The state `dp` accumulates the best solutions using the 
                    # first `i` divisors.
                    # So after processing all divisors, `dp[target]` is the global min.
                    # We can prune: if we have a solution for target with count C,
                    # any state with count >= C is useless for further extension 
                    # (since it can only get worse).
                    # But we still need to process other divisors to see if they 
                    # form a better sum? No, if a state has count >= C, 
                    # extending it will only increase count.
                    # So we can skip extending states with len(lst) >= current_best_target_count.
                    pass
        
    if target in dp:
        return dp[target]
    else:
        return []
