import sys

def construct():
    max_target = 128
    
    # Generate smooth numbers up to max_target
    smooth = []
    for n in range(1, max_target + 1):
        x = n
        while x % 2 == 0:
            x //= 2
        while x % 3 == 0:
            x //= 3
        while x % 5 == 0:
            x //= 5
        if x == 1:
            smooth.append(n)
            
    # Precompute divisibility: divs[a] is a set of indices b such that smooth[b] % smooth[a] == 0
    # Or simpler: a function is_divisible(a, b)
    
    def is_anticain(vals):
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                if vals[i] % vals[j] == 0 or vals[j] % vals[i] == 0:
                    return False
        return True

    # To speed up, we can precompute for each pair if they are comparable
    # But for small k, is_anticain is fast enough.
    
    results = []
    
    # For each target, find the minimal k and the corresponding antichain
    for target in range(1, max_target + 1):
        found = False
        # Try k from 1 to 30 (limit is 30, but likely < 5)
        for k in range(1, 31):
            # Use backtracking to find k distinct smooth numbers summing to target
            # that form an antichain.
            # We can use a recursive helper.
            # To avoid deep recursion overhead, we can use iterative deepening or simple recursion since k is small.
            
            # Optimization: The largest element in the antichain cannot be too large.
            # Also, the sum of the smallest k smooth numbers must be <= target.
            # Smallest smooth numbers: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12...
            # Sum of first k smooth numbers.
            # If sum(smooth[0:k]) > target, then no solution for this k? 
            # No, because we can pick larger numbers? No, if we pick larger numbers, sum increases.
            # So if sum of the k smallest smooth numbers > target, then no k distinct smooth numbers sum to target.
            min_sum_k = sum(smooth[:k])
            if min_sum_k > target:
                # If k=1 and target not smooth, min_sum_1=1. If target<1 impossible.
                # If k=2, min_sum_2=1+2=3. If target=2, no solution with k=2 distinct.
                # So we can break if min_sum_k > target? 
                # Wait, we need distinct positive integers.
                # Yes, the minimum sum of k distinct positive smooth numbers is the sum of the first k smooth numbers.
                # If that sum > target, it's impossible to form target with k distinct smooth numbers.
                # So we can break the k-loop? No, larger k means larger min sum.
                # So if min_sum_k > target, then min_sum_{k+1} > target too.
                # Thus we can break.
                break
            
            # Now search for a combination of k smooth numbers summing to target.
            # We can use a recursive function.
            # To prune, we can sort smooth numbers.
            # Let's use a helper function.
            
            def dfs(start_idx, remaining_sum, current_vals, k_left):
                if k_left == 0:
                    if remaining_sum == 0:
                        return list(current_vals)
                    else:
                        return None
                if remaining_sum < 0:
                    return None
                
                # Pruning: if the sum of the smallest k_left smooth numbers starting from start_idx 
                # is greater than remaining_sum, we can't fill the remaining slots.
                # Actually, we just need to pick from smooth[start_idx:].
                # The minimum possible sum we can add with k_left more terms is sum(smooth[start_idx : start_idx + k_left]).
                if start_idx + k_left <= len(smooth):
                    min_add = sum(smooth[start_idx:start_idx+k_left])
                    if min_add > remaining_sum:
                        return None
                
                # Also, if we pick the largest possible numbers, the max sum is sum(smooth[-k_left:]).
                # But we are summing to a specific target, so we just iterate.
                
                for i in range(start_idx, len(smooth)):
                    val = smooth[i]
                    if val > remaining_sum:
                        break # Since smooth is sorted, further values are larger
                    
                    # Check antichain constraint with existing values
                    valid = True
                    for v in current_vals:
                        if v % val == 0 or val % v == 0:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    current_vals.append(val)
                    res = dfs(i + 1, remaining_sum - val, current_vals, k_left - 1)
                    if res is not None:
                        return res
                    current_vals.pop()
                
                return None
            
            sol = dfs(0, target, [], k)
            if sol is not None:
                results.append(sol)
                found = True
                break
        
        if not found:
            # This should not happen for 1..128 with k up to 30
            # But just in case, we might need to handle it.
            # For now, raise an error or use a fallback.
            raise ValueError(f"No representation found for {target}")

    return results
