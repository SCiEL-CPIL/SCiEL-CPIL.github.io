# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 80  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Get all divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            if d <= target:
                divisors.append(d)
    
    # Sort divisors descending for better pruning in backtracking
    divisors.sort(reverse=True)
    num_divs = len(divisors)
    
    # Precompute prefix sums of divisors (largest first)
    # prefix_sum[i] = sum of divisors[0..i-1]
    prefix_sum = [0] * (num_divs + 1)
    for i in range(num_divs):
        prefix_sum[i+1] = prefix_sum[i] + divisors[i]
    
    # Precompute suffix sums (for min possible sum checks)
    # suffix_sum[i] = sum of divisors[i..num_divs-1]
    suffix_sum = [0] * (num_divs + 1)
    for i in range(num_divs - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i+1] + divisors[i]
    
    # Step 1: Find minimum K using Bitset DP
    # dp[k] is a bitset where bit s is 1 if sum s is achievable with k divisors
    # We only need to go up to K such that we find the target.
    # Max K is num_divs.
    
    # Initialize dp
    # dp[0] = 1 (sum 0)
    # dp[k] = 0 for k > 0
    dp = [0] * (num_divs + 1)
    dp[0] = 1
    
    mask = (1 << (target + 1)) - 1
    
    for d in divisors:
        # Update from high k to low k to ensure each divisor is used at most once
        # We can limit k to num_divs
        for k in range(num_divs, 0, -1):
            if dp[k-1]:
                dp[k] |= (dp[k-1] << d)
                dp[k] &= mask
    
    # Find minimum k
    min_k = -1
    for k in range(1, num_divs + 1):
        if (dp[k] >> target) & 1:
            min_k = k
            break
    
    if min_k == -1:
        # Fallback if not found (should not happen)
        return []
    
    # Step 2: Backtrack to find the specific divisors
    # Use iterative backtracking with pruning to avoid recursion depth issues
    
    # We'll use a stack-based approach for backtracking
    # State: (index, k, remaining_sum, selected)
    # selected is a list of indices chosen so far
    
    # For efficiency, we'll use a recursive approach with memoization of failed states
    # But since the search space is small for min_k, let's try a smarter backtracking
    
    import sys
    sys.setrecursionlimit(10000)
    
    # Use a set to track visited (index, k, remaining_sum) states that failed
    # This can help avoid re-exploring dead ends
    # But the state space might be large, so let's be careful
    
    failed_states = set()
    
    def backtrack(index, k, remaining_sum, selected):
        """
        Returns a list of divisors or None.
        index: current index in divisors
        k: number of divisors to pick
        remaining_sum: sum to achieve
        selected: list of divisors chosen so far
        """
        # Time check
        if time.time() - start_time > DEADLINE:
            return None
        
        if k == 0:
            if remaining_sum == 0:
                return list(selected)
            else:
                return None
        
        if index >= num_divs:
            return None
            
        # Pruning: Not enough divisors left
        if num_divs - index < k:
            return None
            
        # Pruning using prefix sums: max possible sum with k divisors from index onwards
        # The max sum is sum of the first k divisors from index
        if k > 0:
            # Sum of largest k divisors from index to index+k-1
            max_possible = prefix_sum[index + k] - prefix_sum[index]
            if max_possible < remaining_sum:
                return None
            
            # Sum of smallest k divisors from index onwards
            # These are the last k divisors in divisors[index:]
            # i.e., divisors[num_divs-k : num_divs]
            # But we need to ensure index <= num_divs - k
            if index <= num_divs - k:
                min_possible = suffix_sum[num_divs - k] - suffix_sum[num_divs]  # This is just sum of last k
                # Actually, suffix_sum[num_divs-k] = sum of divisors[num_divs-k .. num_divs-1]
                # suffix_sum[num_divs] = 0
                # So min_possible = suffix_sum[num_divs-k]
                min_possible = suffix_sum[num_divs - k]
                if min_possible > remaining_sum:
                    return None
        
        # Check if this state has been tried and failed
        state = (index, k, remaining_sum)
        if state in failed_states:
            return None
        
        # Option 1: Include divisors[index]
        if divisors[index] <= remaining_sum:
            selected.append(divisors[index])
            res = backtrack(index + 1, k - 1, remaining_sum - divisors[index], selected)
            selected.pop()
            if res is not None:
                return res
        
        # Option 2: Exclude divisors[index]
        res = backtrack(index + 1, k, remaining_sum, selected)
        if res is not None:
            return res
            
        # Mark this state as failed
        failed_states.add(state)
        
        return None

    solution = backtrack(0, min_k, target, [])
    if solution:
        # Verify the solution
        assert len(solution) == min_k
        assert len(set(solution)) == min_k  # distinct
        assert sum(solution) == target
        for d in solution:
            assert n % d == 0
        return solution
    else:
        # Fallback: try to find any valid solution (not necessarily minimal)
        # Use a simple greedy approach
        divs_asc = sorted(divisors)
        remaining = target
        result = []
        for d in reversed(divs_asc):
            if d <= remaining and d not in result:
                result.append(d)
                remaining -= d
                if remaining == 0:
                    break
        if sum(result) == target and len(set(result)) == len(result):
            return result
        return []

# EVOLVE-BLOCK-END
