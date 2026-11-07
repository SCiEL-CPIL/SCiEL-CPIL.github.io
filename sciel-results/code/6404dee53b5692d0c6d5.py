"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    N = 210
    MAX_VAL = 10210
    
    # Precompute squarefree kernels using sieve
    sf = [0] * (MAX_VAL + 1)
    sf[1] = 1
    for i in range(2, MAX_VAL + 1):
        x = i
        kernel = 1
        d = 2
        while d * d <= x:
            count = 0
            while x % d == 0:
                x //= d
                count += 1
            if count % 2 == 1:
                kernel *= d
            d += 1
        if x > 1:
            kernel *= x
        sf[i] = kernel
    
    # Get all divisors of 210
    divisors = []
    for i in range(1, int(math.isqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    divisors_set = set(divisors)
    
    # Map each divisor to a bitmask representation
    # 210 = 2^1 * 3^1 * 5^1 * 7^1
    # We can represent each divisor by which primes it contains
    prime_factors = [2, 3, 5, 7]
    divisor_to_mask = {}
    for d in divisors:
        mask = 0
        x = d
        for i, p in enumerate(prime_factors):
            if x % p == 0:
                mask |= (1 << i)
                x //= p
        divisor_to_mask[d] = mask
    
    target_mask = 0
    for i in range(4):
        target_mask |= (1 << i)
    
    # For each number > 210, check if its squarefree kernel divides 210
    # Group numbers by their kernel
    kernel_to_numbers = {}
    for x in range(N + 1, MAX_VAL + 1):
        k = sf[x]
        if N % k == 0:
            if k not in kernel_to_numbers:
                kernel_to_numbers[k] = []
            kernel_to_numbers[k].append(x)
    
    # Sort numbers in each kernel group (already sorted by construction)
    # We want to find a combination of numbers whose kernels multiply to 210
    # and minimize the maximum number selected
    
    # Use BFS/DP: state is the current product of kernels (which divides 210)
    # dp[state] = list of selected numbers (or we track parent for backtracking)
    
    # Better: for each possible max value M from 211 upward, check if there's a solution
    # using only numbers <= M. Binary search on M.
    
    def can_solve(max_val):
        """Check if we can solve using only numbers in (210, max_val]"""
        # DP: dp[mask] = True if we can achieve this mask using numbers <= max_val
        # We need to reach target_mask
        # Use a set of achievable masks, starting from 0
        # For each number, we can either use it or not
        
        # Actually, since we need distinct numbers and their kernels multiply to 210,
        # let's use a different approach:
        # dp[state] = set of (min_max, selected_numbers) but that's expensive
        
        # Simpler: BFS where state is the current kernel product
        # For each number x with kernel k, we can transition from state s to s*k if s*k divides 210
        # We want to reach state 210
        
        reachable = {1: []}  # state: list of numbers used
        
        # Process numbers in order, but we need to be careful about distinctness
        # Since each number is used at most once, and we're building incrementally:
        
        # For each candidate number (sorted), try to extend existing solutions
        # This is exponential in worst case, but the state space is small (16 states)
        
        # Let's use: dp[state] = list of solutions (each solution is a tuple of numbers)
        # But we want to minimize the max, so let's process numbers in increasing order
        
        # Actually, let's use a cleaner approach:
        # states = {1: []}
        # For each number x from 211 to max_val:
        #   k = sf[x]
        #   if 210 % k != 0: skip
        #   new_states = {}
        #   for s, sol in states.items():
        #       new_s = s * k
        #       if 210 % new_s == 0:
        #           if new_s not in new_states or len(sol) + 1 < len(new_states[new_s]):
        #               new_states[new_s] = sol + [x]
        #   for ns, nsol in new_states.items():
        #       if ns not in states or len(nsol) < len(states[ns]):
        #           states[ns] = nsol
        #   if 210 in states:
        #       return states[210]
        # return None
        
        states = {1: []}
        
        for x in range(N + 1, max_val + 1):
            k = sf[x]
            if N % k != 0:
                continue
            
            # Try to extend existing solutions
            to_add = {}
            for s, sol in states.items():
                new_s = s * k
                if N % new_s == 0:
                    # new_s is a valid state
                    if new_s not in to_add:
                        to_add[new_s] = sol + [x]
                    else:
                        # Keep the one with smaller max (which is x since we're going in order)
                        # Actually both have max = x since we're adding x
                        # Keep the one with fewer elements for efficiency
                        if len(sol) + 1 < len(to_add[new_s]):
                            to_add[new_s] = sol + [x]
            
            for ns, nsol in to_add.items():
                if ns not in states:
                    states[ns] = nsol
                else:
                    # Keep the solution with smaller maximum element
                    # Since we're processing in order, the new solution has max = x
                    # The old solution has max <= x
                    # Prefer the old one if it has smaller max
                    old_max = max(states[ns]) if states[ns] else 0
                    new_max = x
                    if new_max < old_max:
                        states[ns] = nsol
                    elif new_max == old_max and len(nsol) < len(states[ns]):
                        states[ns] = nsol
            
            if N in states:
                return states[N]
        
        return None
    
    # Binary search on the maximum value
    lo = N + 1
    hi = MAX_VAL
    best_solution = None
    
    # First, try to find the minimum possible max
    while lo <= hi:
        mid = (lo + hi) // 2
        sol = can_solve(mid)
        if sol is not None:
            best_solution = sol
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_solution is None:
        # Fallback: use the original approach
        reachable = {1}
        parent = {1: (None, None)}
        for x in range(N + 1, MAX_VAL + 1):
            k = sf[x]
            if N % k != 0:
                continue
            new_states = {}
            for p in list(reachable):
                q = p * k
                if N % q == 0:
                    if q not in reachable:
                        new_states[q] = (p, x)
            for q, (p, num) in new_states.items():
                reachable.add(q)
                parent[q] = (p, num)
            if N in reachable:
                selected = []
                current_state = N
                while current_state != 1:
                    prev_state, num = parent[current_state]
                    selected.append(num)
                    current_state = prev_state
                return selected
        return [4 * N]
    
    # Verify the solution
    prod = 1
    for num in best_solution:
        prod *= num
    total = prod * N
    root = math.isqrt(total)
    if root * root == total and len(best_solution) >= 1 and len(best_solution) <= 100:
        if all(isinstance(x, int) and not isinstance(x, bool) for x in best_solution):
            if all(210 < x <= 10210 for x in best_solution):
                if len(set(best_solution)) == len(best_solution):
                    return list(best_solution)
    
    # If verification failed, fallback
    return [4 * N]
# EVOLVE-BLOCK-END
