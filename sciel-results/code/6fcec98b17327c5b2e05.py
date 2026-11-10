import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort()
    num_divs = len(divisors)
    
    # For each count k, maintain a dict: sum -> list of divisors used
    # Start with k=0: {0: []}
    # For each divisor, update for k=1..num_divs
    
    # To save memory, only track sums <= target
    # And we want the first k where target is achievable
    
    # Use a list of dicts: sums_by_count[k] = {sum: tuple_of_divisors}
    # But this could be huge. Let's be smarter.
    
    # Alternative: For small k, use direct search with pruning
    # Let's try k from 1 upward with a more efficient DFS
    
    divisors_desc = sorted(divisors, reverse=True)
    
    def solve_with_k(k):
        """Find k distinct divisors summing to target using DFS with pruning."""
        if time.time() > deadline:
            return None
        
        # Precompute suffix sums for pruning
        # suffix_max[i] = sum of largest (k) divisors from index i onwards
        # Actually, let's compute prefix and suffix for bounds
        
        # Use iterative deepening with better pruning
        chosen = [0] * k
        path = []
        
        def dfs(idx, remaining_sum, slots_left):
            if time.time() > deadline:
                return False
            if slots_left == 0:
                return remaining_sum == 0
            if idx >= num_divs:
                return False
            
            # Can we still fill slots_left with remaining divisors?
            if num_divs - idx < slots_left:
                return False
            
            # Max sum we can get with slots_left divisors from idx onwards
            max_sum = sum(divisors_desc[idx:idx+slots_left])
            if remaining_sum > max_sum:
                return False
            
            # Min sum we can get with slots_left divisors from idx onwards
            # Take the smallest slots_left divisors available
            min_sum = sum(divisors_desc[num_divs-slots_left:num_divs])
            if remaining_sum < min_sum:
                return False
            
            # Try including divisors_desc[idx]
            if divisors_desc[idx] <= remaining_sum:
                path.append(divisors_desc[idx])
                if dfs(idx + 1, remaining_sum - divisors_desc[idx], slots_left - 1):
                    return True
                path.pop()
            
            # Try excluding
            # Check if we can still fill slots_left-1 from remaining
            if num_divs - idx - 1 >= slots_left - 1:
                if dfs(idx + 1, remaining_sum, slots_left):
                    return True
            
            return False
        
        if dfs(0, target, k):
            return path[:]
        return None
    
    for k in range(1, num_divs + 1):
        if time.time() > deadline:
            break
        result = solve_with_k(k)
        if result is not None:
            return result
    
    # Fallback greedy
    remainder = target
    chosen = []
    for d in divisors_desc:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    return chosen
