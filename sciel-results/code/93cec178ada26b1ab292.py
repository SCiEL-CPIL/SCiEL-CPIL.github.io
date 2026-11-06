import time
import heapq

def construct():
    # Deadline guard: 90s wall-clock limit, use 75s safety margin
    start_time = time.time()
    deadline = start_time + 75.0
    
    n = 5040
    target = 4979
    
    # 1. Get all divisors of 5040 efficiently
    divisors = [i for i in range(1, n + 1) if n % i == 0]
    divisors.sort(reverse=True)
    D = len(divisors)
    divisor_set = set(divisors)
    
    # Precompute prefix sums for O(1) range sum queries
    # prefix[i] = sum of divisors[0:i]
    prefix = [0] * (D + 1)
    for i in range(D):
        prefix[i + 1] = prefix[i] + divisors[i]
    
    def range_sum(lo, hi):
        """Sum of divisors[lo:hi] using prefix sums"""
        return prefix[hi] - prefix[lo]
    
    # Specialized fast check for k=1
    def solve_k1(target_val):
        if target_val in divisor_set:
            return [target_val]
        return None
    
    # Specialized fast check for k=2
    def solve_k2(target_val):
        # d1 + d2 = target_val
        # Iterate d1, check if target_val - d1 is in set
        for i in range(D):
            d1 = divisors[i]
            if d1 >= target_val:
                continue
            d2 = target_val - d1
            if d2 < d1:
                break  # Since divisors are sorted desc, d2 will only get smaller
            if d2 in divisor_set and d2 != d1:
                return [d1, d2]
        return None
    
    # Specialized fast check for k=3
    def solve_k3(target_val):
        # d1 + d2 + d3 = target_val
        # Iterate d1, d2. Check if d3 = target_val - d1 - d2 is in set.
        # To avoid permutations, enforce d1 > d2 > d3.
        for i in range(D):
            d1 = divisors[i]
            if d1 >= target_val:
                continue
            rem1 = target_val - d1
            # d2 must be < d1
            for j in range(i + 1, D):
                d2 = divisors[j]
                if d2 >= rem1:
                    continue
                rem2 = rem1 - d2
                # d3 must be < d2
                if rem2 in divisor_set and rem2 < d2:
                    return [d1, d2, rem2]
                # Pruning: if d2 is too small, rem2 becomes too large to be a divisor < d2?
                # Actually, as d2 decreases, rem2 increases. If rem2 > d2, it might still be valid if rem2 is a divisor.
                # But we require d3 < d2. So if rem2 > d2, it's invalid for this d2.
                # Since d2 is decreasing, rem2 is increasing.
                # If rem2 > d2, then for smaller d2, rem2 will be even larger relative to d2?
                # Not necessarily. But if rem2 > max(divisor_set) it's invalid.
                # Simple break condition: if rem2 > d2, we can't use this d2.
                # Can we break? If rem2 > d2, then d3 > d2, which violates order.
                # Since d2 is decreasing, rem2 is increasing.
                # So if rem2 > d2, for all subsequent j (smaller d2), rem2 will be even larger,
                # and d2 will be smaller, so rem2 > d2 will hold.
                # Thus we can break.
                if rem2 > d2:
                    break
        return None
    
    # Dijkstra's algorithm on state space of (remainder, last_index)
    # State: (cost, remainder, last_index, path_tuple)
    # We want to reach remainder=0 with minimum cost (number of divisors)
    # Since we want to minimize the number of terms, each edge has cost 1.
    # This is a BFS in terms of depth, but Dijkstra handles it generally.
    # However, BFS is sufficient since all edges have unit weight.
    # But we need to track the last index to ensure distinctness and ordering.
    # State: (remainder, last_index)
    # We start from (target, -1) with cost 0.
    # We add a divisor d_i where i > last_index.
    # New state: (remainder - d_i, i) with cost + 1.
    # We stop when remainder == 0.
    
    # To ensure we find the minimal number of terms, we can do a BFS layer by layer.
    # Or use a priority queue where the key is the number of terms.
    
    def solve_dijkstra(target_val):
        # State: (num_terms, remainder, last_index)
        # We want to find the path to remainder=0 with minimum num_terms.
        # visited: set of (remainder, last_index) to avoid cycles (though graph is acyclic due to increasing index)
        # Actually, since indices strictly increase, there are no cycles.
        # We can just use a dictionary or set to store the best (min terms) for a given (remainder, last_index).
        # But since we are doing BFS (unit weights), the first time we visit a state is the optimal.
        
        # Optimization: We can prune states where remainder < 0.
        # Also, we can prune if remainder is not achievable with remaining divisors.
        
        # Start state: (0 terms, target_val, -1)
        # Queue: (num_terms, remainder, last_index, path)
        # Path is a tuple of indices to reconstruct the solution.
        
        # To save memory, we can store the path as a tuple of indices.
        # Given the small size (D=60), this is manageable.
        
        # However, the number of states is D * Target. Target is ~5000. D is 60.
        # 60 * 5000 = 300,000 states. This is very fast.
        
        # Let's use a queue for BFS since all edges have weight 1.
        # We want the shortest path in terms of number of edges.
        
        # visited[remainder][last_index] = True
        # Using a set of tuples (remainder, last_index)
        
        visited = set()
        # Queue elements: (num_terms, remainder, last_index, path)
        # path is a tuple of divisor values (or indices)
        
        # We want to minimize num_terms. BFS guarantees this.
        
        # Initial state
        # remainder = target_val
        # last_index = -1
        # num_terms = 0
        # path = ()
        
        # We use a list as a queue (deque is better for popping from left)
        from collections import deque
        queue = deque()
        queue.append((0, target_val, -1, ()))
        visited.add((target_val, -1))
        
        # Precompute min and max possible sums for remaining indices to prune?
        # For a state (remainder, last_index), we can only pick divisors with index > last_index.
        # The max sum we can add is sum(divisors[last_index+1:])
        # The min sum we can add is 0 (pick none, but we need to reach 0)
        # Actually, we need to reach exactly 0.
        
        while queue:
            if time.time() > deadline:
                return None
                
            num_terms, rem, last_idx, path = queue.popleft()
            
            if rem == 0:
                # Found a solution
                return list(path)
                
            # Next divisors must have index > last_idx
            # Iterate from last_idx + 1 to D-1
            # Pruning: if rem < 0, skip (should not happen if we check before adding)
            
            # Optimization: 
            # If rem is very small, we can break early?
            # If divisors[i] > rem, we can skip this divisor.
            # Since divisors are sorted desc, if divisors[i] > rem, then all subsequent divisors are smaller.
            # So we can't break, we must continue to find a divisor <= rem.
            # Wait, divisors are sorted in DESCENDING order.
            # divisors[0] is largest, divisors[D-1] is smallest.
            # So if divisors[i] > rem, we skip.
            # If divisors[i] <= rem, we consider it.
            
            for i in range(last_idx + 1, D):
                d = divisors[i]
                if d > rem:
                    continue
                    
                new_rem = rem - d
                if new_rem < 0:
                    continue
                    
                # Check if we've visited this state
                state = (new_rem, i)
                if state in visited:
                    continue
                visited.add(state)
                
                # Pruning: Can we reach 0 from new_rem using divisors with index > i?
                # The maximum sum we can form is sum(divisors[i+1:])
                # If new_rem > max_sum, we can't reach it.
                # max_sum = prefix[D] - prefix[i+1]
                max_possible = prefix[D] - prefix[i+1]
                if new_rem > max_possible:
                    continue
                
                # The minimum sum is 0 (if we pick no more, but we need to reach 0 exactly)
                # Actually, we just need new_rem to be formable by distinct divisors from i+1 onwards.
                # This check is expensive. The max_possible check is a good cheap filter.
                
                new_path = path + (d,)
                queue.append((num_terms + 1, new_rem, i, new_path))
                
        return None

    # Iterate k from 1 upwards
    # First try specialized solvers for k=1, 2, 3 as they are very fast
    # Then use Dijkstra for k>=4 (or just use Dijkstra for all, but specialized are faster for small k)
    
    # k=1
    result = solve_k1(target)
    if result:
        return result
        
    # k=2
    result = solve_k2(target)
    if result:
        return result
        
    # k=3
    result = solve_k3(target)
    if result:
        return result
        
    # k>=4: Use Dijkstra/BFS
    # The BFS will find the minimal number of terms automatically.
    # So we don't need to iterate k. We just run BFS and it returns the shortest path.
    # Wait, the specialized solvers return the solution for a specific k.
    # The BFS returns the solution with the minimum number of terms overall.
    # So if k=1, 2, 3 failed, BFS will find the solution with k>=4 if it exists.
    # And it will be the minimal k.
    
    result = solve_dijkstra(target)
    if result:
        return result
        
    # Fallback: Greedy approach (should not be reached if a solution exists)
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
    if remainder == 0:
        return chosen
        
    return []
