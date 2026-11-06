"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from heapq import heappush, heappop

def construct():
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    # Filter to divisors <= target (we can't use divisors larger than target)
    divisors = [d for d in divisors if d <= target]
    divisors.sort(reverse=True)
    num_divs = len(divisors)
    
    # Dijkstra's algorithm on state space of (sum, used_mask)
    # Each edge adds one divisor, so edge weight = 1
    # We want to find the shortest path from (0, 0) to (target, any_mask)
    # State: (sum, mask) where mask is a bitmask of used divisor indices
    # Since num_divs <= 60, mask fits in a 60-bit integer
    
    # dist[sum][mask] = minimum number of divisors used to reach this state
    # But we can't store all masks for each sum. Instead, we use a dict:
    # dist[(sum, mask)] = count
    
    # For efficiency, we'll use a priority queue: (count, sum, mask, path)
    # But storing the full path is expensive. Instead, we store parent pointers.
    
    # Actually, let's use a simpler approach: BFS level by level (by count)
    # Since all edges have weight 1, BFS gives us the minimum count.
    
    # State: (sum, mask)
    # We want to reach sum = target with any mask.
    
    # Use a set to track visited states: (sum, mask)
    visited = set()
    visited.add((0, 0))
    
    # Queue: (count, sum, mask, parent_sum, parent_mask)
    # We'll store the path implicitly by tracking parents.
    
    # For memory efficiency, let's store for each state its parent state.
    parent = {}  # (sum, mask) -> (prev_sum, prev_mask)
    
    # BFS
    from collections import deque
    queue = deque()
    queue.append((0, 0))
    
    # We'll process level by level
    count = 0
    found = False
    result_state = None
    
    while queue and not found:
        # Process all states at current count level
        level_size = len(queue)
        for _ in range(level_size):
            current_sum, current_mask = queue.popleft()
            
            if current_sum == target:
                found = True
                result_state = (current_sum, current_mask)
                break
            
            # Try adding each unused divisor
            for i in range(num_divs):
                if current_mask & (1 << i):
                    continue  # Already used
                d = divisors[i]
                new_sum = current_sum + d
                if new_sum > target:
                    continue
                new_mask = current_mask | (1 << i)
                new_state = (new_sum, new_mask)
                if new_state not in visited:
                    visited.add(new_state)
                    parent[new_state] = (current_sum, current_mask)
                    queue.append(new_state)
        
        count += 1
        if count > 20:  # Safety limit
            break
    
    if found:
        # Reconstruct the path
        path = []
        state = result_state
        while state != (0, 0):
            sum_val, mask = state
            prev_sum, prev_mask = parent[state]
            # The divisor added is the one that was newly set in mask
            # Find which bit was added
            diff_mask = mask ^ prev_mask
            # Get the index of the set bit
            i = diff_mask.bit_length() - 1
            path.append(divisors[i])
            state = (prev_sum, prev_mask)
        path.reverse()
        return path
    
    # Fallback: greedy approach
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    assert remainder == 0
    return chosen

# EVOLVE-BLOCK-END
