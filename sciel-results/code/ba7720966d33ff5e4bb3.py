import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 24
    rng = random.Random(614)
    
    # Generate the shuffled edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    rng.shuffle(edges)
    
    # Map edge to its 1-based position
    pos = {}
    for i, (u, v) in enumerate(edges):
        pos[(u, v)] = i + 1
        pos[(v, u)] = i + 1
    
    # Precompute for each K: adjacency bitmask
    # adj_mask[u] = bitmask of neighbors of u in first K edges
    # We'll build incrementally
    
    # Find min K where all degrees >= 2
    deg = [0] * n
    min_deg_K = n
    for K in range(n):
        u, v = edges[K]
        deg[u] += 1
        deg[v] += 1
        if min(deg) >= 2:
            min_deg_K = K + 1
            break
    
    lo = max(n, min_deg_K)
    hi = n * (n - 1) // 2
    
    def find_hamilton_cycle(K, time_limit):
        """Try to find a Hamilton cycle in G_K using bitmask DFS."""
        
        # Build adjacency bitmasks
        adj_mask = [0] * n
        for i in range(K):
            u, v = edges[i]
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # Check min degree >= 2
        for u in range(n):
            if bin(adj_mask[u]).count('1') < 2:
                return None
        
        # Check connectivity via BFS
        visited = [False] * n
        visited[0] = True
        stack = [0]
        count = 1
        while stack:
            u = stack.pop()
            m = adj_mask[u]
            while m:
                v = (m & (-m)).bit_length() - 1
                m &= m - 1
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    stack.append(v)
        if count < n:
            return None
        
        full_mask = (1 << n) - 1
        
        # Try multiple starting nodes and strategies
        # Warnsdorff: prefer neighbor with fewest unvisited neighbors
        # We'll try different starting points and tie-breaking seeds
        
        for strat in range(30):
            if time.time() > time_limit:
                break
            
            # Choose start node
            if strat == 0:
                start = 0
            elif strat == 1:
                start = 1
            elif strat == 2:
                start = min(range(n), key=lambda x: bin(adj_mask[x]).count('1'))
            elif strat == 3:
                start = max(range(n), key=lambda x: bin(adj_mask[x]).count('1'))
            else:
                r = random.Random(strat * 7919 + 13)
                start = r.randint(0, n - 1)
            
            # Iterative DFS with bitmask
            visited_mask = 1 << start
            path = [start]
            # Stack: list of (node, available_neighbors_mask)
            # available = adj_mask[node] & ~visited_mask
            
            # For Warnsdorff, we need to sort neighbors by their unvisited degree
            # Precompute a helper: for each vertex, its adj_mask
            # unvisited_deg(v) = popcount(adj_mask[v] & ~visited_mask)
            
            def popcount(x):
                return bin(x).count('1')
            
            # Use explicit stack for backtracking
            # Each frame: (current_node, next_neighbor_to_try, choices_list)
            # But with bitmask, we can iterate through set bits
            
            # Simpler: recursive-style with explicit stack
            # State: (node, untried_neighbors_mask)
            # When we go to a neighbor, we push (new_node, adj_mask[new_node] & ~new_visited)
            
            # Let's use a cleaner approach:
            # stack of (node, remaining_choices_mask)
            # path tracks the current path
            # visited_mask tracks visited vertices
            
            visited_mask = 1 << start
            path = [start]
            
            # Get initial choices for start
            initial_choices = adj_mask[start] & ~visited_mask
            # Sort by Warnsdorff (fewest unvisited neighbors first)
            choices_list = []
            m = initial_choices
            while m:
                v = (m & (-m)).bit_length() - 1
                m &= m - 1
                choices_list.append(v)
            
            # Sort by unvisited degree (Warnsdorff)
            choices_list.sort(key=lambda v: popcount(adj_mask[v] & ~visited_mask))
            
            # Stack: each entry is (node, choices_iterator_index)
            # We'll use a list of (node, list_of_remaining_choices)
            stack = [(start, choices_list)]
            
            found = False
            max_steps = 500000  # Safety limit per strategy
            
            steps = 0
            while stack:
                steps += 1
                if steps > max_steps or time.time() > time_limit:
                    break
                
                node, choices = stack[-1]
                
                if len(path) == n:
                    # Check if we can close the cycle
                    if (1 << start) & adj_mask[node]:
                        found = True
                        break
                    # Can't close, backtrack
                    stack.pop()
                    path.pop()
                    visited_mask &= ~(1 << node)
                    continue
                
                if not choices:
                    # No more choices, backtrack
                    stack.pop()
                    path.pop()
                    visited_mask &= ~(1 << node)
                    continue
                
                # Take next choice
                v = choices.pop(0)
                stack[-1] = (node, choices)
                
                visited_mask |= (1 << v)
                path.append(v)
                
                # Get next choices for v
                new_choices = adj_mask[v] & ~visited_mask
                cl = []
                m2 = new_choices
                while m2:
                    w = (m2 & (-m2)).bit_length() - 1
                    m2 &= m2 - 1
                    cl.append(w)
                
                # Sort by Warnsdorff
                cl.sort(key=lambda w: popcount(adj_mask[w] & ~visited_mask))
                
                stack.append((v, cl))
            
            if found:
                return path[:]
        
        return None
    
    # Binary search for minimum K
    # First verify hi works
    best_cycle = None
    
    # Do binary search
    lo_bs = lo
    hi_bs = hi
    
    # First check if hi works (should, since K_n has Hamilton cycles)
    cycle_hi = find_hamilton_cycle(hi_bs, deadline - 10.0)
    if cycle_hi is None:
        # Fallback
        return list(range(n))
    
    best_cycle = cycle_hi
    best_K = hi_bs
    
    # Binary search
    while lo_bs < hi_bs and time.time() < deadline - 5.0:
        mid = (lo_bs + hi_bs) // 2
        cycle_mid = find_hamilton_cycle(mid, deadline - 10.0)
        if cycle_mid is not None:
            best_cycle = cycle_mid
            best_K = mid
            hi_bs = mid
        else:
            lo_bs = mid + 1
    
    # After binary search, try to go a bit lower to see if we can find something
    # Actually, binary search should give us the minimum
    # But let's verify: try best_K - 1
    if best_K > lo and time.time() < deadline - 5.0:
        cycle_lower = find_hamilton_cycle(best_K - 1, deadline - 10.0)
        if cycle_lower is not None:
            best_cycle = cycle_lower
            best_K = best_K - 1
    
    return best_cycle

if __name__ == "__main__":
    print(construct())
