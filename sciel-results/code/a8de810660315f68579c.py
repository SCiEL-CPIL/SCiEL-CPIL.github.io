import collections
import itertools
import heapq

def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # Encode permutations of n-1 symbols from 1..n
    # We use integers for states.
    # State is a tuple of n-1 distinct symbols.
    # Total states: n!
    
    # Precompute all permutation states
    perms = list(itertools.permutations(range(1, n + 1), n - 1))
    num_nodes = len(perms)
    node_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Compute deterministic next (for cycle detection)
    # This defines the "main" cycles
    det_next = [0] * num_nodes
    for i, p in enumerate(perms):
        missing = 0
        # Optimization: sum of 1..n is n(n+1)/2
        # Sum of p is sum_p
        # missing = total_sum - sum_p
        total_sum = n * (n + 1) // 2
        sum_p = sum(p)
        missing = total_sum - sum_p
        next_p = p[1:] + (missing,)
        det_next[i] = node_to_idx[next_p]
        
    # Find cycles
    visited = [False] * num_nodes
    cycles = []
    cycle_id = [0] * num_nodes
    for i in range(num_nodes):
        if not visited[i]:
            cycle = []
            curr = i
            while not visited[curr]:
                visited[curr] = True
                cycle.append(curr)
                cycle_id[curr] = len(cycles)
                curr = det_next[curr]
            cycles.append(cycle)
            
    num_cycles = len(cycles)
    
    # We need to visit all cycles.
    # Start at node 0.
    start_node_idx = 0
    current_node = perms[start_node_idx]
    word = list(current_node)
    
    visited_cycles = set()
    visited_cycles.add(cycle_id[start_node_idx])
    
    # Helper to get next symbol and node for a given state
    # We need to explore all possible next states, not just the deterministic one.
    # But we want to find the shortest path to the next unvisited cycle.
    
    # To speed up BFS, we use integer encoding for states.
    # Map tuple to int index?
    # We already have node_to_idx for permutation states.
    # But BFS might visit non-permutation states.
    # We need a mapping for ALL states of length n-1 from 1..n.
    # Total states: n^(n-1). For n=7, 7^6 = 117649.
    # We can encode a state (a, b, c, d, e, f) as a number in base n.
    # state_val = a * n^(n-2) + b * n^(n-3) + ...
    
    pow_n = [1] * n
    for i in range(1, n):
        pow_n[i] = pow_n[i-1] * n
        
    def encode_state(t):
        val = 0
        for i in range(len(t)):
            val = val * n + t[i]
        return val

    def decode_state(val):
        t = []
        for i in range(n - 1):
            t.append(val % n + 1)
            val //= n
        return tuple(t)

    # Precompute neighbors for all possible states?
    # Or compute on the fly.
    # On the fly is fine if we use efficient encoding.
    
    # We will run BFS from current_node to find the closest node in an unvisited cycle.
    # We need to check if a state is a permutation and if its cycle is unvisited.
    
    # To make this faster, we can precompute the cycle_id for ALL permutation states.
    # cycle_id is already computed for permutation states (indices 0..num_nodes-1).
    # For non-permutation states, we don't have a cycle_id, but we don't care unless we land on a permutation.
    
    # Optimization: 
    # 1. Use integer states in BFS.
    # 2. Store parent pointers to reconstruct path.
    # 3. Early termination when an unvisited cycle is found.
    
    for _ in range(num_cycles):
        if len(visited_cycles) == num_cycles:
            break
            
        # BFS from current_node
        start_val = encode_state(current_node)
        
        # dist[state] = distance
        # parent[state] = previous state
        # We use a dictionary for visited states in this BFS
        visited_bfs = {start_val: None} # state -> parent state (None for start)
        
        queue = collections.deque()
        queue.append(start_val)
        
        found = False
        target_val = -1
        target_node_tuple = None
        
        while queue:
            curr_val = queue.popleft()
            curr_tuple = decode_state(curr_val)
            
            # Check if this state is a permutation and in an unvisited cycle
            if curr_tuple in node_to_idx:
                idx = node_to_idx[curr_tuple]
                cid = cycle_id[idx]
                if cid not in visited_cycles:
                    found = True
                    target_val = curr_val
                    target_node_tuple = curr_tuple
                    break
            
            # Expand neighbors
            # Current state is a tuple of n-1 symbols.
            # Next state is curr_tuple[1:] + (s,)
            # We can generate the integer directly?
            # curr_val represents a sequence a1...a_{n-1}
            # Next state is a2...a_{n-1} s
            # This is equivalent to (curr_val % n^(n-2)) * n + s
            # Let's verify.
            # val = a1 * n^{n-2} + a2 * n^{n-3} + ... + a_{n-1}
            # val % n^{n-2} = a2 * n^{n-3} + ... + a_{n-1}
            # new_val = (val % n^{n-2}) * n + s
            # This works.
            
            base = curr_val % pow_n[n-2] if n > 2 else 0 # n-2 power
            # Wait, pow_n[n-2] is n^(n-2).
            # If n=2, n-2=0, pow_n[0]=1. base = val % 1 = 0.
            # If n=3, n-2=1, pow_n[1]=3. base = val % 3.
            
            for s in range(1, n + 1):
                next_val = base * n + s
                if next_val not in visited_bfs:
                    visited_bfs[next_val] = curr_val
                    queue.append(next_val)
                    
        if not found:
            # Should not happen
            break
            
        # Reconstruct path from start_val to target_val
        path = []
        curr = target_val
        while visited_bfs[curr] is not None:
            parent = visited_bfs[curr]
            # Determine the symbol appended to go from parent to curr
            # parent_val -> curr_val
            # curr_val = (parent_val % pow_n[n-2]) * n + s
            # s = curr_val % n
            s = curr_val % n
            path.append(s)
            curr = parent
            
        path.reverse()
        
        # Append path to word
        for s in path:
            word.append(s)
            
        # Update current_node
        current_node = target_node_tuple
        visited_cycles.add(cycle_id[node_to_idx[current_node]])
        
    return "".join(map(str, word))
