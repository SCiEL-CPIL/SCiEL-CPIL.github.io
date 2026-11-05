"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    
    N = 512
    
    # Precompute LCM check efficiently
    def lcm(a, b):
        g = math.gcd(a, b)
        return a * b // g
    
    # Build conflict graph: edge between i,j if lcm(i,j) > N
    # Use 0-indexed: node i corresponds to value i+1
    # adj[i] is a bitset where bit j is 1 if lcm(i+1, j+1) > N
    adj = [0] * N
    
    for i in range(N):
        ai = i + 1
        for j in range(i + 1, N):
            aj = j + 1
            if lcm(ai, aj) > N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Greedy coloring upper bound for the induced subgraph on `available`
    # Returns the number of colors used, which is an upper bound on the
    # size of the maximum independent set in the induced subgraph.
    def color_bound(available):
        """Greedy coloring of the conflict graph induced by `available`.
        Returns number of colors (upper bound on MIS size)."""
        if available == 0:
            return 0
        
        # Get list of available nodes, ordered by degree descending
        nodes = []
        for v in range(N):
            if (available >> v) & 1:
                nodes.append(v)
        
        if not nodes:
            return 0
        
        # Sort by degree in available subgraph (descending)
        def deg(v):
            return bin(adj[v] & available).count('1')
        
        nodes.sort(key=deg, reverse=True)
        
        # Greedy coloring: assign smallest available color
        color_of = [-1] * N
        num_colors = 0
        
        for v in nodes:
            # Find colors used by already-colored neighbors of v
            used = set()
            for u in nodes:
                if u == v:
                    break
                if (adj[v] >> u) & 1 and color_of[u] >= 0:
                    used.add(color_of[u])
            
            c = 0
            while c in used:
                c += 1
            color_of[v] = c
            if c + 1 > num_colors:
                num_colors = c + 1
        
        return num_colors
    
    # Greedy independent set: repeatedly pick min-degree node
    def greedy_is():
        avail = (1 << N) - 1
        result = set()
        while avail:
            # Pick node with minimum degree in available subgraph
            best_v = -1
            best_deg = N + 1
            for v in range(N):
                if (avail >> v) & 1:
                    d = bin(adj[v] & avail).count('1')
                    if d < best_deg:
                        best_deg = d
                        best_v = v
            if best_v == -1:
                break
            result.add(best_v + 1)
            avail &= ~(1 << best_v)
            avail &= ~adj[best_v]
        return result
    
    # Get initial solution
    greedy_result = greedy_is()
    best_size = len(greedy_result)
    best_set = set(greedy_result)
    
    # Try a second greedy: max-degree first on complement? Or different order.
    # Try: sort by value and greedily add if no conflict
    def greedy_is_sorted():
        result = set()
        for v in range(N):
            val = v + 1
            # Check if val conflicts with any in result
            conflict = False
            for u in result:
                if lcm(val, u) > N:
                    conflict = True
                    break
            if not conflict:
                result.add(val)
        return result
    
    greedy2 = greedy_is_sorted()
    if len(greedy2) > best_size:
        best_size = len(greedy2)
        best_set = set(greedy2)
    
    # Branch and bound with coloring bound
    # We track: available mask, current chosen set (as a set of 1-indexed values)
    
    def solve(available, current_size, current_set):
        nonlocal best_size, best_set
        
        if time.time() > deadline - 3:
            return
        
        if available == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return
        
        # Upper bound via coloring
        ub = color_bound(available)
        if current_size + ub <= best_size:
            return
        
        # Also: if current_size + popcount(available) <= best_size, prune
        remaining_count = bin(available).count('1')
        if current_size + remaining_count <= best_size:
            return
        
        # Pick a node to branch on
        # Heuristic: pick node with maximum degree in available
        best_v = -1
        best_deg = -1
        for v in range(N):
            if (available >> v) & 1:
                d = bin(adj[v] & available).count('1')
                if d > best_deg:
                    best_deg = d
                    best_v = v
        
        if best_v == -1:
            return
        
        # Branch 1: include best_v (add to independent set)
        new_available = available & ~(1 << best_v) & ~adj[best_v]
        solve(new_available, current_size + 1, current_set | {best_v + 1})
        
        # Check time before second branch
        if time.time() > deadline - 3:
            return
        
        # Branch 2: exclude best_v
        new_available = available & ~(1 << best_v)
        solve(new_available, current_size, current_set)
    
    # Start with all nodes available
    all_nodes = (1 << N) - 1
    solve(all_nodes, 0, set())
    
    return list(best_set)

# EVOLVE-BLOCK-END
