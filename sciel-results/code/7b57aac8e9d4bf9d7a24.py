import random
import time
import sys

def construct(k, g):
    # For the specific case k=4, g=10, we use a random lift of K4.
    # For other cases, we fall back to a simple random regular graph or LUW if applicable, 
    # but the task implies k=4, g=10.
    
    if k != 4 or g != 10:
        # Fallback to LUW-like or random
        # This is a placeholder for general case, but we focus on k=4, g=10
        return _construct_luw(k, g)
    
    # Base graph: K4
    # Vertices 0, 1, 2, 3
    # Edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    base_edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    num_base_vertices = 4
    num_base_edges = len(base_edges)
    
    # Pre-compute non-backtracking closed walks of length L < g
    # Walk is a sequence of edges (e1, e2, ..., eL)
    # We need to ensure the walk is closed in the base graph and non-backtracking.
    
    # Map edge index to (u, v) and (v, u)
    # Let's represent edges as indices 0..5
    # e0: 0-1, e1: 0-2, e2: 0-3, e3: 1-2, e4: 1-3, e5: 2-3
    
    # Adjacency list for base graph
    adj = [[] for _ in range(num_base_vertices)]
    for i, (u, v) in enumerate(base_edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    walks_by_length = {L: [] for L in range(3, g)}
    
    # Generate walks
    # Start from each vertex, each edge
    # State: (current_vertex, prev_edge_index)
    # To avoid backtracking, next edge cannot be prev_edge_index
    
    for L in range(3, g):
        walks = []
        for start_v in range(num_base_vertices):
            for next_v, e0 in adj[start_v]:
                # Start walk with e0
                # We are at next_v, came from e0
                # We need to extend to length L
                # Current path edges: [e0]
                # Current vertex: next_v
                # Prev edge: e0
                
                # Recursive generation or iterative
                # Since L is small (up to 9), recursion is fine
                def dfs(current_v, prev_e, current_edges, depth):
                    if depth == L:
                        # Check if closed
                        # The last edge in current_edges connects current_v to some u
                        # We need to check if u == start_v
                        # Actually, the walk is a sequence of edges.
                        # Let's store the sequence of vertices too to easily check closure.
                        # Or just check the end vertex of the last edge.
                        # last_edge = current_edges[-1]
                        # u, v = base_edges[last_edge]
                        # The walk ends at current_v.
                        # The start vertex is start_v.
                        # We need the walk to start at start_v and end at start_v.
                        # The first edge e0 was (start_v, next_v).
                        # So the walk starts at start_v.
                        # If the current vertex (end of last edge) is start_v, it is closed.
                        if current_v == start_v:
                            walks.append(tuple(current_edges))
                        return
                    
                    for next_v2, e_next in adj[current_v]:
                        if e_next == prev_e:
                            continue
                        # Add e_next
                        # Determine new current vertex
                        u, v = base_edges[e_next]
                        if u == current_v:
                            new_v = v
                        else:
                            new_v = u
                        dfs(new_v, e_next, current_edges + [e_next], depth + 1)
                
                dfs(next_v, e0, [e0], 1)
        
        # Remove duplicates? A walk is defined by its edge sequence.
        # Different start vertices might generate the same cyclic sequence?
        # We are counting walks, not cycles. The voltage sum is the same for the same edge sequence.
        # We just need the set of edge sequences.
        # Using a set to deduplicate is good.
        walks_by_length[L] = list(set(walks))
        
    # Now we have a list of edge sequences for each length.
    # We want to assign voltages to edges 0..5 such that no sum is 0 mod M.
    
    # We will search for M.
    # Best known n=320 => M=80.
    # Let's try M values.
    
    deadline = time.time() + 110 # Leave 40s margin for construction and safety
    
    best_M = None
    best_voltages = None
    found_solution = False
    
    # Try M from 20 to 100
    # If we find a valid one, we can try to minimize M further?
    # The score is n, so smaller M is better.
    # Let's iterate M from small to large, and for each M, do a short search.
    # If we find a solution, we return it immediately? 
    # No, we should try to find the smallest M.
    
    # However, finding a solution for small M might be hard/impossible.
    # Let's start with M=80 and work down? Or start small and go up?
    # If we start small, we might waste time on impossible M.
    # Let's just pick a likely good M, e.g., 80, and search.
    # If we find a solution, we return it.
    # To be safe, we can try a few M values.
    
    # Actually, the random lift approach is known to find cages.
    # Let's try M = 80 first.
    
    M = 80
    
    # Precompute walk edge counts for faster evaluation?
    # Or just compute sum directly.
    # Number of walks is small.
    
    def get_energy(volts):
        """Returns the number of short closed walks (energy). 0 is optimal."""
        energy = 0
        for L in range(3, g):
            for walk in walks_by_length[L]:
                s = 0
                for e in walk:
                    s += volts[e]
                if s % M == 0:
                    energy += 1
        return energy
    
    # Simulated Annealing
    # Initial random voltages
    voltages = [random.randint(0, M-1) for _ in range(num_base_edges)]
    current_energy = get_energy(voltages)
    
    if current_energy == 0:
        found_solution = True
    else:
        # SA parameters
        T_start = 10.0
        T_end = 0.1
        alpha = 0.995 # Cooling rate
        T = T_start
        best_voltages = voltages[:]
        best_energy = current_energy
        
        while T > T_end and time.time() < deadline:
            # Pick a random edge to change
            e_idx = random.randint(0, num_base_edges - 1)
            old_volt = voltages[e_idx]
            new_volt = random.randint(0, M-1)
            
            if new_volt == old_volt:
                continue
                
            voltages[e_idx] = new_volt
            new_energy = get_energy(voltages)
            
            if new_energy < current_energy:
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_voltages = voltages[:]
                    if best_energy == 0:
                        break
            else:
                # Accept worse solution with probability exp(-deltaE / T)
                delta_e = new_energy - current_energy
                if delta_e > 0:
                    prob = pow(2.718281828, -delta_e / T)
                    if random.random() < prob:
                        current_energy = new_energy
                    else:
                        voltages[e_idx] = old_volt
            
            T *= alpha
            
        if best_energy == 0:
            found_solution = True
            voltages = best_voltages
    
    if not found_solution:
        # Try other M values if M=80 failed
        for M_try in [60, 70, 90, 100]:
            if time.time() > deadline: break
            M = M_try
            voltages = [random.randint(0, M-1) for _ in range(num_base_edges)]
            current_energy = get_energy(voltages)
            
            if current_energy == 0:
                found_solution = True
                break
            
            T = 10.0
            best_energy = current_energy
            best_voltages = voltages[:]
            
            while T > 0.1 and time.time() < deadline:
                e_idx = random.randint(0, num_base_edges - 1)
                old_volt = voltages[e_idx]
                new_volt = random.randint(0, M-1)
                
                if new_volt == old_volt:
                    continue
                    
                voltages[e_idx] = new_volt
                new_energy = get_energy(voltages)
                
                if new_energy < current_energy:
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_voltages = voltages[:]
                        if best_energy == 0:
                            break
                else:
                    delta_e = new_energy - current_energy
                    if delta_e > 0:
                        prob = pow(2.718281828, -delta_e / T)
                        if random.random() < prob:
                            current_energy = new_energy
                        else:
                            voltages[e_idx] = old_volt
                
                T *= 0.995
                
            if best_energy == 0:
                found_solution = True
                voltages = best_voltages
                break
    
    if not found_solution:
        # Fallback to LUW if SA failed
        return _construct_luw(k, g)
        
    # Construct the graph
    n = num_base_vertices * M
    edges = []
    # Map (v, x) to index
    # index = v * M + x
    for u, v in base_edges:
        # Edge u-v has voltage voltages[edge_idx]
        # We need the index of this edge in base_edges
        # base_edges list order:
        # 0: (0,1), 1: (0,2), 2: (0,3), 3: (1,2), 4: (1,3), 5: (2,3)
        # We need to find the index for (u,v)
        # Since base_edges is small, we can just search or precompute
        pass
        
    # Precompute edge indices
    edge_to_idx = {}
    for i, (u, v) in enumerate(base_edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i
        
    for u in range(num_base_vertices):
        for v, e_idx in adj[u]:
            if u < v: # Only add each edge once
                volt = voltages[e_idx]
                for x in range(M):
                    y = (x + volt) % M
                    i1 = u * M + x
                    i2 = v * M + y
                    edges.append((i1, i2))
                    
    return n, edges

def _construct_luw(k, g):
    # Original LUW code as fallback
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
