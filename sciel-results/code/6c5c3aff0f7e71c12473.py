"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def construct(k, g):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # For k=3, g=16, we try a random lift of the Petersen graph.
    # The Petersen graph is 3-regular with girth 5.
    # A lift with voltage group Z_m has 10*m vertices.
    # We seek a voltage assignment such that no cycle of length < g in the base
    # graph closes to 0 mod m.
    
    if k != 3:
        # Fallback to LUW for other k if needed, though task specifies k=3
        return _luw_construct(k, g)

    # Petersen Graph Definition
    # Vertices 0..4 are the outer 5-cycle, 5..9 are the inner 5-point star
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer cycle
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # Inner star (5-7-9-6-8-5)
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # Spokes
    ]
    n_base = 10
    
    # Find all simple cycles of length < g in the Petersen graph
    # We need to check cycles of length 5 to g-1
    # Using DFS from each vertex
    cycles = []
    adj = [[] for _ in range(n_base)]
    for u, v in petersen_edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # DFS to find simple cycles
    # To avoid duplicates, we only start cycles from the smallest vertex in the cycle
    for start in range(n_base):
        # Stack: (current, path, visited_mask)
        stack = [(start, [start], 1 << start)]
        while stack:
            curr, path, mask = stack.pop()
            for nb in adj[curr]:
                if nb == start:
                    # Cycle found
                    if len(path) < g:
                        # Only record if start is the min vertex to avoid duplicates
                        if min(path) == start:
                            cycles.append(path)
                elif not (mask & (1 << nb)):
                    # Only proceed if nb > start? No, just ensure we don't revisit.
                    # To enforce "start is min", we can prune if nb < start and nb != start?
                    # Actually, standard simple cycle enumeration:
                    # We just record if min(path + [start]) == start.
                    # But to be efficient, we can restrict neighbors to be > start 
                    # except when closing the cycle.
                    # Let's just collect all and deduplicate later or use min check.
                    if nb < start and nb != start:
                        continue
                    stack.append((nb, path + [nb], mask | (1 << nb)))
    
    # Deduplicate cycles (as sets of edges or tuples of vertices)
    # Represent cycle as frozenset of edges
    unique_cycles = set()
    for c in cycles:
        edges_in_cycle = []
        for i in range(len(c)):
            u = c[i]
            v = c[(i+1) % len(c)]
            if u > v: u, v = v, u
            edges_in_cycle.append((u, v))
        unique_cycles.add(frozenset(edges_in_cycle))
        
    cycles_list = list(unique_cycles)
    # Convert back to vertex lists for voltage summation
    # We need the sequence of vertices to sum voltages.
    # Actually, for voltage summation, we just need the edges and their orientation.
    # A cycle is a closed walk. Sum of voltages depends on direction.
    # If sum is 0 in one direction, it is 0 in the other.
    # We can represent a cycle by its edge set.
    # But to compute the sum, we need to traverse it.
    # Let's store the vertex list.
    # The cycles found above are vertex lists.
    # We need to ensure we have the actual vertex path.
    
    # Re-do cycle collection to keep vertex paths
    cycles_paths = []
    for start in range(n_base):
        stack = [(start, [start], 1 << start)]
        while stack:
            curr, path, mask = stack.pop()
            for nb in adj[curr]:
                if nb == start:
                    if len(path) < g:
                        if min(path) == start:
                            cycles_paths.append(path)
                elif not (mask & (1 << nb)):
                    if nb < start:
                        continue
                    stack.append((nb, path + [nb], mask | (1 << nb)))
    
    # Deduplicate paths (reverse paths are the same cycle)
    # Normalize: rotate so min vertex is first, and second vertex is smaller than last?
    def normalize_cycle(path):
        min_idx = path.index(min(path))
        rotated = path[min_idx:] + path[:min_idx]
        # Check reverse
        reversed_rotated = list(reversed(rotated))
        # Compare lexicographically
        if tuple(reversed_rotated) < tuple(rotated):
            return tuple(reversed_rotated)
        return tuple(rotated)
    
    unique_paths = set()
    for p in cycles_paths:
        unique_paths.add(normalize_cycle(p))
    
    cycles_final = list(unique_paths)
    
    # Map edges to indices for voltage assignment
    edge_indices = {}
    edge_list = []
    for i, (u, v) in enumerate(petersen_edges):
        edge_indices[(u, v)] = i
        edge_indices[(v, u)] = i
        edge_list.append((u, v))
        
    # Function to check if a voltage assignment is valid
    def is_valid_assignment(voltages, m):
        for cycle in cycles_final:
            s = 0
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i+1) % len(cycle)]
                # Get edge index
                idx = edge_indices[(u, v)]
                # Determine direction
                # The edge in edge_list is (u, v) or (v, u)
                # If edge_list[idx] is (u, v), voltage is +voltages[idx]
                # If edge_list[idx] is (v, u), voltage is -voltages[idx]
                eu, ev = edge_list[idx]
                if eu == u and ev == v:
                    s += voltages[idx]
                else:
                    s -= voltages[idx]
            if s % m == 0:
                return False
        return True

    # Try different m
    best_result = None
    for m in range(2, 500):
        if time.time() > deadline:
            break
            
        # Try a few random assignments
        for _ in range(50):
            if time.time() > deadline:
                break
            voltages = [random.randint(0, m-1) for _ in range(len(petersen_edges))]
            if is_valid_assignment(voltages, m):
                # Construct the lifted graph
                n_lifted = n_base * m
                edges = []
                for (u, v) in petersen_edges:
                    idx = edge_indices[(u, v)]
                    vol = voltages[idx]
                    for i in range(m):
                        # Vertex (u, i) connects to (v, (i + vol) % m)
                        # And (v, j) connects to (u, (j - vol) % m) which is same edge
                        u_lifted = u * m + i
                        v_lifted = v * m + (i + vol) % m
                        edges.append((u_lifted, v_lifted))
                        
                # Verify girth
                # BFS from each vertex up to depth g//2 - 1
                # If no cycle found, girth >= g
                # Since it's a lift, girth is determined by base cycles.
                # We already checked base cycles.
                # But to be safe, run a quick check.
                if _verify_girth(n_lifted, edges, g):
                    return n_lifted, edges
                # If verification fails (shouldn't happen), continue searching

    # Fallback to LUW if no small lift found
    return _luw_construct(k, g)

def _verify_girth(n, edges, g):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each vertex
    # We only need to check up to depth g//2
    max_depth = g // 2 - 1
    for start in range(n):
        visited = [-1] * n
        parent = [-1] * n
        dist = [0] * n
        queue = [start]
        visited[start] = start
        for v in queue:
            if dist[v] >= max_depth:
                continue
            for nb in adj[v]:
                if visited[nb] == -1:
                    visited[nb] = v
                    parent[nb] = v
                    dist[nb] = dist[v] + 1
                    queue.append(nb)
                elif visited[nb] != v and visited[v] != nb:
                    # Found a cycle
                    # Calculate length
                    # The cycle is v -> ... -> nb
                    # Length = dist[v] + dist[nb] + 1
                    # But we need to be careful with the BFS tree.
                    # If we find a back edge, we have a cycle.
                    # The length is dist[v] + dist[nb] + 1.
                    # If this length < g, we return False.
                    cycle_len = dist[v] + dist[nb] + 1
                    if cycle_len < g:
                        return False
    return True

def _luw_construct(k, g):
    # Original seed code logic
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

# EVOLVE-BLOCK-END
