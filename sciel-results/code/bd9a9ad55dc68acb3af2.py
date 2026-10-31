"""Construct a 3-regular graph with girth >= g on as few vertices as possible."""
import random
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 140  # 10s safety margin
    
    # Petersen graph definition
    # Vertices 0..9
    # Outer pentagon: 0-1-2-3-4-0
    # Inner star: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # Build adjacency for Petersen
    pet_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        pet_adj[u].append(v)
        pet_adj[v].append(u)
        
    # Find all simple cycles of length L in Petersen
    def find_cycles(adj, max_len):
        cycles = []
        n = len(adj)
        for start in range(n):
            # DFS to find cycles starting and ending at start
            # To avoid duplicates, we only consider cycles where start is the smallest vertex
            # and the second vertex is smaller than the last? 
            # Simpler: just find all and use a set of frozensets of edges?
            # Or just find all and deduplicate by canonical form.
            stack = [(start, [start])]
            while stack:
                node, path = stack.pop()
                if len(path) > max_len:
                    continue
                for neighbor in adj[node]:
                    if neighbor == path[0] and len(path) >= 3:
                        # Found a cycle
                        # Canonicalize: rotate so min vertex is first, then reverse if needed
                        cyc = path[:]
                        min_idx = cyc.index(min(cyc))
                        cyc = cyc[min_idx:] + cyc[:min_idx]
                        # Check if reversed is "smaller"
                        rev = cyc[::-1]
                        if rev < cyc:
                            cyc = rev
                        cycles.append(tuple(cyc))
                    elif neighbor not in path:
                        stack.append((neighbor, path + [neighbor]))
        # Remove duplicates
        unique_cycles = list(set(cycles))
        return unique_cycles

    # We need cycles of length < g
    # For g=16, we need cycles of length 5, 6, ..., 15
    # Petersen has girth 5.
    # Let's find all cycles up to length 15
    short_cycles = find_cycles(pet_adj, 15)
    
    def check_girth(n, adj, k, g):
        """Return True if girth >= g, False otherwise. Fast check."""
        radius = g // 2 - 1
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            count = [0] * (radius + 1)
            count[0] = 1
            while queue:
                u = queue.popleft()
                if dist[u] >= radius:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] <= radius:
                            count[dist[v]] += 1
                            queue.append(v)
            
            expected = [1] + [k * (k-1)**i for i in range(radius)]
            if count != expected:
                return False
        return True

    def build_lift(m, voltages):
        """
        Build the lift of Petersen with voltage assignment.
        Vertices are (base_vertex, offset) where offset in 0..m-1.
        Edge (u, v) in base with voltage w becomes edges:
        (u, i) -- (v, (i + w) % m)
        """
        n = 10 * m
        adj = [[] for _ in range(n)]
        
        # Map base edge index to voltage
        # We need to ensure we use the same voltage for the edge in both directions?
        # In a voltage graph, the voltage of (v, u) is -voltage(u, v).
        # Our edges are undirected. We assign a voltage to each undirected edge.
        # If we traverse u->v, we add w. If v->u, we add -w (mod m).
        
        # Let's assign voltages to the 15 edges.
        # petersen_edges list has 15 tuples.
        # voltages is a list of 15 ints.
        
        for idx, (u, v) in enumerate(petersen_edges):
            w = voltages[idx]
            for i in range(m):
                # Edge from (u, i) to (v, (i + w) % m)
                u_lift = u * m + i
                v_lift = v * m + ((i + w) % m)
                adj[u_lift].append(v_lift)
                
                # Edge from (v, i) to (u, (i - w) % m)
                # This is the same undirected edge.
                # We just need to add it once to the adjacency lists.
                # adj is a list of lists. We append to both.
                # Wait, the loop above adds u_lift -> v_lift.
                # We also need v_lift -> u_lift.
                adj[v_lift].append(u_lift)
                
        return adj

    # Try different m values
    # Lower bound is 512. n = 10m.
    # m=51 -> 510 (too small if LB is 512)
    # m=52 -> 520
    # m=53 -> 530
    # ...
    
    # We will try m from 52 upwards.
    # For each m, we try random voltage assignments.
    
    # To speed up, we can pre-check voltage sums for short cycles.
    # For a cycle C = (v0, v1, ..., v_{L-1}, v0), the sum of voltages must be non-zero mod m.
    # We map each edge in the cycle to its index in petersen_edges.
    
    edge_index_map = {}
    for idx, (u, v) in enumerate(petersen_edges):
        edge_index_map[(u, v)] = idx
        edge_index_map[(v, u)] = idx
        
    # Pre-compute the edge indices for each short cycle
    cycle_edge_indices = []
    for cyc in short_cycles:
        indices = []
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i + 1) % len(cyc)]
            idx = edge_index_map[(u, v)]
            indices.append(idx)
        cycle_edge_indices.append(indices)
        
    for m in range(52, 120):
        if time.time() > deadline:
            break
        n = 10 * m
        if n < 512:
            continue
            
        # Try random voltage assignments
        attempts = 0
        max_attempts = 200
        while attempts < max_attempts:
            attempts += 1
            if time.time() > deadline:
                break
                
            # Generate random voltages in 1..m-1 (avoid 0 to increase chance of non-zero sums)
            voltages = [random.randint(1, m-1) for _ in range(15)]
            
            # Check if any short cycle has sum 0 mod m
            valid_voltages = True
            for indices in cycle_edge_indices:
                s = 0
                for idx in indices:
                    s += voltages[idx]
                if s % m == 0:
                    valid_voltages = False
                    break
                    
            if not valid_voltages:
                continue
                
            # Construct the graph
            adj = build_lift(m, voltages)
            
            # Verify girth
            if check_girth(n, adj, k, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
                
    # Fallback: LUW or Random Search
    # If we reach here, we didn't find a lift.
    # Try a simple random search for small n just in case, or return a known valid graph if we have one.
    # Since 520-600 is very likely to work with the lift method, this fallback is rarely needed.
    
    # Simple fallback: Return a large LUW graph to ensure validity, though score will be worse.
    # Or just return a random regular graph if we can find one with high girth (unlikely).
    # Let's try a few random regular graphs of size 512, 520, etc.
    for n in range(512, 600, 8):
        if time.time() > deadline:
            break
        if n * k % 2 != 0:
            continue
        for _ in range(50):
            if time.time() > deadline:
                break
            # Configuration model
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, n * k, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if not valid:
                continue
            # Check connectivity
            visited = [False] * n
            queue = deque([0])
            visited[0] = True
            count = 1
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        count += 1
                        queue.append(v)
            if count != n:
                continue
            if check_girth(n, adj, k, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges

    # Absolute fallback: return a trivially valid but large graph?
    # The task requires a valid graph. If we fail, we should return something.
    # But the lift method is very strong.
    # Let's just return the first valid lift we find, or raise an error if none.
    # But we must return a valid graph.
    # If we get here, it's a failure.
    # Let's try m=100, 150, 200 with more attempts if we didn't find one earlier.
    for m in range(120, 300, 10):
        if time.time() > deadline:
            break
        n = 10 * m
        for _ in range(10):
            if time.time() > deadline:
                break
            voltages = [random.randint(1, m-1) for _ in range(15)]
            valid_voltages = True
            for indices in cycle_edge_indices:
                s = 0
                for idx in indices:
                    s += voltages[idx]
                if s % m == 0:
                    valid_voltages = False
                    break
            if not valid_voltages:
                continue
            adj = build_lift(m, voltages)
            if check_girth(n, adj, k, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges

    # If all else fails, return a dummy valid graph (e.g. a large LUW)
    # This part is just to prevent crashing.
    q = k
    d = g - 5 if (g - 5) % 2 else g - 4
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
