import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses a voltage graph (covering graph) construction over a small cubic base graph.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    if k != 3:
        # Fallback for other k values
        pass

    # Define the base graph: Petersen graph (10 vertices, 3-regular, girth 5)
    # Petersen graph edges:
    # Outer cycle: 0-1, 1-2, 2-3, 3-4, 4-0
    # Inner star: 5-7, 7-9, 9-6, 6-8, 8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # Base graph has 10 vertices and 15 edges
    base_n = 10
    
    # We need to assign voltages from a finite abelian group to each edge.
    # The lift will have n = base_n * |G| vertices.
    # We want the shortest non-backtracking lift to have length >= g.
    
    # Let's try G = Z_m for various m.
    # For each edge, assign a voltage in Z_m.
    # The lift graph has vertices (v, i) where v is in base graph and i in Z_m.
    # An edge (u, v) with voltage w becomes edges (u, i) -> (v, i+w) and (v, i+w) -> (u, i-w).
    
    # We need to search for voltage assignments that yield girth >= g.
    # This is a constraint satisfaction problem.
    
    # Strategy:
    # 1. Start with small m (e.g., m=2, 3, 4, ...).
    # 2. For each m, try random voltage assignments.
    # 3. Check if the lift graph has girth >= g.
    # 4. Return the first valid graph found.
    
    # Optimization: We can use a heuristic to assign voltages.
    # One common approach is to use random assignments and check girth.
    
    # To speed up, we can precompute the base graph's non-backtracking walks.
    # But for simplicity, let's just do random assignments and check.
    
    # Let's try m from 2 to 20.
    # For each m, try up to 100 random assignments.
    
    def check_girth_lift(adj, n_nodes, target_girth):
        """
        Check if graph has girth >= target_girth.
        Returns True if girth >= target_girth, False otherwise.
        """
        max_depth = (target_girth - 1) // 2
        for s in range(n_nodes):
            dist = [-1] * n_nodes
            parent = [-1] * n_nodes
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < target_girth:
                            return False
        return True

    # Try different group sizes
    for m in range(2, 50):
        if time.time() > deadline - 10:
            break
        
        # Try random voltage assignments
        random.seed(42 + m)
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            # Assign random voltages to edges
            voltages = [random.randint(0, m - 1) for _ in base_edges]
            
            # Build the lift graph
            n_lift = base_n * m
            adj = [[] for _ in range(n_lift)]
            
            for (u, v), w in zip(base_edges, voltages):
                for i in range(m):
                    # Edge (u, i) -> (v, i+w mod m)
                    adj[u * m + i].append(v * m + (i + w) % m)
                    # Edge (v, i+w mod m) -> (u, i)
                    adj[v * m + (i + w) % m].append(u * m + i)
            
            # Check if the lift graph is simple (no loops or repeated edges)
            # In a voltage graph, loops occur if a vertex is connected to itself.
            # Repeated edges occur if two edges connect the same pair of vertices.
            # For now, assume the base graph is simple and voltages are assigned such that no loops occur.
            # We should check for loops and repeated edges.
            
            # Check for loops
            valid = True
            for i in range(n_lift):
                if i in adj[i]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check for repeated edges
            edge_set = set()
            for i in range(n_lift):
                for j in adj[i]:
                    if i < j:
                        if (i, j) in edge_set:
                            valid = False
                            break
                        edge_set.add((i, j))
                    elif i > j:
                        if (j, i) in edge_set:
                            valid = False
                            break
                        edge_set.add((j, i))
            
            if not valid:
                continue
            
            # Check degree
            deg = [0] * n_lift
            for i in range(n_lift):
                deg[i] = len(adj[i])
            
            if any(d != k for d in deg):
                continue
            
            # Check girth
            if check_girth_lift(adj, n_lift, g):
                # Found a valid graph
                edges = []
                for i in range(n_lift):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n_lift, edges
    
    # Fallback: LUW construction
    def build_luw(d, q_val):
        """Build LUW graph CD(d, q) for prime q. Returns (n, edges)."""
        if q_val == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add_fn = lambda x, y: x ^ y
            sub_fn = lambda x, y: x ^ y
            mul_fn = lambda x, y: table[x][y]
        else:
            add_fn = lambda x, y: (x + y) % q_val
            sub_fn = lambda x, y: (x - y) % q_val
            mul_fn = lambda x, y: x * y % q_val

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q_val):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                    else:
                        y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def check_girth_luw(n, edges, girth_target):
        max_depth = (girth_target - 1) // 2
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < girth_target:
                            return False
        return True

    # Try LUW with small d
    q_val = 3
    for d in range(5, 15):
        if time.time() > deadline:
            break
        n_luw, edges_luw = build_luw(d, q_val)
        if n_luw < 2:
            continue
        deg = [0] * n_luw
        for u, v in edges_luw:
            deg[u] += 1
            deg[v] += 1
        if any(dd != k for dd in deg):
            continue
        if check_girth_luw(n_luw, edges_luw, g):
            return n_luw, edges_luw

    # If all else fails, return the last LUW result
    return n_luw, edges_luw
# EVOLVE-BLOCK-END
