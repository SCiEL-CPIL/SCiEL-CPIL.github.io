"""Construct small 4-regular graphs of high girth via voltage graph lifts."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction (fallback)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
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


def _verify(n, edges, k, g):
    """Verify graph is k-regular, simple, and has girth >= g.
    Uses non-backtracking walks from every vertex."""
    # Check simple and degree
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
    for i in range(n):
        if len(adj[i]) != k:
            return False

    # Girth check via non-backtracking walks from every vertex
    # A graph has girth >= g iff for every vertex, the non-backtracking
    # BFS finds no cycle of length < g.
    # We do BFS from each vertex, tracking (node, prev) states,
    # and detect cycles when we revisit a state or hit a node at distance
    # that would form a short cycle.
    # Simpler: BFS from each vertex; when we see an edge (u,v) where
    # dist[u] + dist[v] + 1 < g and v != parent[u] and u != parent[v],
    # we have a cycle of length dist[u]+dist[v]+1.
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True


def _petersen_lift(m):
    """Lift the Petersen graph with voltage group Z_m to get a 4-regular graph.
    
    The Petersen graph has 10 vertices and 15 edges. Each edge gets a voltage
    in Z_m. The lift has 10*m vertices.
    
    We need to choose voltages so that the lift has girth >= 10.
    The Petersen graph has girth 5, so we need the voltages to "spread out"
    the short cycles.
    """
    # Petersen graph vertices: 0-4 (outer 5-cycle), 5-9 (inner 5-pointed star)
    # Outer cycle: 0-1-2-3-4-0
    # Spokes: i - (i+5) for i in 0..4
    # Inner star: 5-7-9-6-8-5 (connecting i+5 to (i+2)%5+5)
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer cycle
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),  # spokes
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # inner star
    ]
    
    # We need to assign voltages to these 15 edges in Z_m such that
    # the lift has girth >= 10.
    # 
    # The Petersen graph has cycles of length 5 (the outer and inner 5-cycles)
    # and triangles? No, Petersen has girth 5.
    # Actually Petersen has no triangles, girth is 5.
    # Cycles of length 5: outer 5-cycle, inner 5-cycle, and several others.
    # 
    # For the lift to have girth >= 10, every closed walk of length < 10 in the
    # base graph must have non-zero total voltage (so it doesn't close in the lift).
    # 
    # The shortest cycles in Petersen are length 5. There are 12 distinct 5-cycles.
    # We need: for every 5-cycle, the sum of voltages (with sign based on direction)
    # is non-zero mod m.
    # 
    # Let me enumerate the 5-cycles in Petersen:
    # 1. Outer: 0-1-2-3-4-0
    # 2. Inner: 5-7-9-6-8-5
    # 3. 0-1-6-8-5-0
    # 4. 0-4-9-6-5-0  (wait, let me be more careful)
    # 
    # Let me just enumerate all 5-cycles programmatically.
    
    adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find all 5-cycles (as sets of edges, up to rotation and direction)
    def find_5_cycles():
        cycles = set()
        for start in range(10):
            # DFS to find paths of length 4 starting and ending at start
            def dfs(path, visited):
                if len(path) == 5:
                    if path[-1] == start:
                        # Normalize: sort edges
                        edge_tuple = tuple(sorted((min(path[i], path[i+1]), max(path[i], path[i+1])) for i in range(4)))
                        cycles.add(edge_tuple)
                    return
                u = path[-1]
                for v in adj[u]:
                    if v == start and len(path) < 4:
                        continue
                    if v not in visited:
                        visited.add(v)
                        path.append(v)
                        dfs(path, visited)
                        path.pop()
                        visited.remove(v)
            dfs([start], {start})
        return list(cycles)
    
    five_cycles = find_5_cycles()
    
    # For each 5-cycle, we need the signed sum of voltages to be non-zero mod m.
    # Let's assign variables to the 15 edges and find a valid assignment.
    # 
    # For a cycle v0-v1-v2-v3-v4-v0, the voltage sum is:
    #   volt(v0,v1) + volt(v1,v2) + volt(v2,v3) + volt(v3,v4) + volt(v4,v0)
    # where volt(u,v) = voltage if edge is oriented u->v, or -voltage if v->u.
    # 
    # We'll orient each edge consistently (smaller vertex first) and track signs.
    
    edge_index = {}
    for i, (u, v) in enumerate(petersen_edges):
        edge_index[(min(u, v), max(u, v))] = i
    
    # For each 5-cycle, compute the signed coefficient vector
    constraints = []
    for cycle in five_cycles:
        coeffs = [0] * 15
        for i in range(4):
            u, v = cycle[i]
            e = (min(u, v), max(u, v))
            idx = edge_index[e]
            # Determine sign: if the cycle goes u->v and e is (u,v), sign +1
            # If cycle goes v->u and e is (u,v), sign -1
            if u < v:
                coeffs[idx] += 1
            else:
                coeffs[idx] -= 1
        # The last edge v4->v0
        u, v = cycle[3], cycle[0]  # Wait, cycle has 5 edges: (v0,v1),(v1,v2),(v2,v3),(v3,v4),(v4,v0)
        # Actually my cycle representation is 4 edges? No, I stored 4 edges but a 5-cycle has 5 edges.
        # Let me fix: path has 5 vertices, so 4 edges in the path, plus the closing edge.
        # Actually path = [v0, v1, v2, v3, v4] and we check path[-1] == start means v4 == v0.
        # So the edges are (v0,v1), (v1,v2), (v2,v3), (v3,v4) where v4=v0.
        # That's only 4 edges! I need 5 edges for a 5-cycle.
        # Let me redo: path should have 5 edges, so 6 vertices with first=last.
        
        break  # This is getting complicated, let me use a different approach
    
    # Simpler approach: just try random voltage assignments and check the lift.
    best_n = float('inf')
    best_edges = None
    
    start_time = time.time()
    deadline = start_time + 105
    
    for attempt in range(2000):
        if time.time() > deadline:
            break
        
        random.seed(attempt * 7919 + 13)
        
        # Assign random voltages
        voltages = [random.randint(0, m - 1) for _ in range(15)]
        
        # Build the lift
        n = 10 * m
        edges = []
        for i, (u, v) in enumerate(petersen_edges):
            vol = voltages[i]
            for x in range(m):
                # Edge from (u, x) to (v, x + vol) mod m
                # Also the reverse: (v, x) to (u, x - vol) mod m
                # In the lift, each base edge gives m edges in each direction,
                # but we only add each undirected edge once.
                a = (u, x)
                b = (v, (x + vol) % m)
                edges.append((u * m + x, v * m + ((x + vol) % m)))
        
        # Deduplicate (shouldn't be needed if construction is correct)
        # Check: for edge (u,v) with voltage vol, we add (u,x)-(v,x+vol) for all x.
        # The reverse edge (v,u) with voltage -vol would add (v,x)-(u,x-vol),
        # which is the same as (u,x+vol)-(v,x). So no duplicates.
        
        # Verify
        if _verify(n, edges, 4, 10):
            if n < best_n:
                best_n = n
                best_edges = edges
            break  # Found a valid one with this m
    
    if best_edges is not None:
        return best_n, best_edges
    
    return None


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    if k == 4 and g == 10:
        # Try voltage graph lift of Petersen with Z_m
        # m=32 gives 320 vertices
        for m in [32, 28, 24, 20, 16, 30, 26, 22, 18, 34, 36, 40]:
            if time.time() > deadline:
                break
            result = _petersen_lift(m)
            if result is not None:
                n, edges = result
                # Double-check
                if _verify(n, edges, k, g):
                    return n, edges
    
    # Fallback: LUW graph
    if time.time() > deadline:
        # Return LUW even if slow
        pass
    
    n, edges = _luw_construct(k, g)
    return n, edges

# EVOLVE-BLOCK-END
