import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Heawood graph: 14 vertices, 21 edges
    # Standard construction: vertices are 2-subsets of {0,1,2,3,4} ? No, that's K5 complement?
    # Heawood is the incidence graph of the Fano plane.
    # Vertices: 7 points + 7 lines.
    # Let's just hardcode the edges.
    # Fano plane points: 0..6
    # Lines:
    # 0: 0,1,2
    # 1: 0,3,4
    # 2: 0,5,6
    # 3: 1,3,5
    # 4: 1,4,6
    # 5: 2,3,6
    # 6: 2,4,5
    # Vertices 0-6 are points, 7-13 are lines.
    # Edge (point, line) if point is on line.
    
    heawood_edges = []
    lines = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    for i, line in enumerate(lines):
        for p in line:
            heawood_edges.append((p, 7 + i))
            
    # Adjacency list for Heawood
    n_base = 14
    adj = [[] for _ in range(n_base)]
    for u, v in heawood_edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Enumerate non-backtracking closed walks of length 7 to g-1
    # g=16, so lengths 7..15
    walks = [] # List of lists of edge indices? Or just edge indices in order.
    # To check label sum, we need the sequence of edges.
    
    def gen_walks(start, current, prev, path, depth, max_depth):
        if depth > max_depth:
            return
        if depth > 0 and current == start:
            if depth >= 7:
                walks.append(path[:])
        for nb in adj[current]:
            if nb == prev:
                continue
            # Find edge index between current and nb
            # We need a map from (u,v) to edge index
            # Let's precompute edge_index map
            e_idx = edge_map.get((current, nb), edge_map.get((nb, current)))
            gen_walks(start, nb, current, path + [e_idx], depth + 1, max_depth)

    # Precompute edge map
    edge_map = {}
    for i, (u, v) in enumerate(heawood_edges):
        edge_map[(u, v)] = i
        edge_map[(v, u)] = i
        
    # Generate walks
    for s in range(n_base):
        gen_walks(s, s, -1, [], 0, g - 1)
        
    # Deduplicate? 
    # A closed walk is defined by its sequence of edges.
    # Different starts might generate the same walk rotated.
    # But for label sum, rotation doesn't matter (sum is commutative for +, XOR is commutative).
    # However, the walk must be non-backtracking.
    # If we find a walk with sum 0, it's a cycle in the lift.
    # We just need to check if ANY walk has sum 0.
    
    num_walks = len(walks)
    if num_walks == 0:
        # Fallback
        return _luw_graph(k, g)

    # Try random lifts
    rng = random.Random(42)
    
    # Try m from 4 to 10
    for m in range(4, 11):
        if time.time() > deadline - 10:
            break
        L = 1 << m
        n_lift = n_base * L
        
        # Try multiple random labelings
        for attempt in range(1000):
            if time.time() > deadline - 5:
                break
            
            # Random labels: m-bit integers
            labels = [rng.getrandbits(m) for _ in range(len(heawood_edges))]
            
            valid = True
            for walk in walks:
                # Compute XOR sum
                s = 0
                for e_idx in walk:
                    s ^= labels[e_idx]
                if s == 0:
                    valid = False
                    break
            
            if valid:
                # Construct the lift
                edges = []
                # Vertices are (base_vertex, group_element)
                # ID = base_vertex * L + group_element
                for e_idx, (u, v) in enumerate(heawood_edges):
                    label = labels[e_idx]
                    for h in range(L):
                        # Edge from (u, h) to (v, h ^ label)
                        a = u * L + h
                        b = v * L + (h ^ label)
                        if a < b:
                            edges.append((a, b))
                        else:
                            edges.append((b, a))
                
                # Verify girth? 
                # The algebraic check should be sufficient if the walks cover all cycles.
                # We enumerated all non-backtracking closed walks of length < g.
                # If none have sum 0, there are no cycles of length < g.
                # We should double check with a BFS girth checker on the lift for safety?
                # n_lift can be up to 14 * 1024 = 14336.
                # BFS girth check on 14k vertices is fast.
                
                if _check_girth_fast(adj_lift, g):
                    return n_lift, edges
                else:
                    # Should not happen if walk enumeration is correct
                    pass
                    
    # Fallback
    return _luw_graph(k, g)

def _check_girth_fast(adj, girth):
    """Check if the graph has no cycle shorter than girth using non-backtracking BFS."""
    n = len(adj)
    for start in range(n):
        from collections import deque
        queue = deque()
        visited = {}
        for nb in adj[start]:
            queue.append((nb, start, 1))
            visited[(nb, start)] = 1
        while queue:
            node, prev, dist = queue.popleft()
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb == start:
                    if dist + 1 < girth:
                        return False
                    continue
                key = (nb, node)
                if key in visited:
                    if visited[key] + dist + 1 < girth:
                        return False
                    continue
                visited[key] = dist
                queue.append((nb, node, dist + 1))
    return True

def _luw_graph(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
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

# Need to define adj_lift in construct or pass it
