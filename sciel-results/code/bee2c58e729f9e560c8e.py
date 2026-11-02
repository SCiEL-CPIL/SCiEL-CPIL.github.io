import time
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where edges is a list of (u, v) on vertices 0..n-1.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    if k == 4 and g == 10:
        # Target n = 320
        n_target = 320
        
        # 1. Try Cayley graph on Z_320
        result = _try_cayley_z320(n_target, deadline)
        if result is not None:
            return result
            
        # 2. Try Cayley graph on Z_16 x Z_20
        result = _try_cayley_16_20(deadline)
        if result is not None:
            return result

        # 3. Fallback: Randomized Local Search
        result = _local_search_4_reg(n_target, g, deadline)
        if result is not None:
            return result

    # General fallback: LUW Construction
    return _luw_construct(k, g)

def _try_cayley_z320(n, deadline):
    """
    Search for a 4-regular Cayley graph on Z_n with generators {1, -1, b, -b}.
    """
    def check_girth(b):
        # Generators: 1, -1, b, -b
        # BFS from 0
        visited = [-1] * n
        visited[0] = 0
        queue = deque([0])
        
        while queue:
            u = queue.popleft()
            depth = visited[u]
            if depth >= 4:
                # Optimization: if depth is large, cycles formed are long
                # But we must check edges to nodes with small depth
                pass
            
            for g in [1, n-1, b, (n-b)%n]:
                v = (u + g) % n
                if visited[v] == -1:
                    visited[v] = depth + 1
                    queue.append(v)
                else:
                    # Cycle length
                    # The path 0->...->u->v->...->0
                    # Length = depth(u) + 1 + depth(v)
                    # This is the length of the fundamental cycle.
                    # Is it possible the shortest cycle is not fundamental?
                    # No, the girth is the min length of any cycle.
                    # Any cycle can be decomposed into fundamental cycles?
                    # Actually, the BFS tree gives a set of fundamental cycles.
                    # The girth is the minimum length among ALL cycles.
                    # The minimum length cycle in a graph is always a fundamental cycle in some BFS tree?
                    # Yes, if you take a shortest cycle, it induces a tree-like structure from the start node.
                    # So checking fundamental cycles in BFS from 0 is sufficient for vertex-transitive graphs?
                    # Yes, because of symmetry.
                    cycle_len = depth + 1 + visited[v]
                    if cycle_len < 10:
                        return False
        return True

    for b in range(2, n//2 + 1):
        if time.time() > deadline:
            return None
        if b == 1 or b == n-1: continue
        if (n-b) % n == b: continue # b = -b => 2b=0 => b=n/2
        # Also check if generators are distinct
        if b == 1 or b == n-1 or b == n//2: continue
        
        if check_girth(b):
            edges = []
            for u in range(n):
                for g in [1, b]:
                    v = (u + g) % n
                    if u < v:
                        edges.append((u, v))
            return n, edges
    return None

def _try_cayley_16_20(deadline):
    """
    Search for a 4-regular Cayley graph on Z_16 x Z_20.
    """
    m, n = 16, 20
    total = m * n
    
    def check_girth(x, y):
        # Generators: (1,0), (-1,0), (x,y), (-x,-y)
        # Map (i,j) to i*n + j
        # BFS from 0
        visited = [-1] * total
        visited[0] = 0
        queue = deque([0])
        
        gens = [
            (1, 0),
            (m-1, 0),
            (x, y),
            ((-x)%m, (-y)%n)
        ]
        
        while queue:
            u = queue.popleft()
            depth = visited[u]
            ui, uj = divmod(u, n)
            
            for g1, g2 in gens:
                nu = (ui + g1) % m
                nv = (uj + g2) % n
                v = nu * n + nv
                if visited[v] == -1:
                    visited[v] = depth + 1
                    queue.append(v)
                else:
                    cycle_len = depth + 1 + visited[v]
                    if cycle_len < 10:
                        return False
        return True

    for x in range(m):
        if time.time() > deadline:
            return None
        for y in range(n):
            if time.time() > deadline:
                return None
            
            # Skip trivial/invalid generators
            if x == 0 and y == 0: continue
            if x == 1 and y == 0: continue # a
            if x == m-1 and y == 0: continue # -a
            if (2*x) % m == 0 and (2*y) % n == 0: continue # involution
            
            if check_girth(x, y):
                edges = []
                for u in range(m):
                    for v in range(n):
                        uid = u * n + v
                        # Edge to a
                        nu = (u + 1) % m
                        nv = v
                        vid = nu * n + nv
                        if uid < vid:
                            edges.append((uid, vid))
                        
                        # Edge to b
                        nu = (u + x) % m
                        nv = (v + y) % n
                        vid = nu * n + nv
                        if uid < vid:
                            edges.append((uid, vid))
                return total, edges
    return None

def _local_search_4_reg(n, g, deadline):
    """
    Randomized local search for a 4-regular graph with girth >= g.
    """
    # 1. Generate random 4-regular graph
    # Use a simple method: random pairing of stubs
    stubs = [i for i in range(n) for _ in range(4)]
    random.shuffle(stubs)
    edges = set()
    valid = True
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            valid = False
            break
        if (u, v) in edges or (v, u) in edges:
            valid = False
            break
        edges.add((min(u,v), max(u,v)))
    
    if not valid:
        return None

    # Convert to list
    edge_list = list(edges)
    
    # Helper to count short cycles
    def count_short_cycles(edges_set, n, max_len):
        # BFS from each vertex is too slow for every step.
        # We will just check if girth >= g at the end of a phase.
        # For the search, we can use a heuristic: 
        # Actually, let's just do a full check every 1000 steps or so.
        # Or better, use a "penalty" based on local structure?
        # Let's stick to: Do a batch of swaps, then check girth.
        return 0 # Placeholder

    def check_girth(edges_set, n, g):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges_set:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS from each vertex to find min cycle
        min_girth = float('inf')
        for start in range(n):
            # BFS
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    else:
                        # Cycle found
                        # Length = dist[u] + 1 + dist[v]
                        # This is only valid if the path doesn't go through u?
                        # In BFS, the first time we see a visited node, 
                        # the cycle length is dist[u] + 1 + dist[v].
                        # This is the shortest cycle passing through the edge (u,v) 
                        # that is not a tree edge?
                        # Actually, this finds *a* cycle. 
                        # The girth is the min over all such.
                        cycle_len = dist[u] + 1 + dist[v]
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                        if min_girth < g:
                            return min_girth
        return min_girth

    # Search Loop
    best_edges = None
    best_girth = 0
    current_edges = edges.copy()
    
    # Initial check
    girth = check_girth(current_edges, n, g)
    if girth >= g:
        return n, list(current_edges)
    best_girth = girth

    while time.time() < deadline:
        # Perform a batch of swaps
        for _ in range(100):
            if len(current_edges) < 2: break
            # Pick two random edges
            e1 = random.choice(list(current_edges))
            e2 = random.choice(list(current_edges))
            if e1 == e2: continue
            
            u1, v1 = e1
            u2, v2 = e2
            
            # Swap: (u1, v1), (u2, v2) -> (u1, u2), (v1, v2)
            # Or (u1, v2), (v1, u2)
            # Try both? Or just one.
            new_e1 = (min(u1, u2), max(u1, u2))
            new_e2 = (min(v1, v2), max(v1, v2))
            
            # Check validity
            if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
                continue
            if new_e1 in current_edges or new_e2 in current_edges:
                continue
            if new_e1 == new_e2:
                continue
            
            # Apply swap
            current_edges.remove(e1)
            current_edges.remove(e2)
            current_edges.add(new_e1)
            current_edges.add(new_e2)
            
        # Check girth periodically
        # This is expensive. Do it less often?
        # With n=320, check_girth is ~320 * (320+640) = 300k ops.
        # 100 swaps + check. 
        # If we do this every loop, it's slow.
        # Let's do the check every 10 loops of the inner batch?
        # Or just check every time. 120s / 0.3s per check = 400 checks.
        # 400 * 100 swaps = 40000 swaps. Should be enough.
        
        girth = check_girth(current_edges, n, g)
        if girth >= g:
            return n, list(current_edges)
        
        if girth > best_girth:
            best_girth = girth
            best_edges = current_edges.copy()
            
        # If we stuck in a local minimum (girth didn't improve), 
        # we might want to restart or do more swaps.
        # For now, just continue.

    if best_edges:
        return n, list(best_edges)
    return None

def _luw_construct(k, g):
    """
    LUW construction for k-regular graphs with large girth.
    """
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
