import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    # Fallback to seed if we fail
    def seed_construct(k, g):
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

    def petersen_edges():
        """Return edge list of Petersen graph (10 vertices, 3-regular, girth 5)."""
        # Outer 5-cycle: 0-1-2-3-4-0
        # Inner star: 5-7-9-6-8-5
        # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
        edges = []
        for i in range(5):
            edges.append((i, (i + 1) % 5))
        edges.append((5, 7))
        edges.append((7, 9))
        edges.append((9, 6))
        edges.append((6, 8))
        edges.append((8, 5))
        for i in range(5):
            edges.append((i, 5 + i))
        return edges

    def build_lift(m, seed_val=42):
        """Build a 3-regular graph as a random lift of Petersen graph with m sheets.
        Each vertex (i, s) for i in 0..9, s in 0..m-1.
        For each Petersen edge (u,v), connect (u,s) to (v, (s + offset) % m) for some offset.
        """
        rng = random.Random(seed_val)
        n = 10 * m
        pet_edges = petersen_edges()
        
        # For each Petersen edge, assign a random offset
        offsets = [rng.randint(0, m - 1) for _ in range(len(pet_edges))]
        
        adj = [[] for _ in range(n)]
        for (u, v), off in zip(pet_edges, offsets):
            for s in range(m):
                a = u * m + s
                b = v * m + ((s + off) % m)
                adj[a].append(b)
                adj[b].append(a)
        
        # Verify 3-regular
        for i in range(n):
            if len(adj[i]) != 3:
                return None
        return adj

    def find_short_cycle_bfs(adj, n, g):
        """Find a cycle of length < g using BFS. Returns list of vertices or None."""
        for start in range(n):
            dist = [-1] * n
            prev = [-1] * n
            queue = [start]
            dist[start] = 0
            while queue:
                u = queue.pop(0)
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        prev[v] = u
                        queue.append(v)
                    elif v != prev[u] and v != start:
                        length = dist[u] + dist[v] + 1
                        if length < g:
                            path1 = []
                            curr = u
                            while curr != -1:
                                path1.append(curr)
                                if curr == start:
                                    break
                                curr = prev[curr]
                            path1.reverse()
                            path2 = []
                            curr = v
                            while curr != -1:
                                path2.append(curr)
                                if curr == start:
                                    break
                                curr = prev[curr]
                            return path1 + path2[1:]
                    elif v == start and dist[u] + 1 < g:
                        length = dist[u] + 1
                        if length < g:
                            path = []
                            curr = u
                            while curr != -1:
                                path.append(curr)
                                if curr == start:
                                    break
                                curr = prev[curr]
                            path.reverse()
                            return path
        return None

    def count_cycles_of_length(adj, n, target_len):
        """Count cycles of exactly target_len using BFS from each vertex."""
        count = 0
        for start in range(n):
            dist = [-1] * n
            prev = [-1] * n
            queue = [start]
            dist[start] = 0
            while queue:
                u = queue.pop(0)
                if dist[u] >= target_len // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        prev[v] = u
                        queue.append(v)
                    elif v != prev[u] and v != start:
                        length = dist[u] + dist[v] + 1
                        if length == target_len:
                            count += 1
        return count // 2  # Each cycle counted twice (once from each endpoint direction)

    def try_switch(adj, u, v, x, y):
        """Try to swap edges (u,v) and (x,y) to (u,x) and (v,y) or (u,y) and (v,x).
        Returns True if a valid switch was made."""
        # Check validity for switch 1: (u,x) and (v,y)
        valid1 = True
        if x == u or x == v or y == u or y == v:
            valid1 = False
        elif x in adj[u] or y in adj[v]:
            valid1 = False
        
        if valid1:
            # Perform switch 1
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            return True
        
        # Check validity for switch 2: (u,y) and (v,x)
        valid2 = True
        if y == u or y == v or x == u or x == v:
            valid2 = False
        elif y in adj[u] or x in adj[v]:
            valid2 = False
        
        if valid2:
            # Perform switch 2
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            adj[u].append(y)
            adj[y].append(u)
            adj[v].append(x)
            adj[x].append(v)
            return True
        
        return False

    def local_search(adj, n, g, time_budget):
        """Apply local search to eliminate cycles of length < g.
        Start with 14-cycles (since lift guarantees girth >= 14), then 15-cycles."""
        end_time = time.time() + time_budget
        rng = random.Random(12345)
        
        # First, find and eliminate 14-cycles
        target_len = g - 2  # Start with g-2 = 14
        for target in [g - 2, g - 1]:
            while time.time() < end_time:
                cycle = find_short_cycle_bfs(adj, n, target + 1)
                if cycle is None:
                    break
                if len(cycle) != target:
                    # Found a shorter cycle, handle it
                    pass
                
                # Pick two edges in the cycle to switch
                cycle_len = len(cycle)
                # Edges in cycle: (cycle[0], cycle[1]), (cycle[1], cycle[2]), ...
                # Pick edge at position 0 and a non-adjacent edge
                e1_u, e1_v = cycle[0], cycle[1]
                
                # Find a partner edge not in the cycle and not adjacent to e1
                in_cycle = set(cycle)
                all_edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            all_edges.append((i, j))
                
                # Filter: not in cycle, not sharing vertex with e1
                candidates = []
                for (a, b) in all_edges:
                    if a in in_cycle or b in in_cycle:
                        continue
                    if a == e1_u or a == e1_v or b == e1_u or b == e1_v:
                        continue
                    candidates.append((a, b))
                
                if not candidates:
                    break
                
                # Try a few random candidates
                tried = 0
                while tried < 50 and time.time() < end_time:
                    x, y = rng.choice(candidates)
                    if try_switch(adj, e1_u, e1_v, x, y):
                        tried += 1
                    else:
                        tried += 1
                        if tried >= 50:
                            break
        return adj

    # Main search
    best_n = None
    best_edges = None

    # Strategy: Try lifts of Petersen graph with various m values
    # m sheets -> 10*m vertices
    # Start with smaller m and increase
    # m=52 -> 520 vertices (below Moore bound, won't work)
    # m=60 -> 600 vertices
    # m=80 -> 800 vertices
    # m=94 -> 940 vertices (close to known 936)
    
    m_values = [60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96]
    
    for m in m_values:
        if time.time() > deadline - 10:
            break
        
        n = 10 * m
        # Try multiple seeds
        for seed_val in range(10):
            if time.time() > deadline - 5:
                break
            
            adj = build_lift(m, seed_val)
            if adj is None:
                continue
            
            # Quick check: does it have girth >= g?
            cycle = find_short_cycle_bfs(adj, n, g)
            if cycle is None:
                # Valid!
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
            
            # Has short cycles, try local search
            time_left = deadline - 5 - time.time()
            if time_left > 3:
                adj = local_search(adj, n, g, time_left)
                cycle = find_short_cycle_bfs(adj, n, g)
                if cycle is None:
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return n, edges
    
    # Fallback to seed
    return seed_construct(k, g)
