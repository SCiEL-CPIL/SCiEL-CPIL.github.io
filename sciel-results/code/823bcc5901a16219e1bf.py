import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin

    # Fallback: LUW Graph (as in current program, but we hope not to use it if we find better)
    def get_luw(k, g):
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

    # Fast Girth Check: Returns the length of the shortest cycle
    def get_girth(n, adj):
        # Use BFS from each vertex, but optimized
        # For small n, this is fine.
        # To be faster, we can stop BFS early if we find a cycle < current best.
        min_cycle = float('inf')
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            while queue:
                u = queue.pop(0)
                if dist[u] >= min_cycle // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v == parent[u]:
                            continue
                        # Cycle found
                        # We need to calculate the length of the cycle formed by u-v edge
                        # This is complex to do exactly in BFS without reconstructing path.
                        # However, for "is girth < g", we can just check if any non-tree edge
                        # creates a cycle < g.
                        # A simpler check: if we find an edge (u,v) where dist[u] + dist[v] + 1 < g,
                        # it doesn't necessarily mean the *shortest* cycle is that, but it's a candidate.
                        # Actually, the cycle length is at least dist[u] + dist[v] + 1?
                        # No, the path from s to u and s to v might share nodes.
                        # The true cycle length is length(LCA path) + ...
                        # This BFS approach for exact girth is tricky.
                        
                        # Alternative: Since g is small (10), we can just do a bounded BFS/DFS.
                        pass
        # Let's use a simpler, exact method for small g:
        # Check all non-backtracking walks of length < g.
        # This is what the current program does for circulants.
        # Let's adapt that for general graphs.
        
        # State: (node, prev_node)
        # Start: for each node s, for each neighbor n1, state (n1, s) at depth 1.
        # If we reach (s, n2) at depth L, we have a cycle of length L+1.
        # We want to know if any cycle length < g exists.
        
        # Optimization: We only need to check if a cycle of length < g exists.
        # We can run BFS from each (s, n1) up to depth g-2.
        # If we hit s, we found a cycle.
        
        for s in range(n):
            for n1 in adj[s]:
                # BFS from (n1, s)
                visited = set()
                queue = [(n1, s, 1)]
                visited.add((n1, s))
                while queue:
                    u, prev, depth = queue.pop(0)
                    if depth >= g - 1:
                        continue
                    for v in adj[u]:
                        if v == prev:
                            continue
                        if v == s:
                            # Cycle of length depth + 1
                            if depth + 1 < g:
                                return depth + 1
                        elif (v, u) not in visited:
                            visited.add((v, u))
                            queue.append((v, u, depth + 1))
        return g # Or inf, but we just need to know if it's >= g

    # Random 4-regular graph generator
    def random_4_reg(n):
        if n < 4: return None
        if n % 2 != 0: return None
        # Use configuration model
        half_edges = [i for i in range(n) for _ in range(2)]
        random.shuffle(half_edges)
        adj = [[] for _ in range(n)]
        for i in range(0, 2*n, 2):
            u, v = half_edges[i], half_edges[i+1]
            if u == v:
                return None # Self loop, retry
            if v in adj[u]:
                return None # Multi-edge, retry
            adj[u].append(v)
            adj[v].append(u)
        return adj

    # 2-Switch operation
    # Given edges (u,v) and (x,y), replace with (u,x) and (v,y) or (u,y) and (v,x)
    # Must ensure no self-loops or multi-edges are created.
    def try_2_switch(adj, u, v, x, y):
        # Remove (u,v) and (x,y)
        adj[u].remove(v)
        adj[v].remove(u)
        adj[x].remove(y)
        adj[y].remove(x)
        
        # Try switch 1: (u,x) and (v,y)
        if x not in adj[u] and y not in adj[v] and u != x and v != y:
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            return True
        else:
            # Revert
            adj[u].append(v)
            adj[v].append(u)
            adj[x].append(y)
            adj[y].append(x)
            
        # Try switch 2: (u,y) and (v,x)
        if y not in adj[u] and x not in adj[v] and u != y and v != x:
            adj[u].append(y)
            adj[y].append(u)
            adj[v].append(x)
            adj[x].append(v)
            return True
        else:
            # Revert
            adj[u].append(v)
            adj[v].append(u)
            adj[x].append(y)
            adj[y].append(x)
            
        return False

    def improve_girth(n, adj, target_girth, max_iterations=1000):
        current_girth = get_girth(n, adj)
        if current_girth >= target_girth:
            return adj
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            # Find a short cycle
            # We need the actual cycle to break it.
            # get_girth only returns length. Let's modify it to return the cycle or just pick random edges.
            # Picking random edges is inefficient.
            # Let's find *a* short cycle.
            
            # Re-implement find_short_cycle to return the cycle
            def find_short_cycle(adj, max_len):
                for s in range(n):
                    for n1 in adj[s]:
                        visited = set()
                        queue = [(n1, s, 1, [s, n1])]
                        visited.add((n1, s))
                        while queue:
                            u, prev, depth, path = queue.pop(0)
                            if depth >= max_len - 1:
                                continue
                            for v in adj[u]:
                                if v == prev:
                                    continue
                                if v == s:
                                    if depth + 1 < max_len:
                                        return path + [s]
                                elif (v, u) not in visited:
                                    visited.add((v, u))
                                    queue.append((v, u, depth + 1, path + [v]))
                return None

            cycle = find_short_cycle(adj, target_girth)
            if not cycle:
                return adj # Girth is large enough
            
            # Break the cycle
            # Pick a random edge in the cycle
            # Cycle is a list of nodes. Edges are (cycle[i], cycle[i+1])
            # Let's pick a random edge (u,v) from the cycle.
            # And another random edge (x,y) in the graph.
            # This is a "random 2-switch" strategy.
            
            # Better: Pick two edges from the cycle to break it?
            # No, 2-switch breaks one edge and another.
            # If we pick an edge (u,v) in the cycle, and a random edge (x,y) elsewhere,
            # the cycle is broken.
            
            idx = random.randint(0, len(cycle) - 1)
            u = cycle[idx]
            v = cycle[(idx + 1) % len(cycle)]
            
            # Find a random other edge (x,y)
            # We need to ensure (x,y) is not in the cycle (to avoid complex interactions, though not strictly necessary)
            # Just pick any edge.
            all_edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        all_edges.append((i, j))
            
            if not all_edges:
                break
                
            x, y = random.choice(all_edges)
            
            # Ensure (x,y) is distinct from (u,v)
            if (u == x and v == y) or (u == y and v == x):
                continue
                
            if try_2_switch(adj, u, v, x, y):
                # Check if we improved
                pass # We just loop again

        return adj

    # Main Loop
    if k == 4 and g == 10:
        # Try n from lower bound upwards
        # Moore bound for k=4, g=10: 1 + 4 + 4*3 + 4*3^2 + 4*3^3 = 1 + 4 + 12 + 36 + 108 = 161?
        # Wait, Moore bound formula: 1 + k + k(k-1) + ...
        # For g=10 (even), radius 4.
        # 1 + 4 + 4*3 + 4*3^2 + 4*3^3 = 1 + 4 + 12 + 36 + 108 = 161.
        # So n >= 161.
        
        # Known best is 320.
        # Let's try n in range [162, 320]
        # But random 4-reg graphs for small n might not be able to reach girth 10.
        # Girth 10 requires n to be sufficiently large.
        
        # Let's try a few specific n values.
        for n in [320, 310, 300, 290, 280, 270, 260, 250, 243, 200, 180, 162]:
            if time.time() > deadline:
                break
            
            # Try to find a graph with this n
            best_adj = None
            for attempt in range(20):
                if time.time() > deadline:
                    break
                adj = random_4_reg(n)
                if adj is None:
                    continue
                
                # Improve
                adj = improve_girth(n, adj, g, max_iterations=50)
                
                if get_girth(n, adj) >= g:
                    best_adj = adj
                    break
            
            if best_adj:
                edges = []
                for u in range(n):
                    for v in best_adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges

    # Fallback
    return get_luw(k, g)
