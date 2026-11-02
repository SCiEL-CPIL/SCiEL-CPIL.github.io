import time
import random
import numpy as np

_DEADLINE = 110.0

def construct(k, g):
    _t0 = time.time()
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
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
            if time.time() - _t0 > _DEADLINE:
                break
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # If the task is not the specific (4,10) cage, use LUW
    if k != 4 or g != 10:
        return luw_construct(k, g)

    # Specialized search for (4,10)-cage
    n_target = 320
    
    def random_regular_graph(n, k):
        # Configuration model
        while True:
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            edges = []
            valid = True
            seen = set()
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if (u, v) in seen or (v, u) in seen:
                    valid = False
                    break
                seen.add((u, v))
                edges.append((u, v))
            if valid:
                adj = [[] for _ in range(n)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                return adj, edges

    def has_girth_at_least(n, adj, g):
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if dist[u] + dist[v] + 1 < g:
                            return False
        return True

    # Local Search
    deadline = _t0 + _DEADLINE
    best_adj = None
    best_edges = None
    
    # Try a few random starts
    for start_attempt in range(15):
        if time.time() > deadline:
            break
            
        adj, edges = random_regular_graph(n_target, k)
        
        # Check girth
        if has_girth_at_least(n_target, adj, g):
            return n_target, edges
            
        # Local Search
        for step in range(3000):
            if time.time() > deadline:
                break
                
            # Find a short cycle
            # We need a faster check. 
            # If we just want to break *some* short cycle, we can pick a random edge and see if it's in a short cycle?
            # Better: Pick a random vertex, do BFS to depth g/2. If we find a back edge, we have a short cycle.
            
            short_cycle_found = False
            # Pick a random start vertex to find a short cycle
            start = random.randint(0, n_target - 1)
            dist = [-1] * n_target
            parent = [-1] * n_target
            dist[start] = 0
            queue = [start]
            head = 0
            cycle_edges = []
            
            while head < len(queue) and not short_cycle_found:
                u = queue[head]
                head += 1
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if dist[u] + dist[v] + 1 < g:
                            # Found a short cycle
                            short_cycle_found = True
                            # Identify edges in the cycle to potentially break
                            # The cycle consists of path start->u, edge (u,v), path v->start
                            # We can try to break the edge (u, v) or any edge in the cycle.
                            # Let's try to break the edge (u, v) by swapping it with another edge.
                            # To keep 4-regular, we swap (u, v) with (u, x) and (v, y)?
                            # No, standard 2-opt: swap (u, v) and (w, z) to (u, w) and (v, z).
                            # We need to find w, z such that (w, z) is an edge, and (u, w), (v, z) are not edges.
                            # And (u, w), (v, z) are not loops.
                            
                            # Let's just try to break the edge (u, v).
                            # Pick a random neighbor w of u (w != v).
                            # Pick a random neighbor z of v (z != u).
                            # If (w, z) is an edge, and (u, w) is not, and (v, z) is not...
                            # Wait, (u, w) is an edge because w is a neighbor of u.
                            # We want to replace (u, v) and (w, z) with (u, w) and (v, z)?
                            # If we remove (u, v) and (w, z), we add (u, w) and (v, z).
                            # But (u, w) is already an edge. So we would create a multi-edge.
                            # We need to swap (u, v) with (u, w)? No, that's not a swap.
                            
                            # Correct move:
                            # Pick edge (u, v).
                            # Pick another edge (w, z).
                            # Remove (u, v) and (w, z).
                            # Add (u, w) and (v, z).
                            # Conditions:
                            # 1. (u, w) is not an edge.
                            # 2. (v, z) is not an edge.
                            # 3. u != w, v != z (no loops).
                            # 4. (u, w) != (v, z) and (u, w) != (z, v) (no multi-edges).
                            
                            # Let's try to find such a swap.
                            swapped = False
                            # Sample some edges
                            for _ in range(80):
                                # Pick random edge (w, z) from the graph
                                # We don't have a list of edges handy in this loop, but we have adj.
                                # Let's pick random w, then random neighbor z of w.
                                w = random.randint(0, n_target - 1)
                                if not adj[w]: continue
                                z = random.choice(adj[w])
                                # Ensure (w, z) is a valid edge
                                # Ensure (w, z) is not (u, v) or (v, u)
                                if (w == u and z == v) or (w == v and z == u):
                                    continue
                                    
                                # Check if we can swap (u, v) and (w, z) to (u, w) and (v, z)
                                # Or (u, z) and (v, w)
                                
                                # Option 1: (u, w) and (v, z)
                                if u != w and v != z:
                                    # Check if (u, w) exists
                                    if w not in adj[u] and z not in adj[v]:
                                        # Check if (u, w) creates multi-edge? No, we check existence.
                                        # Check if (v, z) creates multi-edge? No.
                                        # Check if (u, w) is loop? u != w.
                                        # Check if (v, z) is loop? v != z.
                                        
                                        # Perform swap
                                        adj[u].remove(v)
                                        adj[v].remove(u)
                                        adj[w].remove(z)
                                        adj[z].remove(w)
                                        
                                        adj[u].append(w)
                                        adj[w].append(u)
                                        adj[v].append(z)
                                        adj[z].append(v)
                                        
                                        swapped = True
                                        break
                                        
                                # Option 2: (u, z) and (v, w)
                                if u != z and v != w:
                                    if z not in adj[u] and w not in adj[v]:
                                        adj[u].remove(v)
                                        adj[v].remove(u)
                                        adj[w].remove(z)
                                        adj[z].remove(w)
                                        
                                        adj[u].append(z)
                                        adj[z].append(u)
                                        adj[v].append(w)
                                        adj[w].append(v)
                                        
                                        swapped = True
                                        break
                            
                            if not swapped:
                                # Try to break a different edge in the cycle?
                                # For now, just restart BFS to find another cycle or break this step.
                                pass
                            
            if short_cycle_found and not swapped:
                # If we couldn't swap, we might be stuck. 
                # But we can just continue to next step, hoping to find another cycle.
                pass

        # After local search, check again
        if has_girth_at_least(n_target, adj, g):
            # Reconstruct edges
            edges = []
            for u in range(n_target):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n_target, edges

    # Fallback
    return luw_construct(k, g)
