import random
import time
import sys
from collections import deque

def construct(k, g):
    # Target specific case for optimization
    if k == 4 and g == 10:
        # Try to find a 4-regular graph with girth 10 on N vertices
        # Known minimum is 320.
        # We will try N=320 first.
        
        def generate_random_k_regular(n, k):
            # Generate a random k-regular simple graph
            # Using configuration model
            while True:
                stubs = []
                for i in range(n):
                    stubs.extend([i] * k)
                random.shuffle(stubs)
                edges = []
                valid = True
                seen = set()
                for i in range(0, len(stubs), 2):
                    u, v = stubs[i], stubs[i+1]
                    if u == v:
                        valid = False
                        break
                    if u > v:
                        u, v = v, u
                    if (u, v) in seen:
                        valid = False
                        break
                    seen.add((u, v))
                    edges.append((u, v))
                if valid:
                    return edges
        
        def calculate_girth(n, edges):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            min_girth = float('inf')
            for start in range(n):
                # BFS
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                found = False
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif v != parent[u]:
                            # Cycle found
                            # Length = dist[u] + dist[v] + 1
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
                                # Early stop if we found a very short cycle?
                                # But we need the exact min.
                            # If we just want to know if < g, we can stop early
                            # But we need to find the cycle to break it.
                # Note: This BFS finds cycles passing through start?
                # Actually, standard girth calculation is more complex.
                # For small N, we can just check all pairs?
                # Or use the fact that girth is min over all nodes of shortest cycle containing it.
                # The BFS above finds the shortest cycle *encountered* during BFS from start.
                # This is not necessarily the global girth.
                # However, for a quick check, if we find a cycle < g, we know girth < g.
                # To get exact girth, we need to be careful.
                # Given N=320, we can afford a more expensive check if needed.
                # But for the search loop, we just need to know if girth < 10.
                # If min_girth found so far < 10, we can return it.
                if min_girth < g:
                    return min_girth
            return min_girth if min_girth < float('inf') else float('inf')

        def find_short_cycle(n, edges, g):
            # Returns a list of nodes in a cycle of length < g, or None
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                # Reconstruct cycle
                                path1 = []
                                curr = u
                                while curr != -1:
                                    path1.append(curr)
                                    curr = parent[curr]
                                path2 = []
                                curr = v
                                while curr != -1:
                                    path2.append(curr)
                                    curr = parent[curr]
                                # Find LCA
                                # This is getting complicated.
                                # Simpler: just return the edges involved?
                                # Or just return the node 'u' and 'v' and the distances?
                                # For the swap, we just need to know which edge to cut.
                                # We can cut the edge (u, parent[u]) or (v, parent[v])?
                                # Actually, the cycle is formed by the tree paths to u and v plus the edge (u,v).
                                # So we can cut the edge (u, v) itself? No, (u,v) is the non-tree edge.
                                # Wait, in BFS, if v is visited and v != parent[u], then (u,v) is a back edge.
                                # The cycle is u-v-...-u.
                                # We can remove the edge (u,v) and reconnect u to some other node?
                                # But (u,v) is an edge in the graph.
                                # If we remove (u,v), the graph is no longer regular.
                                # We must perform a swap: remove (u,v) and (x,y), add (u,x) and (v,y).
                                # To break the cycle, we should ensure the new edges don't reconnect the cycle.
                                # If we pick x, y far away, it's likely to break the cycle.
                                return (u, v)
            return None

        # Local Search
        N = 320
        start_time = time.time()
        limit = 60 # seconds
        
        # Initial graph
        edges = generate_random_k_regular(N, k)
        
        # Iterate
        while time.time() - start_time < limit:
            # Check girth
            # If girth >= 10, we are done
            # We need a fast check.
            # If we just check if there is any cycle < 10.
            
            # Find a short cycle
            short_edge = find_short_cycle(N, edges, g)
            
            if short_edge is None:
                # Verify exactly
                # (find_short_cycle returns None if no cycle < g found)
                # But BFS might miss some? No, BFS from every node finds all cycles.
                # If no cycle < g found from any start, girth >= g.
                return N, edges
            
            u, v = short_edge
            # Find a partner edge (x, y) to swap with
            # Randomly pick an edge not (u,v)
            # Ensure no loops/multi-edges
            
            edge_set = set((min(a,b), max(a,b)) for a,b in edges)
            swapped = False
            for _ in range(100): # Try 100 random partners
                x, y = random.choice(edges)
                if (x, y) == (u, v) or (y, x) == (u, v):
                    continue
                # Check if x, y are connected to u, v
                # New edges: (u, x) and (v, y) OR (u, y) and (v, x)
                # Try both
                
                # Option 1: (u, x), (v, y)
                if x != u and y != v and (u, x) not in edge_set and (v, y) not in edge_set:
                    # Also check if x, y are same as u, v?
                    # If x == v, then (u, v) is re-added? No, (u, x) = (u, v).
                    # We removed (u, v).
                    # So if x == v, we add (u, v) back. Bad.
                    if x != v and y != u:
                        # Perform swap
                        edges.remove((u, v))
                        edges.remove((x, y))
                        edges.append((u, x))
                        edges.append((v, y))
                        swapped = True
                        break
                
                # Option 2: (u, y), (v, x)
                if y != u and x != v and (u, y) not in edge_set and (v, x) not in edge_set:
                    if y != v and x != u:
                        edges.remove((u, v))
                        edges.remove((x, y))
                        edges.append((u, y))
                        edges.append((v, x))
                        swapped = True
                        break
            
            if not swapped:
                # Restart?
                edges = generate_random_k_regular(N, k)
                
    else:
        # Fallback to LUW for other parameters
        # (Copy LUW code from seed)
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

