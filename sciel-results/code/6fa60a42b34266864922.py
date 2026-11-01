import random
import time
import sys
import heapq
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin for safety
    
    # Fallback: LUW graph
    def luw_graph(k, g):
        q = k
        # Determine d such that girth is at least g
        # LUW graph girth is roughly 2d+1 or similar depending on parameters
        # For k=5, g=9, the standard LUW construction uses d=4, yielding n=1250.
        # Let's just hardcode the logic that produced 1250 for k=5, g=9.
        # The general formula for n in LUW is 2 * q * q^(d-1).
        # For k=5, q=5.
        # If d=4, n = 2 * 5 * 125 = 1250.
        
        # Re-implementing the specific LUW logic from the seed to ensure correctness.
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

    n_luw, edges_luw = luw_graph(k, g)
    best_n, best_edges = n_luw, edges_luw

    # Attempt to find a smaller graph using random switching
    # Target n slightly smaller than LUW, down to Moore bound
    moore_bound = 1 + k * sum((k-1)**i for i in range((g-3)//2 + 1))
    
    # We will try to construct a graph of size 1116 directly using switching.
    # This is the known best upper bound.
    
    def gen_random_regular(n, k):
        # Use the "configuration model" with retries
        # For n=1116, k=5, this should be fast.
        for _ in range(50):
            stubs = [i for i in range(n) for _ in range(k)]
            random.shuffle(stubs)
            adj = {i: set() for i in range(n)}
            valid = True
            for i in range(0, n * k, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            if valid:
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return adj, edges
        return None, None

    def find_short_cycle(n, adj, g):
        # Returns a list of vertices forming a cycle of length < g, or None
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
                        # Found a back edge (u, v)
                        # Reconstruct cycle
                        # Path from start to u and start to v
                        # We need the cycle length
                        # dist[u] + dist[v] + 1 is the length of the cycle formed by the tree paths and edge (u,v)
                        # IF this is the shortest cycle from start, we can return it.
                        # But we need the actual vertices to switch.
                        # Let's trace back to LCA.
                        
                        # Simple check: if dist[u] + dist[v] + 1 < g, we have a short cycle.
                        # To get the vertices:
                        path_u = []
                        cur = u
                        while cur != -1:
                            path_u.append(cur)
                            if cur == start: break
                            cur = parent[cur]
                        # path_u is u ... start
                        # We need to find the LCA of u and v in the BFS tree rooted at start.
                        # Since it's a BFS tree, the LCA is the first common ancestor.
                        set_u = set(path_u)
                        lca = None
                        cur = v
                        while cur != -1:
                            if cur in set_u:
                                lca = cur
                                break
                            cur = parent[cur]
                        
                        if lca is not None:
                            # Cycle is path from u to lca + path from v to lca
                            # path u -> lca
                            cycle_part_u = []
                            cur = u
                            while cur != lca:
                                cycle_part_u.append(cur)
                                cur = parent[cur]
                            cycle_part_u.append(lca)
                            
                            # path v -> lca
                            cycle_part_v = []
                            cur = v
                            while cur != lca:
                                cycle_part_v.append(cur)
                                cur = parent[cur]
                            cycle_part_v.append(lca)
                            
                            cycle = cycle_part_u + cycle_part_v[::-1]
                            # Remove duplicate lca at the end of cycle_part_v[::-1] ?
                            # cycle_part_u ends with lca.
                            # cycle_part_v[::-1] starts with lca.
                            # So we have ... lca, lca ...
                            # We should remove one lca.
                            if len(cycle) > 1 and cycle[-1] == cycle[0]:
                                cycle.pop() # This might be wrong if cycle is small
                            # Actually, let's just check length
                            # The cycle is u...lca...v...u? No, u...lca and v...lca.
                            # The cycle is u -> ... -> lca -> ... -> v -> (edge v-u? No, edge is u-v in graph, but in tree they are separate)
                            # Wait, the edge (u,v) is the one that closed the cycle.
                            # So the cycle is u -> parent[u] ... lca ... parent[v] -> v -> u.
                            # My construction:
                            # cycle_part_u: u, ..., lca
                            # cycle_part_v: v, ..., lca
                            # Reversed cycle_part_v: lca, ..., v
                            # Concatenation: u, ..., lca, lca, ..., v
                            # This has lca twice.
                            # Correct cycle: u, ..., lca, ..., v. And then edge (v,u) closes it?
                            # No, the edge is (u,v).
                            # So the cycle is u -> ... -> lca -> ... -> v -> u.
                            # The vertices are u, ..., lca, ..., v.
                            # So we should remove the duplicate lca.
                            if len(cycle) >= 2 and cycle[1] == cycle[-1]: # Not quite
                                pass
                            
                            # Let's just verify length
                            # len(cycle_part_u) + len(cycle_part_v) - 1
                            if len(cycle_part_u) + len(cycle_part_v) - 1 < g:
                                # Return the unique vertices
                                return cycle_part_u + cycle_part_v[-2::-1] # v ... lca (excluding lca which is already in part_u)
                        # If no LCA found, continue
        return None

    def try_switch(adj, n, k, g, max_iterations=5000):
        # Try to improve girth
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            cycle = find_short_cycle(n, adj, g)
            if not cycle:
                # Girth is >= g
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return adj, edges
            
            # Pick an edge in the cycle to switch
            # Cycle is a list of vertices [v0, v1, ..., vm]
            # Edges are (v0,v1), (v1,v2), ..., (vm,v0)
            # Pick a random index
            idx = random.randint(0, len(cycle)-1)
            u = cycle[idx]
            v = cycle[(idx+1) % len(cycle)]
            
            # Pick a random edge (x,y) from the graph
            # To be efficient, we can pick a random vertex x and a random neighbor y
            # Ensure (x,y) is not (u,v) or (v,u) and doesn't create self-loop/multi-edge
            # We will try a few times to find a valid switch
            switched = False
            for _ in range(20):
                x = random.randint(0, n-1)
                if not adj[x]: continue
                y = random.choice(list(adj[x]))
                
                # Check if edges are disjoint
                if x in (u, v) or y in (u, v):
                    continue
                
                # Try switch (u,v) and (x,y) -> (u,x) and (v,y)
                # Check validity
                if x == u or y == v: # Self loop
                    continue
                if x in adj[v] or y in adj[u]: # Multi-edge
                    continue
                # Also check (u,y) and (v,x) option?
                # Let's try (u,x) and (v,y) first
                
                # Perform switch
                adj[u].remove(v)
                adj[v].remove(u)
                adj[x].remove(y)
                adj[y].remove(x)
                
                adj[u].add(x)
                adj[x].add(u)
                adj[v].add(y)
                adj[y].add(v)
                
                switched = True
                break
                
            if not switched:
                # Try the other pairing (u,y) and (v,x)
                for _ in range(20):
                    x = random.randint(0, n-1)
                    if not adj[x]: continue
                    y = random.choice(list(adj[x]))
                    
                    if x in (u, v) or y in (u, v):
                        continue
                    
                    if y == u or x == v: # Self loop
                        continue
                    if y in adj[v] or x in adj[u]: # Multi-edge
                        continue
                    
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[x].remove(y)
                    adj[y].remove(x)
                    
                    adj[u].add(y)
                    adj[y].add(u)
                    adj[v].add(x)
                    adj[x].add(v)
                    
                    switched = True
                    break
            
            if not switched:
                # Stuck? Try a different cycle edge or break
                pass

        # If we didn't find a valid graph, return None
        # We check one last time
        if find_short_cycle(n, adj, g) is None:
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return adj, edges
        return None, None

    # Try to find a graph with n=1116
    target_n = 1116
    if target_n >= moore_bound:
        adj, edges = gen_random_regular(target_n, k)
        if adj:
            # Check if it's already valid (unlikely)
            if find_short_cycle(target_n, adj, g) is None:
                return target_n, edges
            
            # Try switching
            result_adj, result_edges = try_switch(adj, target_n, k, g)
            if result_adj:
                return target_n, result_edges
    
    # Fallback to LUW
    return best_n, best_edges
