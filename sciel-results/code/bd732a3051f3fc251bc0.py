import random
import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Fallback: LUW construction
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
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def generate_random_3_regular(N):
        # Configuration model
        while True:
            stubs = []
            for i in range(N):
                stubs.extend([i, i, i])
            random.shuffle(stubs)
            edges = []
            seen = set()
            valid = True
            for i in range(0, 2 * N * 3 // 2, 2):
                u, v = stubs[i], stubs[i + 1]
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

    def get_girth_and_short_cycle(edges, N, min_girth):
        adj = [[] for _ in range(N)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check girth
        min_girth_found = float('inf')
        short_cycle = None
        
        for start in range(N):
            dist = [-1] * N
            parent = [-1] * N
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if parent[u] != v and parent[v] != u:
                            cycle_len = dist[u] + 1 + dist[v]
                            if cycle_len < min_girth_found:
                                min_girth_found = cycle_len
                                # Reconstruct cycle if needed
                                if cycle_len < min_girth:
                                     # Find the actual cycle path
                                     # This is a bit expensive, but we only do it if we find a short one
                                     pass
            if min_girth_found >= min_girth:
                return min_girth_found, None
        
        # If we get here, we found a short cycle somewhere, but didn't store it.
        # We need to return a cycle to break.
        # Let's modify the function to return the cycle edges.
        
        # Actually, let's just return the min girth. If < min_girth, we need a cycle.
        # We can find one by storing the first one found.
        
        # Re-implement to capture cycle
        for start in range(N):
            dist = [-1] * N
            parent = [-1] * N
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if parent[u] != v and parent[v] != u:
                            cycle_len = dist[u] + 1 + dist[v]
                            if cycle_len < min_girth:
                                # Reconstruct
                                cycle = []
                                # Path from u to start
                                curr = u
                                while curr != -1:
                                    cycle.append(curr)
                                    curr = parent[curr]
                                cycle.reverse()
                                # Path from v to start
                                curr = v
                                while curr != -1:
                                    cycle.append(curr)
                                    curr = parent[curr]
                                cycle.reverse()
                                # Merge
                                # The cycle is the part from start to u + start to v
                                # Actually, the cycle is u->...->start->...->v->u
                                # Let's just return the edge (u,v) and the path?
                                # Simpler: return the list of edges in the cycle.
                                # We can just return the edge (u,v) and the two paths.
                                # But for swapping, we just need to pick an edge in the cycle.
                                # Any edge in the cycle will do.
                                # Let's return the edge (u,v) and the cycle length.
                                # Actually, to break the cycle, we can pick any edge (a,b) in the cycle.
                                # Let's pick (u,v).
                                return cycle_len, (u, v)
        
        return min_girth_found, None

    def local_search(k, g, N, time_limit):
        # Only works for k=3 currently
        if k != 3:
            return None
        
        edges = generate_random_3_regular(N)
        if not edges:
            return None
            
        current_edges = edges
        best_n = N
        best_edges = current_edges
        
        while time.time() < time_limit:
            girth, cycle_edge = get_girth_and_short_cycle(current_edges, N, g)
            if girth >= g:
                return N, current_edges
            
            if cycle_edge is None:
                break
                
            u, v = cycle_edge
            # Find an edge (x, y) not in the cycle (or just any other edge) to swap with
            # To be safe, pick a random edge from the list
            # Ensure it's not (u,v) and doesn't create loops
            valid_swaps = []
            for i, (x, y) in enumerate(current_edges):
                if (x, y) == (u, v) or (x, y) == (v, u):
                    continue
                # Check if swapping (u,v) and (x,y) with (u,x) and (v,y) is valid
                # Must not create loops: u!=x, v!=y, u!=y, v!=x
                if u == x or v == y or u == y or v == x:
                    continue
                # Must not create multi-edges: (u,x) not in edges, (v,y) not in edges
                # We need a set of edges for fast lookup
                # This is slow. Let's just try and validate.
                valid_swaps.append((i, x, y))
                
            if not valid_swaps:
                # Try to regenerate or break
                break
                
            idx, x, y = random.choice(valid_swaps)
            
            # Perform swap
            # Remove (u,v) and (x,y)
            # Add (u,x) and (v,y)
            new_edges = []
            for i, (a, b) in enumerate(current_edges):
                if (a, b) == (u, v) or (a, b) == (v, u):
                    continue
                if (a, b) == (x, y) or (a, b) == (y, x):
                    continue
                new_edges.append((a, b))
            
            # Check if (u,x) or (v,y) already exist
            edge_set = set((min(a,b), max(a,b)) for a,b in new_edges)
            if (min(u,x), max(u,x)) in edge_set or (min(v,y), max(v,y)) in edge_set:
                continue
            if (min(u,y), max(u,y)) in edge_set or (min(v,x), max(v,x)) in edge_set:
                 # Try other pairing
                 pass
                 
            new_edges.append((u, x))
            new_edges.append((v, y))
            current_edges = new_edges
            best_edges = current_edges
            
        return best_n, best_edges

    # Main Logic
    if k == 3 and g == 16:
        # Try local search for the known cage size
        # Known best is 936
        target_n = 936
        result = local_search(k, g, target_n, deadline - 5)
        if result:
            return result
        
        # If local search failed, try slightly larger
        for n_try in [950, 1000, 1100]:
            if time.time() > deadline - 5:
                break
            result = local_search(k, g, n_try, deadline - 5)
            if result:
                return result
                
    # Fallback
    return luw_construct(k, g)
