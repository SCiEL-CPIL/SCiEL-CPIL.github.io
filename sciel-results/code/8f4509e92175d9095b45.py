<<<<<<< SEARCH
import random
import time
import numpy as np

def construct(k, g):
    # Fallback to LUW if we don't find a better one
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

    n_luw, edges_luw = luw_construct(k, g)
    
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Moore bound for k=3, g=16 is approx 762? 
    # Let's start from a reasonable lower bound.
    # The prompt says best lower bound is 512.
    # We search for n in [512, n_luw]
    
    best_n = n_luw
    best_edges = edges_luw
    
    # We assume k=3, g=16 for this task, but code should be generic-ish or at least handle k=3.
    if k != 3:
        return n_luw, edges_luw
        
    # Search for smaller n
    # n must be even for 3-regular
    for n in range(512, n_luw + 1, 2):
        if time.time() > deadline:
            break
            
        # Try to find a valid graph for this n
        # We try a few random instances
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Generate random 3-regular bipartite graph
            # Left: 0..n/2-1, Right: n/2..n-1
            half_n = n // 2
            left_stubs = []
            right_stubs = []
            for i in range(half_n):
                left_stubs.extend([i, i, i])
                right_stubs.extend([i, i, i])
                
            # Random permutation of right stubs
            perm = np.random.permutation(right_stubs)
            
            # Create edges
            edges = []
            valid = True
            # To check for multi-edges, we can use a set of (u, v) pairs
            # But for speed, we can just check degrees and loops (loops impossible in bipartite)
            # Multi-edge: if (u, v) appears twice.
            # We can use a set of tuples.
            edge_set = set()
            
            for i in range(3 * half_n):
                u = left_stubs[i]
                v = perm[i] + half_n
                if (u, v) in edge_set:
                    valid = False
                    break
                edge_set.add((u, v))
                edges.append((u, v))
                
            if not valid:
                continue
                
            # Check girth
            # BFS from each vertex to depth g/2 - 1
            depth_limit = g // 2 - 1
            
            has_short_cycle = False
            for v in range(n):
                # BFS
                visited = [-1] * n
                visited[v] = 0
                queue = [(v, 0)]
                
                while queue:
                    node, dist = queue.pop(0)
                    if dist >= depth_limit:
                        continue
                    
                    for neighbor in edges:
                        if neighbor[0] == node:
                            nb = neighbor[1]
                        elif neighbor[1] == node:
                            nb = neighbor[0]
                        else:
                            continue
                            
                        if visited[nb] == -1:
                            visited[nb] = dist + 1
                            queue.append((nb, dist + 1))
                        else:
                            # Found a cycle
                            # Cycle length is dist + 1 + visited[nb]?
                            # No, if nb is visited at depth d_nb, and we are at dist d_u.
                            # The cycle length is d_u + d_nb + 1?
                            # This is only true if the paths are disjoint.
                            # In BFS, the first time we see a visited node, it might not form the shortest cycle.
                            # However, for girth check, we just need to know if ANY cycle is < g.
                            # A back edge to a node at depth d_nb from a node at depth d_u
                            # creates a cycle of length d_u + d_nb + 1.
                            # If this length < g, we have a short cycle.
                            cycle_len = dist + 1 + visited[nb]
                            if cycle_len < g:
                                has_short_cycle = True
                                break
                                
                    if has_short_cycle:
                        break
                if has_short_cycle:
                    break
                    
            if not has_short_cycle:
                # Found a valid graph
                return n, edges
                
    return best_n, best_edges
=======
import random
import time
import numpy as np

def construct(k, g):
    # Fallback to LUW if we don't find a better one
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

    n_luw, edges_luw = luw_construct(k, g)
    
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_n = n_luw
    best_edges = edges_luw
    
    if k != 3:
        return n_luw, edges_luw
        
    def check_girth_fast(n, edges, g):
        """Check girth >= g using BFS with adjacency lists."""
        if time.time() > deadline:
            return False
            
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        depth_limit = g // 2
        
        for start in range(n):
            if time.time() > deadline:
                return False
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            qi = 0
            
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                if dist[u] >= depth_limit:
                    continue
                    
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if parent[u] == v or parent[v] == u:
                            continue
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
                            
        return True

    def generate_random_3reg(n):
        """Generate a random 3-regular graph on n vertices."""
        if n % 2 != 0 or n < 4:
            return None
        stubs = []
        for i in range(n):
            stubs.extend([i, i, i])
        perm = np.random.permutation(len(stubs))
        edges = set()
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[perm[i]]
            if u == v:
                return None
            if u > v:
                u, v = v, u
            if (u, v) in edges:
                return None
            edges.add((u, v))
        return list(edges)

    def generate_random_3reg_bipartite(n):
        """Generate a random 3-regular bipartite graph."""
        half_n = n // 2
        if half_n < 3:
            return None
        left_stubs = []
        right_stubs = []
        for i in range(half_n):
            left_stubs.extend([i, i, i])
            right_stubs.extend([i, i, i])
        perm = np.random.permutation(len(right_stubs))
        edge_set = set()
        edges = []
        for i in range(3 * half_n):
            u = left_stubs[i]
            v = perm[i]
            if (u, v) in edge_set:
                return None
            edge_set.add((u, v))
            edges.append((u, v + half_n))
        return edges

    # Moore bound for k=3, g=16: n >= 2*(2^8-1) = 510, so n >= 512
    search_start = 512
    search_end = min(n_luw, 1000)
    
    for n in range(search_start, search_end + 1, 2):
        if time.time() > deadline:
            break
        time_remaining = deadline - time.time()
        num_attempts = min(15, max(2, int(time_remaining / 8)))
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            if attempt % 2 == 0:
                edges = generate_random_3reg(n)
            else:
                edges = generate_random_3reg_bipartite(n)
            if edges is None:
                continue
            if check_girth_fast(n, edges, g):
                return n, edges
                
    return best_n, best_edges
>>>>>>> REPLACE
