import random
import time
import collections
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Moore bound for k=5, g=9 (odd)
    # 1 + k * sum((k-1)^i for i in range((g-1)//2))
    # g=9, r=4. 1 + 5 * (1 + 4 + 16 + 64) = 426.
    n_min = 426
    
    best_n = float('inf')
    best_edges = None
    
    # Fallback: The seed algorithm
    def seed_construct(k, g):
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

    # Helper: Check if graph has girth at least g
    def check_girth(adj, n, g):
        # For odd girth g=2r+1, we need no cycles of length < g.
        # We can check by BFS from each vertex up to depth r.
        # If we find a cycle of length < g, return False.
        r = (g - 1) // 2
        for u in range(n):
            # BFS
            dist = [-1] * n
            dist[u] = 0
            queue = collections.deque([u])
            while queue:
                v = queue.popleft()
                if dist[v] >= r:
                    continue
                for w in adj[v]:
                    if dist[w] == -1:
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    elif w != u and dist[v] + dist[w] + 1 < g:
                        # Found a cycle of length dist[v] + dist[w] + 1 < g
                        # Note: This checks for cycles passing through u?
                        # Actually, if dist[v] + dist[w] + 1 < g, and v, w are neighbors,
                        # then we have a cycle of length < g.
                        # However, we must ensure this is a simple cycle.
                        # BFS tree edges don't form cycles. Non-tree edges do.
                        # If (v,w) is an edge, and dist[v] + dist[w] + 1 < g,
                        # then the cycle formed by the paths to u and the edge (v,w)
                        # has length dist[v] + dist[w] + 1.
                        # This cycle contains u.
                        # What about cycles not containing u?
                        # If a cycle of length L < g exists, it contains some vertex x.
                        # Running BFS from x will find it.
                        # So checking from every u covers all cycles.
                        return False
        return True

    # Helper: Generate a simple k-regular graph on n vertices
    def generate_simple_regular(n, k):
        if n * k % 2 != 0:
            return None
        if k >= n:
            return None
            
        # Configuration model
        while True:
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, n * k, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            
            if valid:
                return adj
            
            # If n is small, retries are cheap. If n is large, this might be slow.
            # For n=430, k=5, probability of simple graph is decent.
            # If this loop takes too long, we should break.
            # But we have a global deadline.

    # Helper: Try to increase girth via edge switches
    def try_increase_girth(adj, n, k, g, max_steps=1000):
        # This is a heuristic.
        # Find shortest cycle, try to break it.
        # For simplicity, we just do random switches and check girth.
        # If girth increases, keep it.
        current_girth = 0
        # Estimate current girth
        # We don't need exact girth, just < g or >= g.
        
        for _ in range(max_steps):
            if time.time() > deadline:
                break
            # Pick two random edges
            # We need a list of edges
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            
            if len(edges) < 2:
                break
                
            # Pick two distinct edges
            e1, e2 = random.sample(edges, 2)
            u1, v1 = e1
            u2, v2 = e2
            
            # Check if they are disjoint
            if len({u1, v1, u2, v2}) < 4:
                continue
                
            # Try switching (u1,v1) and (u2,v2) to (u1,u2) and (v1,v2)
            # or (u1,v2) and (v1,u2)
            for (a, b, c, d) in [(u1, u2, v1, v2), (u1, v2, v1, u2)]:
                # Check if new edges are valid (no loops, no multi-edges)
                if a == b or c == d:
                    continue
                if b in adj[a] or d in adj[c]:
                    continue
                if a in adj[b] or c in adj[d]:
                    continue
                
                # Apply switch
                adj[a].remove(b)
                adj[b].remove(a)
                adj[c].remove(d)
                adj[d].remove(c)
                
                adj[a].append(c)
                adj[c].append(a)
                adj[b].append(d)
                adj[d].append(b)
                
                # Check girth
                if check_girth(adj, n, g):
                    return adj
                else:
                    # Revert
                    adj[a].remove(c)
                    adj[c].remove(a)
                    adj[b].remove(d)
                    adj[d].remove(b)
                    
                    adj[a].append(b)
                    adj[b].append(a)
                    adj[c].append(d)
                    adj[d].append(c)
            # If we didn't return, we reverted everything.
            # We can also accept switches that don't decrease girth, but checking girth is expensive.
            # For now, only accept if girth >= g.
            
        return None

    # Main search loop
    n = n_min
    # Step size for n
    # If n is too small, we might not find one.
    # We can try a few random graphs for each n.
    
    while n < 2000:
        if time.time() > deadline:
            break
            
        # Try a few random graphs for this n
        attempts = 5
        for _ in range(attempts):
            if time.time() > deadline:
                break
                
            adj = generate_simple_regular(n, k)
            if adj is None:
                continue
                
            if check_girth(adj, n, g):
                # Found a valid graph
                best_n = n
                # Convert adj to edge list
                edge_list = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edge_list.append((u, v))
                best_edges = edge_list
                # We can stop or try to find smaller?
                # We started from n_min and increased, so this is the smallest n we found.
                # Return immediately to save time?
                # Yes, since we want to minimize n.
                return n, edge_list
            else:
                # Try to repair
                repaired = try_increase_girth(adj, n, k, g, max_steps=50)
                if repaired is not None:
                    best_n = n
                    edge_list = []
                    for u in range(n):
                        for v in repaired[u]:
                            if u < v:
                                edge_list.append((u, v))
                    best_edges = edge_list
                    return n, edge_list
                    
        # Increase n
        n += 10
        
    # If we didn't find one, return the best found or fallback
    if best_edges is not None:
        return best_n, best_edges
    else:
        # Fallback to seed
        return seed_construct(k, g)

# EVOLVE-BLOCK-END
