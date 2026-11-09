import time
import sys
import itertools
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Target N
    # Moore bound for k=4, g=10 is 243. Known min is 320.
    # We try to find a graph with N vertices.
    
    def check_girth(adj, N, g):
        """Check if graph has girth >= g. Returns True if valid."""
        # For each vertex, BFS to check shortest cycle
        # A cycle of length L exists if there is an edge (u,v) such that dist(u,v) < L in G - (u,v)
        # Or simpler: BFS from each vertex, if we find a back edge to an ancestor at depth < g/2, cycle is short.
        # Actually, just BFS from each vertex up to depth g/2 - 1. If we find a neighbor of the start vertex 
        # at depth >= 1 (excluding the parent in BFS tree), we have a cycle.
        # Wait, standard girth check:
        # For each edge (u,v), check if there is a path of length < g-1 between u and v not using the edge.
        # With N=320, this is fast.
        
        # Optimization: Only need to check if min cycle >= g.
        # We can do BFS from each vertex.
        for u in range(N):
            dist = [-1] * N
            parent = [-1] * N
            dist[u] = 0
            queue = [u]
            head = 0
            while head < len(queue):
                curr = queue[head]
                head += 1
                if dist[curr] >= g // 2: 
                    continue
                for v in adj[curr]:
                    if dist[v] == -1:
                        dist[v] = dist[curr] + 1
                        parent[v] = curr
                        queue.append(v)
                    else:
                        # Found a cycle
                        # Cycle length = dist[curr] + dist[v] + 1
                        # But this detects *any* cycle. We need to ensure it's not the tree edge.
                        # If v is not parent[curr] and curr is not parent[v] (in undirected, just check if it's a back edge)
                        # In BFS, if dist[v] == dist[curr] or dist[v] == dist[curr] - 1, it's a cross/back edge.
                        # The cycle length is dist[curr] + dist[v] + 1.
                        # We need to be careful with the starting vertex.
                        pass
            # This BFS approach is tricky for exact girth.
            # Better: For each edge (u,v), remove it, find shortest path u->v. If < g-1, invalid.
            # This is O(E * (N+E)). E ~ 4N. N=320. 1280 * 320 ~ 400k ops. Very fast.
        
        for u in range(N):
            for v in adj[u]:
                if u > v: continue
                # Temporarily remove (u,v)
                # Find shortest path u to v
                dist = [-1] * N
                dist[u] = 0
                queue = [u]
                head = 0
                found = False
                while head < len(queue):
                    curr = queue[head]
                    head += 1
                    if curr == v:
                        found = True
                        break
                    if dist[curr] >= g - 1:
                        continue
                    for w in adj[curr]:
                        if (curr == u and w == v) or (curr == v and w == u):
                            continue
                        if dist[w] == -1:
                            dist[w] = dist[curr] + 1
                            queue.append(w)
                if found and dist[v] < g - 1:
                    return False
                if not found:
                    # Graph is disconnected? Or path doesn't exist?
                    # If path doesn't exist, no cycle.
                    pass
        return True

    def backtrack(N):
        adj = [set() for _ in range(N)]
        deg = [0] * N
        
        # Symmetry breaking:
        # v0 connected to 1,2,3,4
        for v in [1,2,3,4]:
            adj[0].add(v)
            adj[v].add(0)
            deg[0] += 1
            deg[v] += 1
            
        # v1 connected to 0, 5,6,7
        for v in [5,6,7]:
            adj[1].add(v)
            adj[v].add(1)
            deg[1] += 1
            deg[v] += 1
            
        # v2 connected to 0, 8,9,10
        for v in [8,9,10]:
            adj[2].add(v)
            adj[v].add(2)
            deg[2] += 1
            deg[v] += 1
            
        # v3 connected to 0, 11,12,13
        for v in [11,12,13]:
            adj[3].add(v)
            adj[v].add(3)
            deg[3] += 1
            deg[v] += 1
            
        # v4 connected to 0, 14,15,16
        for v in [14,15,16]:
            adj[4].add(v)
            adj[v].add(4)
            deg[4] += 1
            deg[v] += 1
            
        # Check if initial setup is valid (no short cycles)
        if not check_girth(adj, 17, g):
            return None
            
        # Recursive step
        # We need to complete degrees for all vertices.
        # We process vertices in order.
        
        def is_safe(u, v):
            # Check if adding edge (u,v) creates a cycle < g
            # Path from u to v in current graph (without u-v) must be >= g-1
            # Since we are adding edges, the current graph is a subgraph.
            # We can do a BFS from u to v in the current adj.
            dist = [-1] * N
            dist[u] = 0
            queue = [u]
            head = 0
            while head < len(queue):
                curr = queue[head]
                head += 1
                if curr == v:
                    return dist[v] >= g - 1
                if dist[curr] >= g - 1:
                    continue
                for w in adj[curr]:
                    if dist[w] == -1:
                        dist[w] = dist[curr] + 1
                        queue.append(w)
            # If v is not reached, it's safe (no cycle)
            return True

        def solve(idx):
            if time.time() > deadline:
                return False
            
            # Find the next vertex to complete
            u = -1
            for i in range(idx, N):
                if deg[i] < k:
                    u = i
                    break
            
            if u == -1:
                # All degrees satisfied
                return check_girth(adj, N, g)
            
            need = k - deg[u]
            # Candidates: vertices v > u with deg[v] < k
            # Also, we might need to connect to v < u if they are not full?
            # In this "forward" construction, we usually only connect to v > u 
            # to avoid symmetry and ensure we fill the graph.
            # But if we only connect to v > u, we might run out of candidates 
            # for early vertices if N is small.
            # For N=320, and k=4, this should be fine.
            
            candidates = [v for v in range(u + 1, N) if deg[v] < k]
            
            # If not enough candidates, we might need to backtrack or 
            # connect to previous vertices? 
            # If we strictly enforce v > u, we might fail.
            # Let's allow v < u if deg[v] < k, but we must avoid duplicates.
            # To avoid duplicates, we only consider v > u for the "first" time?
            # Actually, if we process vertices in order, and only add edges (u,v) with v > u,
            # then when we get to v, u is already processed? No, u is current.
            # If we only add edges to v > u, then when we process v later, 
            # it will only add edges to w > v.
            # This ensures each edge is considered exactly once (from the smaller index).
            # However, this might make it impossible to satisfy degrees if 
            # the graph structure requires "backwards" edges.
            # But any graph can be ordered such that edges go forward? No.
            # But we are *constructing* the graph. We can choose the edges.
            # So we can always choose v > u.
            
            if len(candidates) < need:
                return False
            
            # Heuristic: Sort candidates by "safety" or degree
            # Prefer connecting to vertices with higher degree (to fill them up)
            # or lower degree (to spread out)?
            # Let's try lower degree first (to keep options open for others)?
            # Actually, high degree vertices are "tight".
            # Let's just iterate.
            
            # To reduce branching, we can fix the first few connections of u?
            # No, just use combinations.
            
            # Optimization: If need is 1, just loop.
            # If need > 1, use combinations.
            
            # Sort candidates to improve pruning?
            # Maybe sort by dist from u?
            
            for combo in itertools.combinations(candidates, need):
                # Check safety for all edges in combo
                safe = True
                for v in combo:
                    if not is_safe(u, v):
                        safe = False
                        break
                
                if not safe:
                    continue
                
                # Add edges
                for v in combo:
                    adj[u].add(v)
                    adj[v].add(u)
                    deg[u] += 1
                    deg[v] += 1
                
                # Recurse
                if solve(u + 1):
                    return True
                
                # Undo
                for v in combo:
                    adj[u].remove(v)
                    adj[v].remove(u)
                    deg[u] -= 1
                    deg[v] -= 1
            
            return False
        
        return solve(0)

    # Try N values
    # Start with known optimum 320
    for N in range(320, 330):
        if time.time() > deadline:
            break
        result = backtrack(N)
        if result:
            # Convert adj to list of edges
            edges = []
            for u in range(N):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return N, edges
    
    # If backtracking fails or times out, try a simpler greedy/randomized approach
    # that might find a larger but valid graph.
    # For now, if we fail, we return the seed's result? 
    # No, we must return *a* valid graph.
    # Let's try a random regular graph and check girth, retrying if invalid.
    # This is unlikely to work for girth 10.
    
    # Fallback: Return a valid graph from a simpler construction if available.
    # Since we don't have one hardcoded, we might return None or raise.
    # But the task requires a valid result.
    # We should ensure the backtracker is robust.
    
    # If we reach here, we failed. 
    # We can try a less constrained backtracker for larger N.
    for N in range(330, 350):
        if time.time() > deadline:
            break
        result = backtrack(N)
        if result:
            edges = []
            for u in range(N):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return N, edges
            
    # Absolute fallback: Return the seed's logic if it's fast?
    # The seed is L-U-W.
    # Let's just return the seed's output if we have to.
    # But we can't easily call the seed code here without duplicating it.
    # Let's duplicate the seed's logic as a last resort.
    
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

