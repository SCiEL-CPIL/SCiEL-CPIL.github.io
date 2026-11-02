import time
import random
from collections import deque, defaultdict

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin

    if k != 3:
        # Fallback for other k, though task is k=3
        # Simple LUW or similar
        return seed_fallback(k, g)

    # Target girth is g (16).
    # We try to construct a 3-regular graph with girth >= g.
    
    def generate_random_3_regular(n):
        # Configuration model
        # Create 3n stubs
        stubs = []
        for i in range(n):
            for _ in range(3):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        adj = [set() for _ in range(n)]
        # Pair stubs
        for i in range(0, 3*n, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                # Self-loop, retry or fix
                # Simple fix: find another stub for u
                # This is messy. Better to regenerate or use a better generator.
                # For now, if self-loop, just swap with a random other stub
                # This can lead to multi-edges.
                # Let's just regenerate if self-loop or multi-edge occurs during pairing.
                # For n=936, probability of self-loop is low but possible.
                # Multi-edges are more likely.
                pass
            # We'll do a check after generation
            
        # Check for multi-edges and self-loops
        valid = True
        for i in range(n):
            if len(adj[i]) != 3:
                valid = False
                break
            if i in adj[i]:
                valid = False
                break
        if not valid:
            return None
            
        # Check multi-edges
        for i in range(n):
            for j in adj[i]:
                if i > j and adj[j].count(i) > 1: # This check is wrong, adj is set of neighbors
                    pass
        
        # Correct check:
        # Since we used sets, multi-edges are automatically merged.
        # But we need degree 3. If a multi-edge occurred, the degree would be < 3.
        # So the degree check above is sufficient.
        
        return adj

    def find_shortest_cycle(adj, n):
        # BFS from each vertex
        min_len = n # max possible cycle length
        # Optimization: BFS from 0, 1, ... until min_len is small
        # For girth 16, we can stop BFS at depth 7 (since cycle 16 implies path 15, 
        # but BFS from one end of a shortest cycle finds the other end at distance L/2? 
        # No, BFS from a vertex on the shortest cycle will find the "other side" of the cycle.
        # The shortest cycle length L. From a vertex on it, the shortest path to another vertex on it 
        # that is not the immediate neighbor is L-1? No.
        # If we do BFS from u, and we encounter a neighbor v of u that is not the parent, 
        # we have a cycle.
        # Standard BFS for shortest cycle:
        # For each start node s:
        #   BFS to find shortest distance to each node.
        #   For each edge (u, v), cycle length is dist[u] + dist[v] + 1.
        #   This finds the shortest cycle passing through s? No, it finds the shortest cycle in the graph 
        #   if we do it for all s? 
        #   Actually, the shortest cycle in the graph will be found when s is on that cycle.
        #   We can just do BFS from every node, but that's O(n^2).
        #   For n=936, 936*936 ~ 1M ops. Fast enough.
        
        # Better: 
        # We only care if there is a cycle < g.
        # So we can limit BFS depth to g//2.
        # If we find a cycle of length < g, we return it.
        
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                # If we go too deep, we don't care about cycles formed by paths longer than g//2
                # because a cycle of length L implies a path of length L-1 between two neighbors.
                # If L < g, then L-1 < g-1.
                # BFS depth limit: g//2 is sufficient to detect cycles of length < g?
                # A cycle of length 16: path of length 15.
                # If we BFS from a node on the cycle, we will reach the "opposite" node at distance 8.
                # The edge closing the cycle connects two nodes at distance 7 or 8?
                # Let's just limit depth to g//2.
                if dist[u] >= g//2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        # Length is dist[u] + dist[v] + 1
                        # But this might not be the shortest cycle through s.
                        # However, if dist[u] + dist[v] + 1 < g, we have a short cycle.
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return cycle_len
        return min_len # Return n if no short cycle found (or girth)

    def local_search(adj, n, target_girth, time_limit):
        # Returns adj if girth >= target_girth, else None
        current_adj = [set(lst) for lst in adj]
        # Initial check
        if find_shortest_cycle(current_adj, n) >= target_girth:
            return current_adj
            
        # Local search
        # We will try to increase the girth by breaking short cycles.
        # We track the current shortest cycle length.
        current_girth = find_shortest_cycle(current_adj, n)
        
        # To speed up, we don't recompute full shortest cycle every step.
        # We just pick a random edge on a known short cycle and try to break it.
        # But finding the cycle is part of the check.
        
        # Heuristic: 
        # 1. Find a short cycle C.
        # 2. Pick a random edge e in C.
        # 3. Pick a random edge f not in C, not incident to e.
        # 4. 2-switch e and f.
        # 5. If valid and new shortest cycle length >= old, accept.
        
        # This is expensive if we recompute shortest cycle every time.
        # Let's limit the number of steps.
        
        max_steps = 5000
        for step in range(max_steps):
            if time.time() > time_limit:
                return None
            
            # Find a short cycle
            # We need the cycle itself to pick an edge.
            # Modify find_shortest_cycle to return the cycle path.
            # For now, let's just pick a random edge in the graph and hope it's on a short cycle?
            # No, that's inefficient.
            
            # Let's stick to:
            # 1. Find shortest cycle length L.
            # 2. If L >= target, break.
            # 3. Find one such cycle.
            # 4. Try to break it.
            
            L = find_shortest_cycle(current_adj, n)
            if L >= target_girth:
                return current_adj
                
            # Find the cycle
            # Re-implement BFS to return the cycle
            cycle = find_a_short_cycle(current_adj, n, target_girth)
            if not cycle:
                # Should not happen if L < target
                return None
                
            # Pick a random edge from the cycle
            # Cycle is list of vertices [v0, v1, ..., v_{L-1}, v0]
            edge_idx = random.randint(0, L-1)
            u1 = cycle[edge_idx]
            v1 = cycle[(edge_idx + 1) % L]
            
            # Pick a random edge (u2, v2) from the graph
            # Not incident to u1 or v1
            # Not the same edge
            all_edges = []
            for i in range(n):
                for j in current_adj[i]:
                    if i < j:
                        all_edges.append((i, j))
                        
            # Filter valid edges for 2-switch
            valid_edges = []
            for u2, v2 in all_edges:
                if u2 == u1 or u2 == v1 or v2 == u1 or v2 == v1:
                    continue
                if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                    continue
                valid_edges.append((u2, v2))
                
            if not valid_edges:
                continue
                
            u2, v2 = random.choice(valid_edges)
            
            # Perform 2-switch
            # Remove (u1, v1) and (u2, v2)
            # Add (u1, u2) and (v1, v2)
            
            # Check for multi-edges
            if u2 in current_adj[u1] or v2 in current_adj[v1]:
                continue
            if u2 in current_adj[v1] or v2 in current_adj[u1]:
                # This check is for the new edges
                # New edges are (u1, u2) and (v1, v2)
                # If u2 is already connected to u1, it's a multi-edge.
                # We already checked that.
                # What if u2 is connected to v1? That's fine, it's a different edge.
                # But we must ensure (u1, u2) is not already an edge.
                # And (v1, v2) is not already an edge.
                pass
            
            # Actually, if u2 in current_adj[u1], we skip.
            # If v2 in current_adj[v1], we skip.
            # Also need to check if u1 in current_adj[u2] (same as above).
            
            # Apply
            current_adj[u1].remove(v1)
            current_adj[v1].remove(u1)
            current_adj[u2].remove(v2)
            current_adj[v2].remove(u2)
            
            current_adj[u1].add(u2)
            current_adj[u2].add(u1)
            current_adj[v1].add(v2)
            current_adj[v2].add(v1)
            
            # Check if this created a shorter cycle
            new_L = find_shortest_cycle(current_adj, n)
            if new_L < L:
                # Revert
                current_adj[u1].remove(u2)
                current_adj[u2].remove(u1)
                current_adj[v1].remove(v2)
                current_adj[v2].remove(v1)
                
                current_adj[u1].add(v1)
                current_adj[v1].add(u1)
                current_adj[u2].add(v2)
                current_adj[v2].add(u2)
                
            # If new_L >= L, we keep it.
            
        return None

    def find_a_short_cycle(adj, n, target_girth):
        # Returns a list of vertices forming a cycle of length < target_girth
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= target_girth // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        # Cycle found
                        # Reconstruct path
                        path_u = [u]
                        while path_u[-1] != s:
                            path_u.append(parent[path_u[-1]])
                            if parent[path_u[-1]] == -1: break
                        path_u.reverse() # s ... u
                        
                        path_v = [v]
                        while path_v[-1] != s:
                            path_v.append(parent[path_v[-1]])
                            if parent[path_v[-1]] == -1: break
                        path_v.reverse() # s ... v
                        
                        # Combine
                        # s ... u - v ... s
                        # The cycle is path_u + path_v (excluding s from one)
                        # Actually, the cycle is the union of the two paths.
                        # Let's just return the edge (u,v) and the paths?
                        # For the purpose of picking an edge, we can just return the cycle vertices.
                        # This reconstruction is O(L).
                        
                        # Simplified: just return the edge (u, v) and let the caller pick it?
                        # No, we need to ensure (u,v) is on a short cycle.
                        # It is.
                        # We can just return [u, v] as the "cycle" to be broken?
                        # No, we need to pick an edge *on* the cycle. (u,v) is on the cycle.
                        # So we can just return (u, v).
                        return [u, v]
        return None

    # Main logic
    # Try n=936 first
    # Then try smaller if time permits?
    # Or just 936.
    
    best_n = None
    best_edges = None
    
    # Try 936
    if time.time() < deadline - 10:
        adj = generate_random_3_regular(936)
        if adj:
            res_adj = local_search(adj, 936, g, deadline - 10)
            if res_adj:
                # Convert to edges
                edges = []
                for i in range(936):
                    for j in res_adj[i]:
                        if i < j:
                            edges.append((i, j))
                return 936, edges

    # Fallback: LUW
    return seed_fallback(k, g)

def seed_fallback(k, g):
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
