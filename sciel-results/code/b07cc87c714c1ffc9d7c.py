import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Moore bound for k=3, g=16 is 382.
    # Prompt says lower bound 512.
    # Known best 936.
    # We try to find n < 936.
    
    best_n = float('inf')
    best_edges = []
    
    # We will try n values from 512 up to 936.
    # But we might not find one.
    # Let's start with a few random attempts for n=512, 528, etc.
    
    def is_valid_graph(n, edges):
        # Check simple
        if len(edges) != n * k // 2:
            return False
        adj = [set() for _ in range(n)]
        for u, v in edges:
            if u == v:
                return False
            if v in adj[u]:
                return False
            adj[u].add(v)
            adj[v].add(u)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        return True

    def compute_girth(n, adj):
        # BFS from each vertex to find shortest cycle
        min_girth = float('inf')
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] * 2 >= min_girth:
                    continue # No need to go deeper if we can't beat current min
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
            # Early exit if girth >= g
            if min_girth >= g:
                return min_girth
        return min_girth

    def generate_random_3_reg(n):
        # Configuration model
        while True:
            stubs = []
            for i in range(n):
                stubs.extend([i, i, i])
            random.shuffle(stubs)
            edges = []
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(0, 2 * n * k // 2, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
            if valid:
                return edges, adj

    def local_search(n, edges, adj, max_steps=10000):
        girth = compute_girth(n, adj)
        if girth >= g:
            return edges, adj, True
        
        for _ in range(max_steps):
            if time.time() > deadline:
                return edges, adj, False
            
            # Find a short cycle
            # We need to identify a cycle of length < g
            # For now, just pick random edges to swap
            
            # Pick two random edges
            e1_idx = random.randint(0, len(edges)-1)
            e2_idx = random.randint(0, len(edges)-1)
            if e1_idx == e2_idx:
                continue
                
            u1, v1 = edges[e1_idx]
            u2, v2 = edges[e2_idx]
            
            # Try swap: (u1, v1), (u2, v2) -> (u1, u2), (v1, v2)
            # Check validity
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                continue
            if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                continue
            
            # Perform swap
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            adj[u1].add(u2)
            adj[u2].add(u1)
            adj[v1].add(v2)
            adj[v2].add(v1)
            
            edges[e1_idx] = (u1, u2)
            edges[e2_idx] = (v1, v2)
            
            # Check if girth improved or maintained
            # We don't want to decrease girth
            new_girth = compute_girth(n, adj)
            if new_girth < girth:
                # Revert
                adj[u1].remove(u2)
                adj[u2].remove(u1)
                adj[v1].remove(v2)
                adj[v2].remove(v1)
                
                adj[u1].add(v1)
                adj[v1].add(u1)
                adj[u2].add(v2)
                adj[v2].add(u2)
                
                edges[e1_idx] = (u1, v1)
                edges[e2_idx] = (u2, v2)
            else:
                girth = new_girth
                if girth >= g:
                    return edges, adj, True
                    
        return edges, adj, False

    # Try a range of n
    # Start from 512
    for n in range(512, 937, 16): # Step by 16? Or 1?
        if time.time() > deadline:
            break
        
        # Try a few random instances
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            edges, adj = generate_random_3_reg(n)
            success = False
            edges, adj, success = local_search(n, edges, adj, max_steps=5000)
            
            if success:
                if n < best_n:
                    best_n = n
                    best_edges = edges
                return n, edges
                
    # If we didn't find one < 936, try to find one at 936 or return best effort
    # Actually, if we didn't find any, we should return a valid graph.
    # The seed program returns one.
    # We should ensure we return *something* valid.
    
    # Fallback: Use the LUW construction from the seed if we ran out of time?
    # Or just return the best_n found (which might be inf).
    # If best_n is inf, we must construct a valid graph.
    
    if best_n == float('inf'):
        # Try a larger n, e.g., 1000
        for n in range(936, 1001, 16):
            if time.time() > deadline:
                break
            edges, adj = generate_random_3_reg(n)
            edges, adj, success = local_search(n, edges, adj, max_steps=5000)
            if success:
                return n, edges
        
        # If still nothing, use a simple construction (e.g., LUW)
        # Re-implementing LUW briefly or just returning a large random graph?
        # A random graph of size 1000 might have girth < 16.
        # But we need a valid graph.
        
        # Let's just use the seed's logic if we have to, but it's slow?
        # Actually, the seed is deterministic.
        
        # For now, return the first valid graph we can make, even if large.
        # We can make a 3-regular graph of size 2000 easily.
        n = 2000
        edges, adj = generate_random_3_reg(n)
        # Check girth
        girth = compute_girth(n, adj)
        # If girth < 16, we might have to accept it?
        # No, the task requires girth >= 16.
        # We must find one.
        
        # Let's try a specific known construction if we can't find one.
        # But for now, we assume we can find one in the loop.
        
        # If we reach here, we failed to find a valid graph.
        # This should not happen if we try enough n.
        
        # Return a dummy valid graph?
        # No, must be valid.
        
        # Let's just return the LUW graph from the seed as a fallback.
        # (Copy of seed logic)
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

    return best_n, best_edges
