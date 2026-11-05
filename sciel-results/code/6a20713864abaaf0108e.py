import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    def build_adj_list(adj, n):
        adj_list = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    adj_list[i].add(j)
                    adj_list[j].add(i)
        return adj_list

    def has_c5_fast(adj_list, n):
        """Check for C5 using adjacency sets for speed."""
        for a in range(n):
            neighbors_a = adj_list[a]
            for b in neighbors_a:
                if b <= a:
                    continue
                for c in adj_list[b]:
                    if c <= b or c == a:
                        continue
                    for d in adj_list[c]:
                        if d <= c or d == a or d == b:
                            continue
                        if a in adj_list[d]:
                            return True
        return False

    def max_independent_set_size(adj_list, n, limit=9):
        """Find if there's an independent set of size >= limit."""
        # Use branch and bound
        def backtrack(remaining, current_size, best):
            if current_size >= limit:
                return True
            if len(remaining) + current_size <= best:
                return False
            if not remaining:
                return False
            # Pick vertex with minimum degree in remaining
            v = min(remaining, key=lambda x: len(adj_list[x] & set(remaining)))
            # Branch: include v
            new_remaining = [u for u in remaining if u != v and u not in adj_list[v]]
            if backtrack(new_remaining, current_size + 1, best):
                return True
            # Branch: exclude v
            remaining.remove(v)
            if backtrack(remaining, current_size, best):
                return True
            remaining.append(v)
            return False
        return backtrack(list(range(n)), 0, 0)

    def try_add_vertex(adj, n, new_vertex, adj_list):
        """Try to add a new vertex connected to maintain C5-free and alpha < 9."""
        pass

    # Strategy: Start with 8 K4s and try to add vertices
    # First, let's try a different structured construction
    
    # Try: complement of a C5-free graph approach
    # Or: use the fact that we can merge K4s with careful edges
    
    # Let's try a systematic local search starting from 8 K4s
    base_n = 32
    best_result = None
    
    # Generate base: 8 disjoint K4s
    adj = np.zeros((base_n, base_n), dtype=bool)
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                adj[base + i, base + j] = True
                adj[base + j, base + i] = True
    
    adj_list = build_adj_list(adj, base_n)
    
    # Try to add a 33rd vertex
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Randomly decide connections for vertex 32
        # Must connect to at least one vertex in each K4 to keep alpha < 9
        connections = set()
        valid = True
        for b in range(8):
            base = b * 4
            # Pick 1-4 vertices from this K4 to connect to
            num_connect = random.randint(1, 4)
            chosen = random.sample(range(4), num_connect)
            for i in chosen:
                connections.add(base + i)
        
        # Check C5-freeness with new vertex
        # Vertex 32's neighbors must not have a path of length 2 between any pair
        # (which would create a C5: 32 - u - w - v - 32)
        c5_found = False
        neighbors = connections
        for u in neighbors:
            for v in neighbors:
                if u >= v:
                    continue
                # Check if there's a common neighbor w of both u and v
                # that is not 32 and not u or v
                common = adj_list[u] & adj_list[v]
                common.discard(32)
                common.discard(u)
                common.discard(v)
                if common:
                    c5_found = True
                    break
            if c5_found:
                break
        
        if c5_found:
            continue
        
        # Also need to check C5s that don't go through vertex 32
        # but the base graph is C5-free, so only new C5s go through 32
        # A C5 through 32: 32-a-b-c-d-32 where a,d in neighbors, 
        # b adjacent to a, c adjacent to d, b adjacent to c, b!=d, c!=a
        # This means: a,d in neighbors, there exist b,c with:
        # b in adj(a), c in adj(d), b in adj(c), b != d, c != a, b != c
        # Let's check more carefully
        c5_found = False
        for a in neighbors:
            for d in neighbors:
                if a >= d:
                    continue
                # Need b adjacent to a, c adjacent to d, b adjacent to c
                # b != d, c != a, b != c, b != 32, c != 32
                for b in adj_list[a]:
                    if b == d or b == 32 or b in neighbors:
                        continue
                    for c in adj_list[d]:
                        if c == a or c == 32 or c in neighbors:
                            continue
                        if b == c:
                            continue
                        if c in adj_list[b]:
                            c5_found = True
                            break
                    if c5_found:
                        break
                if c5_found:
                    break
            if c5_found:
                break
        
        if c5_found:
            continue
        
        # Check alpha: with vertex 32 connected to at least one in each K4,
        # max independent set is at most 8 (one from each K4, can't include 32)
        # Actually, we need to verify alpha < 9
        # Build new adj_list
        new_adj = adj.copy()
        new_adj[32, :] = False
        new_adj[:, 32] = False
        for v in connections:
            new_adj[32, v] = True
            new_adj[v, 32] = True
        
        new_adj_list = build_adj_list(new_adj, 33)
        
        if max_independent_set_size(new_adj_list, 33, 9):
            continue
        
        # Valid! Return n=33
        edges = []
        for i in range(33):
            for j in range(i + 1, 33):
                if new_adj[i, j]:
                    edges.append([i, j])
        return {'n': 33, 'edges': edges}
    
    # Fallback: 8 K4s
    edges = []
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    return {'n': 32, 'edges': edges}
