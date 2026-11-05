import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def build_adj_list(adj, n):
        adj_list = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    adj_list[i].add(j)
                    adj_list[j].add(i)
        return adj_list

    def has_c5(adj_list, n):
        """Check for C5 using adjacency sets."""
        for a in range(n):
            na = adj_list[a]
            for b in na:
                if b <= a:
                    continue
                nb = adj_list[b]
                for c in nb:
                    if c <= b or c == a:
                        continue
                    nc = adj_list[c]
                    for d in nc:
                        if d <= c or d == a or d == b:
                            continue
                        if a in adj_list[d]:
                            return True
        return False

    def has_c5_with_new_vertex(adj_list, base_n, new_v, neighbors):
        """Check if adding new_v with given neighbors creates a C5."""
        # A C5 through new_v: new_v - a - b - c - d - new_v
        # where a, d in neighbors, b adjacent to a (b != new_v, b != d),
        # c adjacent to d (c != new_v, c != a), b adjacent to c, b != c
        for a in neighbors:
            for d in neighbors:
                if a >= d:
                    continue
                for b in adj_list[a]:
                    if b == d or b == new_v or b in neighbors:
                        continue
                    for c in adj_list[d]:
                        if c == a or c == new_v or c in neighbors:
                            continue
                        if b == c:
                            continue
                        if c in adj_list[b]:
                            return True
        return False

    def has_independent_set(adj_list, n, limit=9):
        """Check if there's an independent set of size >= limit."""
        def backtrack(remaining, current_size):
            if current_size >= limit:
                return True
            if len(remaining) + current_size < limit:
                return False
            if not remaining:
                return False
            # Pick vertex with minimum degree in remaining subgraph
            v = min(remaining, key=lambda x: len(adj_list[x] & set(remaining)))
            # Branch: include v
            new_remaining = [u for u in remaining if u != v and u not in adj_list[v]]
            if backtrack(new_remaining, current_size + 1):
                return True
            # Branch: exclude v
            remaining.remove(v)
            if backtrack(remaining, current_size):
                return True
            remaining.append(v)
            return False
        return backtrack(list(range(n)), 0)

    # Strategy: Start with 8 K4s (n=32), try to add vertex 32
    base_n = 32
    adj = np.zeros((base_n, base_n), dtype=bool)
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                adj[base + i, base + j] = True
                adj[base + j, base + i] = True
    
    adj_list = build_adj_list(adj, base_n)

    # Try to add vertex 32 connected to exactly one vertex per K4
    # This guarantees alpha <= 8 (can't pick 2 from any K4, and if 32 is picked,
    # none of its neighbors can be)
    # And we need to verify C5-freeness
    
    # First try deterministic: connect to vertex 0 of each K4
    connections = set()
    for b in range(8):
        connections.add(b * 4)
    
    if not has_c5_with_new_vertex(adj_list, base_n, 32, connections):
        # Build the full graph and verify
        new_adj = np.zeros((33, 33), dtype=bool)
        new_adj[:base_n, :base_n] = adj
        for v in connections:
            new_adj[32, v] = True
            new_adj[v, 32] = True
        
        new_adj_list = build_adj_list(new_adj, 33)
        
        if not has_c5(new_adj_list, 33) and not has_independent_set(new_adj_list, 33, 9):
            edges = []
            for i in range(33):
                for j in range(i + 1, 33):
                    if new_adj[i, j]:
                        edges.append([i, j])
            return {'n': 33, 'edges': edges}

    # Try random choices of one vertex per K4
    for attempt in range(2000):
        if time.time() > deadline:
            break
        
        connections = set()
        for b in range(8):
            connections.add(b * 4 + random.randint(0, 3))
        
        if has_c5_with_new_vertex(adj_list, base_n, 32, connections):
            continue
        
        new_adj = np.zeros((33, 33), dtype=bool)
        new_adj[:base_n, :base_n] = adj
        for v in connections:
            new_adj[32, v] = True
            new_adj[v, 32] = True
        
        new_adj_list = build_adj_list(new_adj, 33)
        
        if not has_c5(new_adj_list, 33) and not has_independent_set(new_adj_list, 33, 9):
            edges = []
            for i in range(33):
                for j in range(i + 1, 33):
                    if new_adj[i, j]:
                        edges.append([i, j])
            return {'n': 33, 'edges': edges}

    # Fallback: 8 K4s (n=32)
    edges = []
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    return {'n': 32, 'edges': edges}
