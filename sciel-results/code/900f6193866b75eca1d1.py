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

    # Strategy: Return the known valid graph with n=32 (8 disjoint K4s)
    # This guarantees a score of 32/28 = 1.1428...
    # Attempts to find n=33 are risky due to C5 constraints.
    
    n = 32
    adj = np.zeros((n, n), dtype=bool)
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                adj[base + i, base + j] = True
                adj[base + j, base + i] = True
    
    # Verify validity just in case
    adj_list = build_adj_list(adj, n)
    if has_c5(adj_list, n) or has_independent_set(adj_list, n, 9):
        # Fallback to a simpler valid graph if this fails (should not happen)
        # e.g. 7 K4s = 28 vertices
        n = 28
        adj = np.zeros((n, n), dtype=bool)
        for b in range(7):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    adj[base + i, base + j] = True
                    adj[base + j, base + i] = True

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append([i, j])
    
    return {'n': n, 'edges': edges}
