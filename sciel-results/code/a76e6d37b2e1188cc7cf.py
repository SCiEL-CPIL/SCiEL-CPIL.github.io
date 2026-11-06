"""Find max triangle-free 16-vertex graph with chromatic number >= 4.

Strategy:
1. Start with Grötzsch graph (11 vertices, 20 edges, chi=4, triangle-free).
2. Extend to 16 vertices using smart greedy + systematic edge addition.
3. Use efficient 3-colorability checking.
4. Maximize edges while maintaining triangle-free and non-3-colorable.
"""

import random
import itertools
from collections import defaultdict

def is_triangle_free_edge(adj, n, u, v):
    """Check if adding edge (u,v) keeps graph triangle-free."""
    if v in adj[u]:
        return False
    # Check no common neighbor
    for w in adj[u]:
        if w in adj[v]:
            return False
    return True

def is_3_colorable(adj, n):
    """Backtracking 3-colorability check with degree ordering."""
    color = [-1] * n
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        used = set()
        for w in adj[v]:
            if color[w] != -1:
                used.add(color[w])
        for c in range(3):
            if c not in used:
                color[v] = c
                if backtrack(idx + 1):
                    return True
                color[v] = -1
        return False
    
    return backtrack(0)

def make_grotzsch():
    """Create the Grötzsch graph on vertices 0..10."""
    adj = [set() for _ in range(11)]
    # Cycle 0-1-2-3-4-0
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    for u, v in cycle:
        adj[u].add(v)
        adj[v].add(u)
    # Mycielski: vertex 5+i connected to i and i+1 (mod 5)... 
    # Standard Mycielski: for each edge (u,v) in C5, add vertex 5+u connected to v,
    # and 5+v connected to u. Then add vertex 10 connected to all 5..9.
    for u, v in cycle:
        adj[u].add(5 + v)
        adj[5 + v].add(u)
        adj[v].add(5 + u)
        adj[5 + u].add(v)
    for i in range(5):
        adj[10].add(5 + i)
        adj[5 + i].add(10)
    return adj

def count_edges(adj, n):
    return sum(len(adj[i]) for i in range(n)) // 2

def add_edge(adj, u, v):
    adj[u].add(v)
    adj[v].add(u)

def remove_edge(adj, u, v):
    adj[u].discard(v)
    adj[v].discard(u)

def try_add_edge(adj, n, u, v):
    """Try to add edge (u,v). Returns True if added (graph remains non-3-colorable)."""
    if u == v or v in adj[u]:
        return False
    if not is_triangle_free_edge(adj, n, u, v):
        return False
    add_edge(adj, u, v)
    if is_3_colorable(adj, n):
        remove_edge(adj, u, v)
        return False
    return True

def greedy_extend(adj, n, start_v, end_v):
    """Greedy extension: for each new vertex, find max independent set among existing."""
    for new_v in range(start_v, end_v):
        existing = list(range(new_v))
        best_conn = set()
        best_edges = -1
        for r in range(min(len(existing), 12), -1, -1):
            found = False
            for subset in itertools.combinations(existing, r):
                valid = True
                for i in range(len(subset)):
                    for j in range(i + 1, len(subset)):
                        if subset[j] in adj[subset[i]]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                if r > best_edges:
                    best_edges = r
                    best_conn = set(subset)
                    found = True
                    break
            if found:
                break
        for u in best_conn:
            add_edge(adj, new_v, u)
    return adj

def systematic_edge_addition(adj, n):
    """Try to add all possible edges systematically."""
    improved = True
    while improved:
        improved = False
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if v not in adj[u] and is_triangle_free_edge(adj, n, u, v):
                    candidates.append((u, v))
        candidates.sort(key=lambda e: -(len(adj[e[0]]) + len(adj[e[1]])))
        for u, v in candidates:
            if v in adj[u]:
                continue
            if try_add_edge(adj, n, u, v):
                improved = True
    return adj

def local_search(adj, n, iterations, seed=42):
    """Local search to improve edge count."""
    random.seed(seed)
    for _ in range(iterations):
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v or v in adj[u]:
            continue
        if not is_triangle_free_edge(adj, n, u, v):
            continue
        try_add_edge(adj, n, u, v)
    return adj, count_edges(adj, n)

def construct():
    """Find a triangle-free graph on 16 vertices with chromatic number >= 4, maximizing edges."""
    n = 16
    adj = make_grotzsch()
    while len(adj) < n:
        adj.append(set())
    
    # Greedy extension
    adj = greedy_extend(adj, n, 11, 16)
    
    # If it became 3-colorable, try a more careful extension
    if is_3_colorable(adj, n):
        adj2 = make_grotzsch()
        while len(adj2) < n:
            adj2.append(set())
        for new_v in range(11, 16):
            existing = list(range(new_v))
            for r in range(6, 0, -1):
                done = False
                for subset in itertools.combinations(existing, r):
                    valid = True
                    for i in range(len(subset)):
                        for j in range(i + 1, len(subset)):
                            if subset[j] in adj2[subset[i]]:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                    for u in subset:
                        add_edge(adj2, new_v, u)
                    if not is_3_colorable(adj2, n):
                        done = True
                        break
                    else:
                        for u in subset:
                            remove_edge(adj2, new_v, u)
                if done:
                    break
        adj = adj2
    
    # Systematic edge addition
    adj = systematic_edge_addition(adj, n)
    best_adj = [set(s) for s in adj]
    best_count = count_edges(adj, n)
    
    # Multiple rounds of local search with different seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        adj_copy = [set(s) for s in best_adj]
        adj_copy, cnt = local_search(adj_copy, n, 15000, seed)
        if cnt > best_count:
            best_count = cnt
            best_adj = [set(s) for s in adj_copy]
    
    # Another round of systematic addition
    best_adj = systematic_edge_addition(best_adj, n)
    best_count = count_edges(best_adj, n)
    
    # Final local search
    for seed in [777, 888, 999, 111, 222]:
        adj_copy = [set(s) for s in best_adj]
        adj_copy, cnt = local_search(adj_copy, n, 20000, seed)
        if cnt > best_count:
            best_count = cnt
            best_adj = [set(s) for s in adj_copy]
    
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
    return edges
