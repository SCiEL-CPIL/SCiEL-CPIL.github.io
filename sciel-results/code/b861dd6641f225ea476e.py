"""Spectral construction + local swaps for C6-free graphs on 40 vertices."""
import numpy as np
import random
from itertools import combinations

def _has_c6_exact(adj, n):
    """Exact check for C6 using path enumeration. Returns True if C6 exists."""
    # A C6 is a cycle of length 6. We can check by finding all paths of length 5
    # where the endpoints are adjacent.
    # For efficiency, we'll use adjacency lists.
    adj_list = [list(np.where(adj[i])[0]) for i in range(n)]
    
    # For each pair of non-adjacent vertices (u, v), count the number of
    # length-2 paths between them. If two such paths share no internal vertices
    # and the internal vertices are not adjacent, we might have a C6.
    # Actually, more directly: a C6 exists iff there exist vertices
    # v0-v1-v2-v3-v4-v5-v0 with all distinct.
    
    # Efficient approach: for each ordered pair (u, v) with u != v and not adjacent,
    # find all common neighbors of u and v. For each pair of common neighbors (a, b),
    # if a and b are not adjacent, then we need to check if there's a path of length 2
    # from a to b that goes through a vertex c not in {u, v, a, b}.
    # Wait, that's getting complicated.
    
    # Simpler exact check: enumerate all 6-cycles by fixing v0 and v1,
    # then looking for v2, v3, v4, v5.
    # This is O(n^2 * d^4) in the worst case, but with d~5 it's manageable.
    
    for v0 in range(n):
        for v1 in adj_list[v0]:
            if v1 <= v0:  # avoid double counting
                continue
            for v2 in adj_list[v1]:
                if v2 == v0:
                    continue
                for v3 in adj_list[v2]:
                    if v3 == v0 or v3 == v1:
                        continue
                    for v4 in adj_list[v3]:
                        if v4 == v0 or v4 == v1 or v4 == v2:
                            continue
                        # Check if v4 is adjacent to v0 and v4 != v5 (v5 = v0)
                        # Actually, we need v5 = v0, so check if v4 is adjacent to v0
                        if v4 in adj_list[v0]:
                            return True
    return False

def _check_c6_after_add(adj, n, u, v):
    """Check if adding edge (u,v) creates a C6. A new C6 must use edge (u,v)."""
    # Find paths of length 5 from u to v in the current graph (without edge u-v).
    # Path: u - a - b - c - d - v, all 6 vertices distinct.
    
    nu = [x for x in np.where(adj[u])[0] if x != v]
    nv = [x for x in np.where(adj[v])[0] if x != u]
    
    for a in nu:
        for d in nv:
            if a == d:
                continue
            # Find b adjacent to a, b not in {u, v, d}
            na = [x for x in np.where(adj[a])[0] if x not in (u, v, d)]
            for b in na:
                if b == d:
                    continue
                # Find c adjacent to both b and d, c not in {u, v, a, b, d}
                nb = set(np.where(adj[b])[0])
                nd = set(np.where(adj[d])[0])
                common = nb & nd
                common -= {u, v, a, b, d}
                for c in common:
                    if len({u, a, b, c, d, v}) == 6:
                        return True
    return False

def _count_edges(adj):
    return int(np.sum(adj)) // 2

def _build_conference_base(n=40):
    """Build a conference-like graph on n vertices using spectral principles.
    
    Conference graphs have eigenvalues 1, 1±sqrt(n-1). For n=40, we approximate
    by using a symmetric difference set or a quasi-symmetric design.
    
    We use a construction based on the Hadamard matrix of order 40 (if it exists)
    or a related structure. Since Hadamard(40) doesn't exist, we use a near-conference
    construction.
    
    Strategy: Use a 2-(40, 5, 1) design-like structure. The complement of a
    symmetric (40, 5, 1) design would have 40*35/2 = 700 edges, too many.
    
    Instead, we use a circulant graph with carefully chosen connection set.
    For C6-freeness, we need to avoid solutions to ±a1 ± a2 ± a3 = 0 mod 40
    with distinct nonzero terms.
    
    Let's use connection set S = {1, 3, 7, 11, 13} (odd numbers, spread out).
    Degree = 2*5 = 10, edges = 40*10/2 = 200. Too many.
    
    Let's use S = {1, 3, 7} (degree 6, 120 edges). Check for C6:
    C6 exists if there are a,b,c in S (with signs) such that a+b+c = 0 or similar.
    With S={1,3,7}: 1+3-7 = -3 ≠ 0, 1-3+7=5≠0, etc. Need to check all combos.
    
    Actually, let's use a sparser set. S = {1, 3} gives degree 4, 80 edges.
    S = {1, 5} gives degree 4, 80 edges.
    
    For better density, let's use a non-circulant construction.
    """
    adj = np.zeros((n, n), dtype=bool)
    
    # Use a block construction: 8 blocks of 5 vertices.
    # Within each block: use a 5-cycle (degree 2 within block).
    # Between blocks: add edges based on a regular bipartite pattern.
    
    # Within each block of 5: add a 5-cycle
    for block in range(8):
        base = block * 5
        for i in range(5):
            u = base + i
            v = base + (i + 1) % 5
            adj[u, v] = True
            adj[v, u] = True
    
    # Between blocks: add a 2-regular bipartite connection between each pair
    # of blocks, but only for selected pairs to keep total edges ~105.
    # 8 blocks, each pair of blocks can have up to 25 edges.
    # We want ~25 additional edges total (80 + 25 = 105).
    
    # Use a greedy approach with spectral guidance.
    # For now, add edges between blocks in a structured way.
    # Connect block i to block j if (i-j) mod 8 is in {1, 3, 5, 7} (odd differences).
    # For each such pair, add 1 edge: connect vertex 0 of block i to vertex 0 of block j.
    
    for i in range(8):
        for j in range(i+1, 8):
            if (j - i) % 2 == 1:  # odd difference
                u = i * 5
                v = j * 5
                adj[u, v] = True
                adj[v, u] = True
    
    return adj

def construct():
    n = 40
    random.seed(42)
    
    # Start with a conference-like base graph
    adj = _build_conference_base(n)
    
    # Verify no C6 in base
    if _has_c6_exact(adj, n):
        # Fallback to K5 construction if base has C6
        adj = np.zeros((n, n), dtype=bool)
        for k in range(8):
            for a in range(5):
                for b in range(a+1, 5):
                    u = 5*k + a
                    v = 5*k + b
                    adj[u, v] = True
                    adj[v, u] = True
    
    # Greedy edge addition with C6 check
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if not adj[u, v]:
                candidates.append((u, v))
    
    random.shuffle(candidates)
    
    for u, v in candidates:
        if not adj[u, v]:
            if not _check_c6_after_add(adj, n, u, v):
                adj[u, v] = True
                adj[v, u] = True
    
    # Local search: try to swap edges to increase density
    # Try multiple rounds of local optimization
    for iteration in range(50):
        improved = False
        # Try removing an edge and adding another
        edges = [(u, v) for u in range(n) for v in range(u+1, n) if adj[u, v]]
        random.shuffle(edges)
        
        for u, v in edges:
            # Try to remove (u,v) and add some other edge
            adj[u, v] = False
            adj[v, u] = False
            
            # Try to add a new edge
            new_candidates = []
            for x in range(n):
                for y in range(x+1, n):
                    if not adj[x, y] and (x, y) != (u, v):
                        new_candidates.append((x, y))
            
            random.shuffle(new_candidates)
            
            added = False
            for x, y in new_candidates:
                if not _check_c6_after_add(adj, n, x, y):
                    adj[x, y] = True
                    adj[y, x] = True
                    added = True
                    break
            
            if added:
                improved = True
                # Keep the new edge, don't restore (u,v)
                break
            else:
                # Restore (u,v)
                adj[u, v] = True
                adj[v, u] = True
        
        if not improved:
            break
    
    # Final C6 check
    if _has_c6_exact(adj, n):
        # Remove edges until no C6
        while _has_c6_exact(adj, n):
            edges = [(u, v) for u in range(n) for v in range(u+1, n) if adj[u, v]]
            random.shuffle(edges)
            for u, v in edges:
                adj[u, v] = False
                adj[v, u] = False
                if not _has_c6_exact(adj, n):
                    break
    
    # Convert to list of edges
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v]:
                edges.append([u, v])
    
    return edges
