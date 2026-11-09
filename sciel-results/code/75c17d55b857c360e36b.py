"""
Hexagon-free graph construction on 40 vertices.
Strategy: Start from 8 disjoint K5s (40 edges), then greedily add cross-edges
in an order that minimizes C6-creation risk, followed by local search (swap moves)
to maximize edge count.
"""
import time
import random
import numpy as np
from itertools import combinations

N = 40
DEADLINE = 140  # Safety margin before 150s kill


def _has_c6(adj, n):
    """Check if graph has a C6. adj is list of bitmasks."""
    # For each pair of vertices at distance 3, check if they can be connected
    # to form a C6.
    # A C6 is u-v-w-x-y-z-u where all distinct.
    # Alternative: enumerate all 6-cycles. Too slow for large graphs.
    # Better: for each edge (u,v), check if there is a path of length 4 from v to u
    # that avoids u and v internally and uses distinct vertices.
    
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] >> v) & 1:
                continue
            # Need path v -> w -> x -> y -> z -> u
            # w, x, y, z distinct from each other and from u, v
            # w in N(v), x in N(w)\{v}, y in N(x)\{v,w}, z in N(y)\{v,w,x}, u in N(z)\{v,w,x,y}
            
            # Use BFS-like enumeration but with pruning
            # Since n=40, this can be expensive. Let's use a smarter approach.
            
            # Actually, let's use the fact that if we have edges (u,v), we look for
            # a path of length 4 from v to u in the graph with u and v removed from
            # internal nodes.
            
            # For efficiency, precompute adjacency lists
            pass
    
    # Let's use a different approach: enumerate all 6-cycles by fixing one edge
    # and looking for the other 5 edges.
    
    # Actually, for n=40, we can afford a more direct check.
    # For each pair of vertices (a,b) at distance 2 (i.e., share a common neighbor),
    # and each pair (c,d) at distance 2, check if they can form a C6.
    
    # Better: For each triple of vertices (w, x, y) that form a path w-x-y,
    # and each triple (a, b, c) that form a path a-b-c, check if they can be
    # connected to form a C6.
    
    # Let's use the standard approach: for each pair of non-adjacent vertices u, v,
    # check if there are two disjoint paths of length 3 from u to v.
    
    # Actually, the most efficient way for n=40 is to use matrix multiplication
    # or bitset operations.
    
    # Let's use bitset operations.
    # A C6 exists if there are vertices u,v,w,x,y,z all distinct such that
    # u~v, v~w, w~x, x~y, y~z, z~u.
    
    # Enumerate u, v (edge), then w in N(v)\{u}, x in N(w)\{u,v}, y in N(x)\{u,v,w},
    # z in N(y)\{u,v,w,x}, and check if z~u.
    
    for u in range(n):
        nu = adj[u]
        for v in range(u + 1, n):
            if not (nu >> v) & 1:
                continue
            nv = adj[v]
            # w in N(v) \ {u}
            nw_candidates = nv & ~(1 << u)
            while nw_candidates:
                lsb = nw_candidates & (-nw_candidates)
                w = lsb.bit_length() - 1
                nw_candidates ^= lsb
                nw = adj[w]
                # x in N(w) \ {u, v}
                nx_candidates = nw & ~(1 << u) & ~(1 << v)
                while nx_candidates:
                    lsb2 = nx_candidates & (-nx_candidates)
                    x = lsb2.bit_length() - 1
                    nx_candidates ^= lsb2
                    nx = adj[x]
                    # y in N(x) \ {u, v, w}
                    ny_candidates = nx & ~(1 << u) & ~(1 << v) & ~(1 << w)
                    while ny_candidates:
                        lsb3 = ny_candidates & (-ny_candidates)
                        y = lsb3.bit_length() - 1
                        ny_candidates ^= lsb3
                        ny = adj[y]
                        # z in N(y) \ {u, v, w, x}
                        nz_candidates = ny & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x)
                        # Check if any z in nz_candidates is adjacent to u
                        if nz_candidates & nu:
                            return True
    return False


def _creates_c6(adj, n, u, v):
    """Check if adding edge (u,v) creates a C6."""
    if (adj[u] >> v) & 1:
        return True  # Edge already exists
    
    # Adding (u,v) creates a C6 if there is a path of length 4 from u to v
    # that uses distinct vertices (excluding u and v).
    # Path: u -> a -> b -> c -> d -> v
    # a in N(u), b in N(a)\{u}, c in N(b)\{u,a}, d in N(c)\{u,a,b}, v in N(d)\{u,a,b,c}
    
    # For efficiency, we can check: is there a path of length 4 from u to v
    # in the graph with u and v removed from internal nodes?
    
    # Let's enumerate:
    nu = adj[u]
    # a in N(u)
    for a in range(n):
        if a == u or a == v:
            continue
        if not (nu >> a) & 1:
            continue
        na = adj[a]
        # b in N(a) \ {u}
        for b in range(n):
            if b == u or b == v or b == a:
                continue
            if not (na >> b) & 1:
                continue
            nb = adj[b]
            # c in N(b) \ {u, a}
            for c in range(n):
                if c == u or c == v or c == a or c == b:
                    continue
                if not (nb >> c) & 1:
                    continue
                nc = adj[c]
                # d in N(c) \ {u, a, b}
                # Check if d is adjacent to v
                dv_candidates = nc & ~(1 << u) & ~(1 << a) & ~(1 << b)
                if dv_candidates & adj[v]:
                    return True
    return False


def _seed_8k5(n):
    """Create 8 disjoint K5s on n vertices."""
    edges = []
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append([base + a, base + b])
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return edges, adj


def _greedy_add_edges(edges, adj, n, time_limit, seed=42):
    """Greedily add edges in an order that minimizes C6 risk."""
    start = time.time()
    rng = random.Random(seed)
    
    # Generate all possible edges not in current graph
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in edge_set]
    rng.shuffle(all_pairs)
    
    # Score each edge by how "safe" it is to add
    # Heuristic: prefer edges between vertices with lower common neighborhood
    def edge_safety(u, v):
        common = adj[u] & adj[v]
        return bin(common).count('1')
    
    # Sort by safety (lower is safer)
    all_pairs.sort(key=lambda p: edge_safety(p[0], p[1]))
    
    added = 0
    for u, v in all_pairs:
        if time.time() - start > time_limit:
            break
        if (adj[u] >> v) & 1:
            continue
        if _creates_c6(adj, n, u, v):
            continue
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edges.append([u, v])
        added += 1
    
    return edges, adj, added


def _local_search(edges, adj, n, time_limit, seed=42):
    """Local search using swap moves: remove one edge, add another."""
    start = time.time()
    rng = random.Random(seed)
    
    best_edges = list(edges)
    best_count = len(edges)
    best_adj = [a for a in adj]
    
    # Precompute edge set
    def build_edge_set(edges):
        s = set()
        for u, v in edges:
            if u > v:
                u, v = v, u
            s.add((u, v))
        return s
    
    edge_set = build_edge_set(edges)
    
    # Degrees
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    
    while time.time() - start < time_limit:
        # Try random swap: remove edge e1, add edge e2
        if len(edges) == 0:
            break
            
        e1_idx = rng.randrange(len(edges))
        u1, v1 = edges[e1_idx]
        
        # Try to add a random non-edge
        attempts = 0
        while attempts < 50:
            u2, v2 = rng.randrange(n), rng.randrange(n)
            if u2 == v2 or u2 > v2:
                u2, v2 = min(u2, v2), max(u2, v2)
            if (u2, v2) in edge_set:
                attempts += 1
                continue
            if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                attempts += 1
                continue
            
            # Check if removing e1 and adding e2 keeps C6-free
            # First, remove e1
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            
            if not _creates_c6(adj, n, u2, v2):
                # Perform swap
                adj[u2] |= (1 << v2)
                adj[v2] |= (1 << u2)
                
                edges[e1_idx] = [u2, v2]
                edge_set.discard((u1, v1))
                edge_set.discard((v1, u1))
                edge_set.add((u2, v2))
                
                deg[u1] -= 1
                deg[v1] -= 1
                deg[u2] += 1
                deg[v2] += 1
                
                best_edges = list(edges)
                best_count = len(edges)
                best_adj = [a for a in adj]
            else:
                # Revert
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
            
            attempts += 1
        
        # Also try single-edge additions
        for _ in range(20):
            u2, v2 = rng.randrange(n), rng.randrange(n)
            if u2 == v2 or u2 > v2:
                u2, v2 = min(u2, v2), max(u2, v2)
            if (u2, v2) in edge_set:
                continue
            if _creates_c6(adj, n, u2, v2):
                continue
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            edges.append([u2, v2])
            edge_set.add((u2, v2))
            deg[u2] += 1
            deg[v2] += 1
            
            best_edges = list(edges)
            best_count = len(edges)
            best_adj = [a for a in adj]
    
    return best_edges, best_adj, best_count


def construct():
    n = N
    start_time = time.time()
    
    # Phase 1: Seed with 8 K5s
    edges, adj = _seed_8k5(n)
    elapsed = time.time() - start_time
    
    # Phase 2: Greedy add edges
    remaining = min(30, DEADLINE - elapsed - 20)
    if remaining > 5:
        edges, adj, added = _greedy_add_edges(edges, adj, n, remaining, seed=42)
    
    elapsed = time.time() - start_time
    
    # Phase 3: Local search
    remaining = min(60, DEADLINE - elapsed - 10)
    if remaining > 5:
        edges, adj, count = _local_search(edges, adj, n, remaining, seed=42)
    
    # Final verification
    if _has_c6(adj, n):
        # Fallback: use 8 K5s
        edges, adj = _seed_8k5(n)
    
    return edges
