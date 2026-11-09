"""
Improved program: C4-free graph search maximizing n with alpha < 11.
Uses polarity graph PG(2,5) as a structured base (31 vertices, C4-free)
combined with randomized construction and local improvement.
"""
import random
import time
import sys

# EVOLVE-BLOCK-START

def _generate_polarity_graph_q5():
    """
    Generate the polarity graph of PG(2, 5) using GF(5).
    Returns adjacency list as a list of sets.
    Vertices: 31 points in PG(2,5).
    Polarity: x1*y1 + x2*y2 + x3*y3 = 0 (mod 5)
    """
    q = 5
    points = []
    # Affine points: (x, y, 1)
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    # Points at infinity: (x, 1, 0)
    for x in range(q):
        points.append((x, 1, 0))
    # Point at infinity: (1, 0, 0)
    points.append((1, 0, 0))
    
    n = len(points)  # 31
    adj = [set() for _ in range(n)]
    
    for i, p in enumerate(points):
        for j in range(i + 1, n):
            r = points[j]
            # Check if p is on r^perp: p . r = 0 mod q
            if (p[0] * r[0] + p[1] * r[1] + p[2] * r[2]) % q == 0:
                adj[i].add(j)
                adj[j].add(i)
    
    return adj

def _has_c4(adj, n):
    """Check if graph with adjacency sets has a C4. Returns True if C4 found."""
    for u in range(n):
        au = adj[u]
        for v in range(u + 1, n):
            if v in au:
                continue
            common = 0
            for w in au:
                if w in adj[v]:
                    common += 1
                    if common >= 2:
                        return True
    return False

def _find_independent_set_k(adj, n, k):
    """Find independent set of size k if exists. Returns set or None."""
    if k <= 1:
        return set(range(min(k, n)))
    
    def _search(remaining, chosen):
        if len(chosen) == k:
            return chosen.copy()
        if len(remaining) < k - len(chosen):
            return None
        if len(chosen) + len(remaining) < k:
            return None
        
        # Pick vertex with fewest neighbors in remaining (to maximize options)
        v = min(remaining, key=lambda x: len(adj[x] & remaining))
        
        # Try including v
        new_remaining = remaining - adj[v]
        new_remaining.discard(v)
        chosen.add(v)
        result = _search(new_remaining, chosen)
        if result is not None:
            return result
        chosen.discard(v)
        
        # Try excluding v
        remaining2 = set(remaining)
        remaining2.discard(v)
        result = _search(remaining2, chosen)
        if result is not None:
            return result
        
        return None
    
    return _search(set(range(n)), set())

def _check_add_edge_c4(adj, u, v, n):
    """Check if adding edge (u,v) creates a C4."""
    # C4 = u-w-z-v-u where w in N(u), z in N(v), w-z edge
    # w != v, z != u, w != z
    for w in adj[u]:
        if w == v:
            continue
        for z in adj[v]:
            if z == u or z == w:
                continue
            if z in adj[w]:
                return True
    return False

def _build_random_graph(n, edge_prob, seed=42):
    """Build a random C4-free graph with given edge probability."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < edge_prob:
                if not _check_add_edge_c4(adj, u, v, n):
                    adj[u].add(v)
                    adj[v].add(u)
    
    return adj

def _try_improve(adj, n, deadline, rng):
    """Try to add edges to break independent sets, maintaining C4-free property."""
    improved = False
    while time.time() < deadline:
        indep = _find_independent_set_k(adj, n, 11)
        if indep is None:
            return adj, True
        
        added = False
        vertices = list(indep)
        rng.shuffle(vertices)
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                u, v = vertices[i], vertices[j]
                if v in adj[u]:
                    continue
                if not _check_add_edge_c4(adj, u, v, n):
                    adj[u].add(v)
                    adj[v].add(u)
                    added = True
                    break
            if added:
                break
        
        if not added:
            break
    
    return adj, False

def _verify_graph(adj, n):
    """Verify graph is C4-free and has no independent set of size 11."""
    if _has_c4(adj, n):
        return False, "C4 found"
    indep = _find_independent_set_k(adj, n, 11)
    if indep is not None:
        return False, f"Independent set of size 11"
    return True, "OK"

def _edges_from_adj(adj, n):
    """Convert adjacency list to edge list."""
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    return edges

def construct():
    """Construct a C4-free graph with no independent set of size 11, maximizing n."""
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    best_n = 30
    best_edges = []
    # Fallback: 10 triangles
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        best_edges += [[a, b], [a, c], [b, c]]
    
    rng = random.Random(12345)
    
    # Strategy 1: Use PG(2,5) polarity graph (31 vertices) as base
    # and try to extend it
    adj_31 = _generate_polarity_graph_q5()
    n_31 = 31
    
    # Verify the base graph
    if not _has_c4(adj_31, n_31):
        indep_31 = _find_independent_set_k(adj_31, n_31, 11)
        if indep_31 is None:
            # Perfect! 31 vertices with alpha < 11
            edges = _edges_from_adj(adj_31, n_31)
            if n_31 > best_n:
                best_n = n_31
                best_edges = edges
            return {'n': n_31, 'edges': edges}
    
    # Try to extend the 31-vertex graph by adding more vertices
    for n in range(43, 30, -1):
        if time.time() >= deadline - 20:
            break
        
        # Try extending from PG(2,5) base
        if n >= 31:
            adj = [set(s) for s in adj_31]  # Copy base
            # Add new vertices
            for v in range(31, n):
                # Try to connect to existing vertices
                candidates = list(range(v))
                rng.shuffle(candidates)
                for u in candidates:
                    if len(adj[v]) >= 4:
                        break
                    if rng.random() < 0.15:
                        if not _check_add_edge_c4(adj, u, v, n):
                            adj[u].add(v)
                            adj[v].add(u)
            
            if not _has_c4(adj, n):
                indep = _find_independent_set_k(adj, n, 11)
                if indep is None:
                    edges = _edges_from_adj(adj, n)
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                    return {'n': n, 'edges': edges}
                # Try to improve
                if time.time() < deadline - 20:
                    adj_imp, success = _try_improve(adj, n, deadline - 20, rng)
                    if success:
                        edges = _edges_from_adj(adj_imp, n)
                        if n > best_n:
                            best_n = n
                            best_edges = edges
                        return {'n': n, 'edges': edges}
        
        # Try random constructions
        max_attempts = 30
        attempts = 0
        while time.time() < deadline - 20 and attempts < max_attempts:
            attempts += 1
            
            for edge_prob in [0.15, 0.2, 0.25, 0.3, 0.35]:
                if time.time() >= deadline - 20:
                    break
                
                adj = _build_random_graph(n, edge_prob, seed=rng.randint(0, 1000000))
                
                if _has_c4(adj, n):
                    continue
                
                indep = _find_independent_set_k(adj, n, 11)
                if indep is None:
                    edges = _edges_from_adj(adj, n)
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                    return {'n': n, 'edges': edges}
            
            # Try improvement from random graph
            adj = _build_random_graph(n, 0.25, seed=rng.randint(0, 1000000))
            if not _has_c4(adj, n):
                adj_imp, success = _try_improve(adj, n, deadline - 20, rng)
                if success:
                    edges = _edges_from_adj(adj_imp, n)
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                    return {'n': n, 'edges': edges}
        
        # Try structured approach: triangles + random extensions
        if time.time() < deadline - 20:
            adj = [set() for _ in range(n)]
            for t in range(10):
                a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
                adj[a].update([b, c])
                adj[b].update([a, c])
                adj[c].update([a, b])
            
            for v in range(30, n):
                candidates = list(range(v))
                rng.shuffle(candidates)
                for u in candidates:
                    if len(adj[v]) >= 5:
                        break
                    if rng.random() < 0.1:
                        if not _check_add_edge_c4(adj, u, v, n):
                            adj[u].add(v)
                            adj[v].add(u)
            
            if not _has_c4(adj, n):
                indep = _find_independent_set_k(adj, n, 11)
                if indep is None:
                    edges = _edges_from_adj(adj, n)
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                    return {'n': n, 'edges': edges}
    
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
