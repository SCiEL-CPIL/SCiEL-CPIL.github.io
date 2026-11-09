"""Improved program: randomized C4-free graph search maximizing n with alpha < 11."""
import random
import time
import sys

# EVOLVE-BLOCK-START

def _has_c4(adj, n):
    """Check if graph with adjacency sets has a C4. Returns True if C4 found."""
    # For each pair of vertices, count common neighbors
    # C4 exists iff some pair has >= 2 common neighbors
    for u in range(n):
        for v in range(u + 1, n):
            if u in adj[v]:
                # u and v are adjacent; they can't be opposite in C4
                continue
            common = adj[u] & adj[v]
            if len(common) >= 2:
                return True
    return False

def _has_c4_fast(adj, n):
    """Faster C4 check using sorted common neighbors."""
    for u in range(n):
        au = adj[u]
        for v in range(u + 1, n):
            if v in au:
                continue
            # Count common neighbors
            common = 0
            for w in au:
                if w in adj[v]:
                    common += 1
                    if common >= 2:
                        return True
    return False

def _alpha_greedy(adj, n):
    """Greedy estimate of independence number. Returns size of greedy independent set."""
    remaining = set(range(n))
    independent = set()
    while remaining:
        # Pick vertex with fewest neighbors in remaining
        best_v = min(remaining, key=lambda v: len(adj[v] & remaining))
        independent.add(best_v)
        remaining -= adj[best_v]
        remaining.discard(best_v)
    return len(independent)

def _alpha_exact(adj, n):
    """Exact independence number via branch and bound."""
    # Use a simple branch and bound
    best = [0]
    
    def _branch(remaining, current_size):
        if not remaining:
            if current_size > best[0]:
                best[0] = current_size
            return
        if current_size + len(remaining) <= best[0]:
            return
        # Pick vertex
        v = next(iter(remaining))
        # Option 1: include v
        new_remaining = remaining - adj[v]
        new_remaining.discard(v)
        _branch(new_remaining, current_size + 1)
        # Option 2: exclude v
        remaining2 = set(remaining)
        remaining2.discard(v)
        _branch(remaining2, current_size)
    
    _branch(set(range(n)), 0)
    return best[0]

def _find_independent_set_k(adj, n, k):
    """Find independent set of size k if exists. Returns set or None."""
    if k <= 1:
        return set(range(min(k, n)))
    
    # Use greedy with backtracking
    def _search(remaining, chosen, depth):
        if len(chosen) == k:
            return chosen
        if len(remaining) < k - len(chosen):
            return None
        # Prune: if even taking all remaining can't reach k
        if len(chosen) + len(remaining) < k:
            return None
        
        # Pick vertex with fewest constraints
        v = min(remaining, key=lambda x: len(adj[x] & remaining))
        
        # Try including v
        new_remaining = remaining - adj[v]
        new_remaining.discard(v)
        chosen.add(v)
        result = _search(new_remaining, chosen, depth + 1)
        if result is not None:
            return result
        chosen.discard(v)
        
        # Try excluding v
        remaining2 = set(remaining)
        remaining2.discard(v)
        result = _search(remaining2, chosen, depth)
        if result is not None:
            return result
        
        return None
    
    return _search(set(range(n)), set(), 0)

def _build_random_graph(n, edge_prob, seed=42):
    """Build a random C4-free graph with given edge probability."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    
    # Try to add edges randomly, checking C4 constraint
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < edge_prob:
                # Check if adding edge (u,v) creates C4
                # An edge (u,v) creates C4 iff there exists w such that
                # w is adjacent to both u and v, AND there's another path
                # Actually, adding edge (u,v) creates C4 iff there are two
                # distinct common neighbors of u and v... no wait.
                # Adding edge (u,v) creates a C4 iff there exists a vertex w
                # such that w is adjacent to u, and there exists vertex z
                # such that z is adjacent to v, and w-z is an edge, and w != v, z != u.
                # More precisely: C4 = u-w-z-v-u. So we need w adj u, z adj v, w adj z,
                # w != v, z != u, w != z.
                creates_c4 = False
                for w in adj[u]:
                    if w == v:
                        continue
                    for z in adj[v]:
                        if z == u or z == w:
                            continue
                        if z in adj[w]:
                            creates_c4 = True
                            break
                    if creates_c4:
                        break
                if not creates_c4:
                    adj[u].add(v)
                    adj[v].add(u)
    
    return adj

def _try_improve(adj, n, deadline, rng):
    """Try to add edges to break independent sets, maintaining C4-free property."""
    improved = False
    while time.time() < deadline:
        # Find an independent set of size 11
        indep = _find_independent_set_k(adj, n, 11)
        if indep is None:
            return adj, True  # No independent set of size 11, we're done!
        
        # Try to add edges within the independent set to break it
        # But we need to maintain C4-free
        added = False
        vertices = list(indep)
        rng.shuffle(vertices)
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                u, v = vertices[i], vertices[j]
                if v in adj[u]:
                    continue
                # Check if adding (u,v) creates C4
                creates_c4 = False
                for w in adj[u]:
                    if w == v:
                        continue
                    for z in adj[v]:
                        if z == u or z == w:
                            continue
                        if z in adj[w]:
                            creates_c4 = True
                            break
                    if creates_c4:
                        break
                if not creates_c4:
                    adj[u].add(v)
                    adj[v].add(u)
                    added = True
                    break
            if added:
                break
        
        if not added:
            # Can't add any edge within the independent set
            # Try adding edges from independent set to other vertices
            # to reduce the effective independent set
            break
    
    return adj, False

def _verify_graph(adj, n):
    """Verify graph is C4-free and has no independent set of size 11."""
    # Check C4
    for u in range(n):
        for v in range(u + 1, n):
            if v in adj[u]:
                continue
            common = adj[u] & adj[v]
            if len(common) >= 2:
                return False, "C4 found"
    
    # Check independence number
    indep = _find_independent_set_k(adj, n, 11)
    if indep is not None:
        return False, f"Independent set of size 11: {indep}"
    
    return True, "OK"

def construct():
    """Construct a C4-free graph with no independent set of size 11, maximizing n."""
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    best_n = 30
    best_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        best_edges += [[a, b], [a, c], [b, c]]
    
    rng = random.Random(12345)
    
    # Try various sizes from large to small
    for n in range(43, 29, -1):
        if time.time() >= deadline - 20:
            break
        
        # Try multiple random constructions
        attempts = 0
        max_attempts = 50
        while time.time() < deadline - 20 and attempts < max_attempts:
            attempts += 1
            
            # Try different edge densities
            for edge_prob in [0.15, 0.2, 0.25, 0.3, 0.35]:
                if time.time() >= deadline - 20:
                    break
                
                adj = _build_random_graph(n, edge_prob, seed=rng.randint(0, 1000000))
                
                # Check C4-free
                if _has_c4(adj, n):
                    continue
                
                # Check independence number
                indep = _find_independent_set_k(adj, n, 11)
                if indep is None:
                    # Found a valid graph!
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append([u, v])
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                    return {'n': n, 'edges': edges}
            
            # Try improvement from a random graph
            adj = _build_random_graph(n, 0.2, seed=rng.randint(0, 1000000))
            if not _has_c4(adj, n):
                adj, success = _try_improve(adj, n, deadline - 20, rng)
                if success:
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append([u, v])
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                    return {'n': n, 'edges': edges}
        
        # If we get here, try a different approach for this n
        # Build from known structure: use projective plane or other designs
    
    # Fallback: try to build from smaller valid graphs
    # Use the triangle construction as base and try to extend
    if time.time() < deadline - 10:
        # Try n=31-35 with structured approach
        for n in range(35, 30, -1):
            if time.time() >= deadline - 10:
                break
            
            # Start with triangles and add vertices
            adj = [set() for _ in range(n)]
            for t in range(10):
                a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
                adj[a].update([b, c])
                adj[b].update([a, c])
                adj[c].update([a, b])
            
            # Add extra vertices with random connections
            for v in range(30, n):
                # Connect to random subset of existing vertices
                candidates = list(range(v))
                rng.shuffle(candidates)
                for u in candidates:
                    if len(adj[v]) >= 5:  # Limit degree
                        break
                    if rng.random() < 0.1:
                        # Check C4
                        creates_c4 = False
                        for w in adj[u]:
                            if w == v:
                                continue
                            for z in adj[v]:
                                if z == u or z == w:
                                    continue
                                if z in adj[w]:
                                    creates_c4 = True
                                    break
                            if creates_c4:
                                break
                        if not creates_c4:
                            adj[u].add(v)
                            adj[v].add(u)
            
            # Verify and try to fix
            if not _has_c4(adj, n):
                indep = _find_independent_set_k(adj, n, 11)
                if indep is None:
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append([u, v])
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                    return {'n': n, 'edges': edges}
    
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
