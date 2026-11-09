import time
import random
import sys

def _alpha_bnb_fast(adj, n, limit=10):
    """
    Fast branch-and-bound: returns True if alpha >= limit, False if alpha < limit.
    Also returns the max independent set size found (capped at limit).
    """
    best = [0]
    
    def bnb(cand, sz):
        if sz >= limit:
            best[0] = limit
            return True
        if sz + cand.bit_count() < limit:
            return False
        if cand == 0:
            if sz > best[0]:
                best[0] = sz
            return False
        
        # Pick vertex with max degree in cand for better pruning
        best_v = -1
        best_deg = -1
        c = cand
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            deg = (adj[v] & cand).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            c ^= lsb
        
        if best_v == -1:
            return False

        v_bit = 1 << best_v
        
        # Branch 1: Include best_v (more promising for finding large IS)
        if bnb(cand & ~v_bit & ~adj[best_v], sz + 1):
            return True
        
        # Branch 2: Exclude best_v
        if bnb(cand & ~v_bit, sz):
            return True
        
        if sz > best[0]:
            best[0] = sz
        return False

    bnb((1 << n) - 1, 0)
    return best[0]

def _is_triangle_free_fast(adj, n):
    """Check triangle-free using bitmask operations."""
    for u in range(n):
        v_bits = adj[u]
        while v_bits:
            lsb = v_bits & -v_bits
            v = lsb.bit_length() - 1
            if u < v:
                if (adj[u] & adj[v]) != 0:
                    return False
            v_bits ^= lsb
    return True

def _find_independent_set(adj, n, limit=10):
    """Find an independent set of size limit, or None."""
    found = [None]
    
    def bnb(cand, sz, path):
        if found[0] is not None:
            return
        if sz >= limit:
            found[0] = list(path)
            return
        if sz + cand.bit_count() < limit:
            return
        if cand == 0:
            return
        
        # Pick vertex with min degree in cand
        best_v = -1
        best_deg = n + 1
        c = cand
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            deg = (adj[v] & cand).bit_count()
            if deg < best_deg:
                best_deg = deg
                best_v = v
            c ^= lsb
        
        if best_v == -1:
            return

        v_bit = 1 << best_v
        
        path.append(best_v)
        bnb(cand & ~v_bit & ~adj[best_v], sz + 1, path)
        path.pop()
        
        if found[0] is not None:
            return
        bnb(cand & ~v_bit, sz, path)
        
    bnb((1 << n) - 1, 0, [])
    return found[0]

def _try_cayley_graph(S, n):
    """Try a Cayley graph on Z_n with connection set S."""
    S_set = set(S)
    # Check sum-free: no s1+s2 in S for s1,s2 in S
    for s1 in S_set:
        for s2 in S_set:
            if (s1 + s2) % n in S_set:
                return None
    
    # Build adjacency
    adj = [0] * n
    for i in range(n):
        for s in S_set:
            j = (i + s) % n
            if j != i:
                adj[i] |= (1 << j)
    
    # Verify triangle-free (should be by construction)
    if not _is_triangle_free_fast(adj, n):
        return None
    
    # Check independence number
    alpha = _alpha_bnb_fast(adj, n, limit=10)
    if alpha < 10:
        edges = []
        for u in range(n):
            v_bits = adj[u]
            while v_bits:
                lsb = v_bits & -v_bits
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                v_bits ^= lsb
        return {'n': n, 'edges': edges}
    return None

def _random_triangle_free(n, seed):
    """Generate a random triangle-free graph on n vertices."""
    rng = random.Random(seed)
    adj = [0] * n
    
    # Random edge order
    edges_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edges_list)
    
    for u, v in edges_list:
        if (adj[u] & adj[v]) == 0:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    return adj

def _repair_graph(adj, n, rng, max_iterations=100):
    """
    Try to reduce alpha below 10 by adding edges within independent sets.
    Returns modified adj if successful, else None.
    """
    for _ in range(max_iterations):
        alpha = _alpha_bnb_fast(adj, n, limit=10)
        if alpha < 10:
            return adj
        
        # Find an independent set of size 10
        iso_set = _find_independent_set(adj, n, limit=10)
        if iso_set is None:
            break
        
        rng.shuffle(iso_set)
        
        # Try to add an edge within iso_set that doesn't create a triangle
        modified = False
        for i in range(len(iso_set)):
            for j in range(i + 1, len(iso_set)):
                u, v = iso_set[i], iso_set[j]
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    modified = True
                    break
            if modified:
                break
        
        if not modified:
            # If we can't add an edge, try removing a vertex from the set?
            # Or just break and restart
            break
    
    alpha = _alpha_bnb_fast(adj, n, limit=10)
    if alpha < 10:
        return adj
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Fallback: Andrasfai graph And(3) on 26 vertices? 
    # Actually And(9) is n=27.
    # Let's use And(9) as a stronger fallback.
    # And(k) on 3k vertices. Edges (i, i+1)...(i, i+k-1) mod 3k?
    # No, that creates triangles.
    # The Andrásfai graph And(k) is the circulant C_{3k}(1, 2, ..., k-1) is NOT triangle free.
    # The actual And(k) is defined differently.
    # Let's stick to the 26-vertex one which is known to be valid.
    n_fallback = 26
    edges_fallback = [[i, j] for i in range(n_fallback) for j in range(i + 1, n_fallback) if (j - i) % n_fallback % 3 == 1]
    
    # Strategy 1: Try algebraic Cayley graph constructions
    cayley_candidates = [
        (39, set(range(13, 26))),
        (39, set(range(14, 26))),
        (39, set(range(12, 26))),
        (38, set(range(13, 25))),
        (37, set(range(12, 24))),
    ]
    
    for n, S in cayley_candidates:
        if time.time() > deadline:
            break
        result = _try_cayley_graph(S, n)
        if result:
            return result
    
    # Strategy 2: Random search with repair
    random.seed(42)
    
    for n_target in [39, 38, 37]:
        if time.time() > deadline:
            break
        
        max_attempts = 100 if n_target == 39 else 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            adj = _random_triangle_free(n_target, seed=42 + attempt)
            
            # Quick check: is alpha < 10?
            alpha = _alpha_bnb_fast(adj, n_target, limit=10)
            if alpha < 10:
                edges = []
                for u in range(n_target):
                    v_bits = adj[u]
                    while v_bits:
                        lsb = v_bits & -v_bits
                        v = lsb.bit_length() - 1
                        if u < v:
                            edges.append([u, v])
                        v_bits ^= lsb
                return {'n': n_target, 'edges': edges}
            
            # Try to repair
            repaired = _repair_graph(adj, n_target, random, max_iterations=50)
            if repaired is not None:
                edges = []
                for u in range(n_target):
                    v_bits = repaired[u]
                    while v_bits:
                        lsb = v_bits & -v_bits
                        v = lsb.bit_length() - 1
                        if u < v:
                            edges.append([u, v])
                        v_bits ^= lsb
                return {'n': n_target, 'edges': edges}
    
    # Return fallback
    return {'n': n_fallback, 'edges': edges_fallback}
