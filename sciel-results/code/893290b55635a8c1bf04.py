"""Construct C4-free graph with high min degree using polarity graph seed + local search."""
import time
import random

def _is_c4_free(n, adj):
    """Check C4-free using bitsets: for every pair (u,v), |N(u)∩N(v)| <= 1."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            # Count bits in common
            c = common
            cnt = 0
            while c:
                cnt += 1
                c &= c - 1
            if cnt >= 2:
                return False
    return True

def _min_degree(n, adj):
    md = n
    for i in range(n):
        d = bin(adj[i]).count('1')
        if d < md:
            md = d
    return md

def _can_add_edge(u, v, adj):
    """O(deg) check: does adding (u,v) create a C4?
    C4 u-v-w-x-u requires w in N(v)\\{u}, x in N(u)\\{v}, w~x, w!=x."""
    A = adj[v] & ~(1 << u)
    B = adj[u] & ~(1 << v)
    a = A
    while a:
        lb = a & (-a)
        w = lb.bit_length() - 1
        a ^= lb
        if adj[w] & B & ~(1 << w):
            return False
    return True

def _polarity_graph(q):
    """
    Build the polarity graph of Q^+(3, q) over F_q.
    Vertices = F_q x F_q (q^2 vertices).
    (a,b) ~ (c,d) iff a*c - b*d = 1  (symmetric for odd q).
    This graph is C4-free and strongly regular with degree q-1.
    """
    n = q * q
    # Modular inverse table for F_q (q is prime)
    inv = [0] * q
    for i in range(1, q):
        # Fermat's little theorem: i^(q-2) = i^{-1} mod q
        inv[i] = pow(i, q - 2, q)

    adj = [0] * n
    for a in range(q):
        for b in range(q):
            u = a * q + b
            for c in range(q):
                for d in range(q):
                    v = c * q + d
                    if u == v:
                        continue
                    # Check a*c - b*d == 1 mod q
                    if (a * c - b * d) % q == 1:
                        adj[u] |= 1 << v
    return n, adj

def _try_add_edges(adj, n, min_deg_needed, deadline, rng):
    """Try to add edges to raise minimum degree, maintaining C4-free."""
    deg = [bin(adj[i]).count('1') for i in range(n)]
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        cur_min = min(deg)
        if cur_min >= min_deg_needed:
            return True
        
        # Find low-degree vertices
        low_verts = [i for i in range(n) if deg[i] == cur_min]
        rng.shuffle(low_verts)
        
        improved = False
        for u in low_verts:
            if time.time() > deadline:
                break
            
            # Candidates: non-neighbors of u
            candidates = []
            for v in range(n):
                if v != u and not (adj[u] >> v) & 1:
                    candidates.append(v)
            rng.shuffle(candidates)
            
            for v in candidates:
                if _can_add_edge(u, v, adj):
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    deg[u] += 1
                    deg[v] += 1
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try swapping: remove an edge from a low vertex and add another
            # For now, just break
            break
    
    return min(deg) >= min_deg_needed

def _extract_edges(n, adj):
    edges = []
    for u in range(n):
        a = adj[u]
        while a:
            lb = a & (-a)
            v = lb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            a ^= lb
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    best_n = 39
    best_edges = []
    
    def elapsed():
        return time.time() - start_time
    
    # Strategy: try polarity graphs for various primes q, take induced subgraphs
    # For target n, we need a C4-free graph on n vertices with min degree >= n-39.
    # Polarity graph Q+(3,q) has q^2 vertices, degree q-1.
    # If we take an induced subgraph on n vertices, min degree could drop.
    # Better: use the full polarity graph if q^2 <= 47, or find good subgraphs.
    
    # q=7: 49 vertices, degree 6. Too many vertices (need <=47).
    # q=5: 25 vertices, degree 4. Too few.
    # So we need to combine or use different approach.
    
    # Alternative: Use the polarity graph as a seed and then ADD edges via local search.
    # For n=46 or 47, min_deg_needed = 7 or 8.
    
    # Let's try: start with a random graph, but seed with structured edges from polarity graph
    # where vertices map to F_q^2.
    
    # Actually, let's try a different structural approach:
    # Use a finite geometry construction directly for the target n.
    
    # For n=47: min_deg_needed = 8
    # For n=46: min_deg_needed = 7
    
    # Approach: Start with empty graph, add edges greedily but with structured priority.
    # Use the polarity graph adjacency to determine edge ordering.
    
    # Let's try multiple strategies:
    
    # Strategy A: Polarity graph seed + edge addition
    # Map n vertices to a subset of F_q^2, use polarity adjacency as initial edges,
    # then add more edges.
    
    primes_to_try = [7, 11, 13]
    
    for target_n in range(47, 39, -1):
        if elapsed() > deadline:
            break
        min_deg_needed = target_n - 39
        
        found = False
        
        for q in primes_to_try:
            if elapsed() > deadline:
                break
            
            # Get full polarity graph
            qn, full_adj = _polarity_graph(q)
            
            # We need to select target_n vertices from qn vertices
            # and possibly add edges between them.
            # The induced subgraph on any subset is C4-free (subgraph of C4-free).
            # But min degree in induced subgraph may be low.
            
            # Strategy: pick vertices that have high degree in the induced subgraph.
            # Or: use all qn vertices if qn <= target_n, pad with isolated? No.
            
            # Better: if qn >= target_n, pick target_n vertices.
            # If qn < target_n, we can't use this directly.
            
            if qn < target_n:
                continue
            
            # Try random subsets of target_n vertices
            for attempt in range(200):
                if elapsed() > deadline:
                    break
                
                rng = random.Random(hash((q, attempt, target_n)) & 0xFFFFFFFF)
                
                # Pick target_n vertices from qn
                selected = rng.sample(range(qn), target_n)
                sel_set = set(selected)
                sel_to_idx = {v: i for i, v in enumerate(selected)}
                
                # Build induced subgraph
                adj = [0] * target_n
                for i, u in enumerate(selected):
                    a = full_adj[u]
                    while a:
                        lb = a & (-a)
                        v = lb.bit_length() - 1
                        a ^= lb
                        if v in sel_to_idx:
                            j = sel_to_idx[v]
                            adj[i] |= 1 << j
                
                deg = [bin(adj[i]).count('1') for i in range(target_n)]
                cur_min = min(deg)
                
                if cur_min >= min_deg_needed:
                    if _is_c4_free(target_n, adj):
                        best_n = target_n
                        best_edges = _extract_edges(target_n, adj)
                        found = True
                        break
                
                # Try to add edges to improve min degree
                _try_add_edges(adj, target_n, min_deg_needed, deadline, rng)
                
                cur_min = _min_degree(target_n, adj)
                if cur_min >= min_deg_needed:
                    if _is_c4_free(target_n, adj):
                        best_n = target_n
                        best_edges = _extract_edges(target_n, adj)
                        found = True
                        break
        
        if found:
            break
        
        # Strategy B: Pure random greedy with C4 check (fallback)
        if elapsed() > deadline:
            break
        
        for seed in range(300):
            if elapsed() > deadline:
                break
            
            rng = random.Random(seed * 7919 + target_n)
            adj = [0] * target_n
            deg = [0] * target_n
            
            # Generate all edges, shuffle
            all_edges = [(u, v) for u in range(target_n) for v in range(u + 1, target_n)]
            rng.shuffle(all_edges)
            
            for u, v in all_edges:
                if elapsed() > deadline:
                    break
                if _can_add_edge(u, v, adj):
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    deg[u] += 1
                    deg[v] += 1
            
            cur_min = min(deg)
            if cur_min >= min_deg_needed:
                if _is_c4_free(target_n, adj):
                    best_n = target_n
                    best_edges = _extract_edges(target_n, adj)
                    found = True
                    break
            
            # Local search: add more edges
            _try_add_edges(adj, target_n, min_deg_needed, deadline, rng)
            cur_min = _min_degree(target_n, adj)
            if cur_min >= min_deg_needed:
                if _is_c4_free(target_n, adj):
                    best_n = target_n
                    best_edges = _extract_edges(target_n, adj)
                    found = True
                    break
        
        if found:
            break
    
    if best_n >= 40:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
