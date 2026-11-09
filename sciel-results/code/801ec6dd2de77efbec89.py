"""C4-free graph with high minimum degree (R(C4,K1,39) lower bound)."""
import time
import random

# EVOLVE-BLOCK-START

def _can_add_edge(adj, u, v):
    """
    Fast check: does adding edge (u, v) create a C4?
    We assume (u, v) is not currently an edge.
    A C4 would be u - y - x - v - u with x in N(v), y in N(u), and edge (x, y).
    So for any x in N(v), if N(x) intersects N(u), then C4 exists.
    """
    nu = adj[u]
    nv = adj[v]
    x = nv
    while x:
        lsb = x & -x
        xi = lsb.bit_length() - 1
        x ^= lsb
        if adj[xi] & nu:
            return False
    return True

def _can_remove_edge(adj, u, v):
    """
    Check if removing edge (u, v) is safe (it always is for C4-freedom,
    since removing edges can only break C4s, not create them).
    But we need to check if it would help us add another edge.
    This is used in swap operations.
    """
    return True  # Removing an edge never creates a C4

def _try_add_edge(adj, u, v):
    """Try to add edge (u,v) if C4-safe. Returns True if added."""
    if u == v:
        return False
    if (adj[u] >> v) & 1:
        return False
    if _can_add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True
    return False

def _try_remove_edge(adj, u, v):
    """Remove edge (u,v) if it exists. Returns True if removed."""
    if u == v:
        return False
    if not ((adj[u] >> v) & 1):
        return False
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)
    return True

def _min_degree(adj, n):
    md = n
    for a in adj:
        d = a.bit_count()
        if d < md:
            md = d
    return md

def _is_c4_free(adj, n):
    """
    Full verification: for every edge (u, v), the number of common neighbors must be at most 1.
    """
    for u in range(n):
        au = adj[u]
        for v in range(u + 1, n):
            if (au >> v) & 1:
                common = au & adj[v]
                if common & (common - 1):
                    return False
    return True

def _polarity_graph_init(n, rng, q=11):
    """
    Initialize using a polarity graph structure.
    For prime power q, the polarity graph on q+1 vertices is C4-free.
    We use this as a template to build a larger graph.
    
    For n > q+1, we use a block structure: partition vertices into groups
    and use polarity-like connections between groups.
    """
    adj = [0] * n
    
    if n <= q + 1:
        # Use the actual polarity graph on n vertices
        # Polarity graph of the Hermitian form over GF(q)
        # Vertices 0..n-1, edge (i,j) if i != j and (i-j)^2 + 1 == 0 in GF(q)
        # Simpler: use a known C4-free construction
        # For small n, just use greedy
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(all_edges)
        for u, v in all_edges:
            if _can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
    
    # For larger n, use a structured approach:
    # Divide vertices into blocks and use polarity-like connections
    # Each vertex gets high degree by connecting to many other blocks
    
    # Strategy: use a complete multipartite-like structure but C4-free
    # Use the incidence structure of a projective plane or similar
    
    # Simpler: greedy with bias toward high-degree connections
    # Start with a random permutation and add edges greedily
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(all_edges)
    
    # Bias: prefer edges that connect vertices with current low degree
    # But for initialization, just do greedy
    for u, v in all_edges:
        if _can_add_edge(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    return adj

def _local_search_with_swaps(adj, n, target_min_deg, rng, max_iter=500, time_limit=None):
    """
    Local search to increase minimum degree while maintaining C4-freedom.
    Operations:
    1. Add an edge from a low-degree vertex to a non-neighbor if C4-safe.
    2. Swap: remove edge (a, b) and add edge (u, v) if it improves min degree
       and maintains C4-freedom.
    """
    start_time = time.time()
    improved = True
    it = 0
    
    while improved and it < max_iter:
        if time_limit and time.time() - start_time > time_limit:
            break
        improved = False
        it += 1
        degrees = [a.bit_count() for a in adj]
        
        # Find vertices below target
        low_verts = [i for i in range(n) if degrees[i] < target_min_deg]
        
        if not low_verts:
            break
        
        # Operation 1: Add edges to low-degree vertices
        for u in low_verts:
            if degrees[u] >= target_min_deg:
                continue
            non_neighbors = [v for v in range(n) if v != u and not ((adj[u] >> v) & 1)]
            rng.shuffle(non_neighbors)
            
            for v in non_neighbors:
                if degrees[u] >= target_min_deg:
                    break
                if _can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
                    improved = True
        
        # If still below target, try edge swaps
        if min(degrees) < target_min_deg:
            degrees = [a.bit_count() for a in adj]
            low_verts = [i for i in range(n) if degrees[i] < target_min_deg]
            if low_verts:
                # Pick the lowest degree vertex
                u = min(low_verts, key=lambda i: degrees[i])
                if degrees[u] >= target_min_deg:
                    continue
                    
                # Find a high-degree vertex to potentially remove an edge from
                high_verts = [i for i in range(n) if i != u and degrees[i] >= target_min_deg + 1]
                if not high_verts:
                    continue
                    
                # Try swaps: remove (a, b) and add (u, v)
                # where a is high-degree, b is a neighbor of a, v is non-neighbor of u
                a = max(high_verts, key=lambda i: degrees[i])
                neighbors_a = [v for v in range(n) if v != a and ((adj[a] >> v) & 1)]
                rng.shuffle(neighbors_a)
                
                non_neighbors_u = [v for v in range(n) if v != u and not ((adj[u] >> v) & 1)]
                rng.shuffle(non_neighbors_u)
                
                for b in neighbors_a:
                    if b == u:
                        continue
                    for v in non_neighbors_u:
                        if v == a or v == b:
                            continue
                        
                        # Check if removing (a,b) and adding (u,v) is C4-safe
                        # First, temporarily remove (a,b)
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                        
                        # Check if adding (u,v) is C4-safe
                        if _can_add_edge(adj, u, v):
                            # Also need to check that removing (a,b) didn't create issues
                            # (it can't, since removing edges only breaks C4s)
                            # Add (u,v)
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            degrees[u] += 1
                            degrees[v] += 1
                            degrees[a] -= 1
                            degrees[b] -= 1
                            improved = True
                            break
                        else:
                            # Restore (a,b)
                            adj[a] |= (1 << b)
                            adj[b] |= (1 << a)
                    
                    if improved:
                        break
                
                if not improved:
                    # Try a different strategy: just try to add any edge to u
                    # after removing a random edge from some high-degree vertex
                    pass
    
    return adj

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        ai = adj[i]
        for j in range(i + 1, n):
            if (ai >> j) & 1:
                edges.append([i, j])
    return edges

def _polarity_graph_q11(n, rng):
    """
    Construct a C4-free graph using the polarity graph of order q=11.
    The polarity graph has q+1 = 12 vertices and is C4-free with degree (q-1)/2 = 5.
    For larger n, we use a blow-up or product construction.
    
    Actually, for our purposes, let's use a different approach:
    Use the known fact that there exist C4-free graphs with high minimum degree.
    We'll use a greedy approach with a good initialization order.
    """
    adj = [0] * n
    
    # Use a structured ordering: interleave vertices to promote high degree
    # Order vertices in a way that maximizes C4-free edge additions
    order = list(range(n))
    rng.shuffle(order)
    
    # Try multiple orderings
    best_adj = None
    best_min_deg = -1
    
    for attempt in range(3):
        adj_try = [0] * n
        order_try = list(range(n))
        rng.shuffle(order_try)
        
        # Process vertices in order, adding edges to all possible later vertices
        for i in order_try:
            for j in order_try:
                if i == j:
                    continue
                if (adj_try[i] >> j) & 1:
                    continue
                # Only add if j comes after i in some sense, or just check C4
                if _can_add_edge(adj_try, i, j):
                    adj_try[i] |= (1 << j)
                    adj_try[j] |= (1 << i)
        
        min_deg = _min_degree(adj_try, n)
        if min_deg > best_min_deg:
            best_min_deg = min_deg
            best_adj = adj_try[:]
    
    return best_adj

def construct():
    deadline = time.time() + 140.0  # safety margin under 150s
    best_n = 39
    best_edges = []
    best_adj = None
    
    # Try to maximize n from 47 down
    for n in range(47, 38, -1):
        target_min_deg = n - 39
        found = False
        
        # Try multiple strategies
        # Strategy 1: Polarity-inspired initialization + local search
        # Strategy 2: Pure greedy with good ordering
        # Strategy 3: Random restarts with local search
        
        max_seeds = 500 if n >= 46 else 300
        
        for seed in range(max_seeds):
            if time.time() > deadline:
                break
            
            rng = random.Random(seed)
            
            # Initialize with structured approach
            adj = _polarity_graph_q11(n, rng)
            
            # Local search to improve min degree
            time_left = deadline - time.time()
            if time_left > 1:
                adj = _local_search_with_swaps(adj, n, target_min_deg, rng, 
                                               max_iter=300, time_limit=time_left * 0.5)
            
            min_deg = _min_degree(adj, n)
            
            if min_deg >= target_min_deg:
                # Verify C4-freedom
                if _is_c4_free(adj, n):
                    edges = _edges_from_adj(adj, n)
                    
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                        best_adj = adj
                    found = True
                    
                    # If we found n=47, return immediately
                    if n == 47:
                        return {'n': n, 'edges': edges}
        
        if found:
            # Since we iterate n downwards, this is the largest n found
            return {'n': best_n, 'edges': best_edges}
        
        # If not found for this n, continue to smaller n
    
    # Fallback
    if best_adj is not None:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}

# EVOLVE-BLOCK-END
