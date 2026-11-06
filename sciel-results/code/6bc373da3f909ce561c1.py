"""Rewritten program for the suite task; builds a C4-free graph via polarity graph + local search."""
import random
import sys

def _is_prime_power(q):
    """Check if q is a prime power."""
    if q < 2:
        return False
    # Trial division up to sqrt(q)
    i = 2
    while i * i <= q:
        if q % i == 0:
            # Check if q is a power of i
            temp = q
            while temp % i == 0:
                temp //= i
            if temp == 1:
                return True
            return False
        i += 1
    return True  # q is prime

def _largest_prime_power_less_than(n):
    """Find the largest prime power less than n."""
    if n <= 2:
        return 1
    for q in range(n - 1, 1, -1):
        if _is_prime_power(q):
            return q
    return 1

def _build_polarity_graph(q):
    """
    Build the polarity graph of the projective plane of order q.
    The projective plane of order q has q^2 + q + 1 points.
    The polarity graph has n = q^2 + q + 1 vertices.
    Vertex i is adjacent to vertex j if point i lies on line j (under the polarity).
    
    For the standard polarity (orthogonality), in PG(2,q), we can label points as:
    - Affine points: (x, y) for x, y in GF(q) -> indices 0 to q^2-1
    - Points at infinity: one for each direction -> indices q^2 to q^2+q
    - One extra point at infinity (for the 'vertical' direction or similar) -> index q^2+q+1... 
    
    Actually, let's use a simpler combinatorial construction.
    
    Standard polarity graph of PG(2,q):
    n = q^2 + q + 1 vertices.
    Each vertex has degree q + 1.
    The graph is C4-free.
    
    Construction:
    Label vertices 0..n-1.
    For q prime, we can use finite field arithmetic.
    For general prime powers, we'd need GF(q) arithmetic, which is complex.
    Let's stick to prime q for simplicity, or use a known explicit construction.
    
    Actually, for the purpose of this task, we can use a simpler approach:
    Use the incidence graph of a projective plane, but that's bipartite.
    
    Let's use the standard "orthogonality" polarity graph.
    For q = 2: n = 7, it's the Fano plane polarity graph (complement of C7? No, it's a 3-regular graph on 7 vertices, which is the triangular prism? No, it's actually the complement of the 7-cycle? Let's check: degree 3, 7 vertices. The complement has degree 3 as well. It's the Petersen-like? No, it's the 7-vertex graph where each vertex is connected to 3 others. Actually, the polarity graph of the Fano plane is the complement of the 7-cycle C7? No. Let's just compute it properly.
    
    For a general approach without full GF(q) implementation, we can use the following:
    If q is prime, we can use modular arithmetic.
    
    Let's implement for prime q only, and if q is not prime, fall back to a smaller prime.
    """
    n = q * q + q + 1
    edges = []
    
    # For prime q, we can use the standard construction.
    # Points: (x, y) for x,y in Z_q, plus points at infinity.
    # Let's label:
    # 0..q^2-1: affine points (x, y) where index = x*q + y
    # q^2..q^2+q: points at infinity, one for each slope m in Z_q, index = q^2 + m
    # q^2+q: the point at infinity for vertical lines (or the "last" point)
    
    # Actually, the standard polarity in PG(2,q) with q odd is:
    # (x,y,z) is orthogonal to (x',y',z') if xx' + yy' + zz' = 0.
    # With homogeneous coordinates.
    
    # Let's use a simpler explicit construction for the polarity graph.
    # Reference: The polarity graph of PG(2,q) has vertices corresponding to points of PG(2,q).
    # Two points are adjacent if one lies on the polar line of the other.
    
    # For q prime, we can enumerate points as:
    # Type 1: (1, a, b) for a,b in Z_q -> q^2 points
    # Type 2: (0, 1, c) for c in Z_q -> q points
    # Type 3: (0, 0, 1) -> 1 point
    # Total: q^2 + q + 1
    
    # The polar of point P = (x, y, z) is the line x*X + y*Y + z*Z = 0.
    # Point Q = (x', y', z') lies on polar of P if x*x' + y*y' + z*z' = 0.
    
    # So vertex i is adjacent to vertex j if dot product of their homogeneous coordinates is 0 mod q.
    
    # Let's assign coordinates:
    coords = []
    # Type 1: (1, a, b)
    for a in range(q):
        for b in range(q):
            coords.append((1, a, b))
    # Type 2: (0, 1, c)
    for c in range(q):
        coords.append((0, 1, c))
    # Type 3: (0, 0, 1)
    coords.append((0, 0, 1))
    
    # Build edges
    for i in range(n):
        x1, y1, z1 = coords[i]
        for j in range(i + 1, n):
            x2, y2, z2 = coords[j]
            if (x1 * x2 + y1 * y2 + z1 * z2) % q == 0:
                edges.append([i, j])
    
    return n, edges

def _has_c4(n, adj):
    """Check if graph has a C4. adj is a list of sets."""
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            # Check if i and j have 2 common neighbors
            common = adj[i] & adj[j]
            if len(common) >= 2:
                return True
    return False

def _min_degree(n, adj):
    return min(len(adj[i]) for i in range(n))

def _try_find_subgraph(n_full, adj_full, n_target, min_deg_needed, rng):
    """
    Try to find a subset of n_target vertices from n_full such that
    the induced subgraph has min degree >= min_deg_needed.
    Uses local search.
    """
    # Start with a random subset of size n_target
    all_vertices = list(range(n_full))
    rng.shuffle(all_vertices)
    S = set(all_vertices[:n_target])
    D = set(all_vertices[n_target:])
    
    # Function to check if current S is valid
    def is_valid(S):
        for v in S:
            # Count neighbors in S
            deg_S = len(adj_full[v] & S)
            if deg_S < min_deg_needed:
                return False
        return True
    
    # Local search
    max_iters = 5000
    for _ in range(max_iters):
        if is_valid(S):
            return S
        
        # Find a vertex in S with low degree
        min_deg = float('inf')
        worst_v = None
        for v in S:
            deg_S = len(adj_full[v] & S)
            if deg_S < min_deg:
                min_deg = deg_S
                worst_v = v
        
        if min_deg >= min_deg_needed:
            return S
            
        # We need to increase the degree of worst_v.
        # worst_v has min_deg neighbors in S.
        # It has deg_full - min_deg neighbors in D.
        # We want to swap some w in S with u in D such that u is a neighbor of worst_v.
        # This will increase worst_v's degree by 1 (if u is neighbor) and decrease w's degree by 1 (if u is neighbor of w).
        # We want to pick w that is NOT a neighbor of worst_v, so worst_v's degree doesn't drop.
        # And we want u to be a neighbor of worst_v.
        
        neighbors_of_worst_in_D = adj_full[worst_v] & D
        if not neighbors_of_worst_in_D:
            # No neighbor in D to swap with. 
            # This means all neighbors of worst_v are in S.
            # But deg_S(worst_v) < min_deg_needed.
            # This implies deg_full(worst_v) < min_deg_needed, which is impossible if min_deg_needed <= min degree of full graph.
            # For polarity graph q=7, min degree is 8. If min_deg_needed=7, this shouldn't happen unless we are stuck.
            # If we are stuck, break.
            break
            
        # Pick a random u from neighbors_of_worst_in_D
        u_list = list(neighbors_of_worst_in_D)
        rng.shuffle(u_list)
        
        # Find candidates for w in S \ {worst_v} such that w is not a neighbor of worst_v
        non_neighbors_in_S = (S - {worst_v}) - adj_full[worst_v]
        if not non_neighbors_in_S:
            # All other vertices in S are neighbors of worst_v.
            # Swapping any w will reduce worst_v's degree.
            # This is bad. Try a different strategy or break.
            break
            
        w_list = list(non_neighbors_in_S)
        rng.shuffle(w_list)
        
        found_swap = False
        for u in u_list:
            for w in w_list:
                # Try swap
                S_new = (S - {w}) | {u}
                # Check if this swap improves the situation or is valid
                # Simple check: does worst_v now have higher degree?
                # deg_S_new(worst_v) = deg_S(worst_v) + 1 (since u is neighbor) 
                # (w is not neighbor, so removing w doesn't affect worst_v's degree)
                # So deg increases by 1.
                # But we must ensure we don't break other vertices.
                # Let's just do the swap and see if is_valid improves or stays same.
                # For speed, just do the swap.
                S = S_new
                D = (D - {u}) | {w}
                found_swap = True
                break
            if found_swap:
                break
                
        if not found_swap:
            break
            
    return None

def construct():
    # Target n = 46.
    # Use polarity graph of order 7 (n=57) as base.
    # Min degree of base is 8.
    # We need a subgraph on 46 vertices with min degree >= 7.
    
    q = 7
    n_full, edges_full = _build_polarity_graph(q)
    
    # Build adjacency list for full graph
    adj_full = [set() for _ in range(n_full)]
    for u, v in edges_full:
        adj_full[u].add(v)
        adj_full[v].add(u)
        
    # Try to find subgraph on 46 vertices with min degree >= 7
    n_target = 46
    min_deg_needed = n_target - 39  # 7
    
    rng = random.Random(42)
    
    S = _try_find_subgraph(n_full, adj_full, n_target, min_deg_needed, rng)
    
    if S:
        # Map original indices to 0..n_target-1
        s_list = sorted(list(S))
        old_to_new = {old: new for new, old in enumerate(s_list)}
        
        edges = []
        for u, v in edges_full:
            if u in old_to_new and v in old_to_new:
                edges.append([old_to_new[u], old_to_new[v]])
                
        # Verify
        adj = [set() for _ in range(n_target)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        if _has_c4(n_target, adj):
             # Should not happen since it's a subgraph of C4-free graph
             pass
             
        min_deg = _min_degree(n_target, adj)
        if min_deg >= min_deg_needed:
            return {'n': n_target, 'edges': edges}
            
    # Fallback: Try n=45 with min degree >= 6
    n_target = 45
    min_deg_needed = n_target - 39  # 6
    rng = random.Random(43)
    S = _try_find_subgraph(n_full, adj_full, n_target, min_deg_needed, rng)
    
    if S:
        s_list = sorted(list(S))
        old_to_new = {old: new for new, old in enumerate(s_list)}
        
        edges = []
        for u, v in edges_full:
            if u in old_to_new and v in old_to_new:
                edges.append([old_to_new[u], old_to_new[v]])
                
        adj = [set() for _ in range(n_target)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        min_deg = _min_degree(n_target, adj)
        if min_deg >= min_deg_needed:
            return {'n': n_target, 'edges': edges}
            
    # Fallback: Polarity graph of order 5
    n_core, core_edges = _build_polarity_graph(5)
    return {'n': n_core, 'edges': core_edges}
