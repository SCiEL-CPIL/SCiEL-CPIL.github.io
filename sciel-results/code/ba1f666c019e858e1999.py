"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

# EVOLVE-BLOCK-START
def _build_polarity_graph(q):
    """Build the polarity graph of PG(2,q) for prime q ≡ 3 mod 4."""
    # Points of PG(2,q) are equivalence classes of (x,y,z) not all zero
    # For q ≡ 3 mod 4, -1 is not a square, so the standard Hermitian-like polarity works
    # Use the polarity: (x,y,z) is orthogonal to (x',y',z') iff xx' + yy' + zz' = 0 in F_q
    # But for q ≡ 3 mod 4, we need a different form to ensure it's a polarity
    
    # Actually, for prime q, use the form: xx' + yy' + zz' = 0
    # This is a polarity when q is odd. For q ≡ 3 mod 4, -1 is nonsquare,
    # so no point is on its polar line (no absolute points) - wait, that's not right.
    
    # Let me use a simpler approach: the Paley-type construction
    # Or directly construct from finite field
    
    # For q prime, the graph where vertices are F_q × F_q ∪ {∞} 
    # Actually, let's use the standard construction:
    
    # Vertices: all points of PG(2,q), represented as (x,y,z) with z=1 or z=0
    # Polarity: (x,y,z) ⊥ (x',y',z') iff xx' + yy' + zz' = 0
    
    # Enumerate all points
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    
    n = len(points)  # q^2 + q + 1
    
    # Build adjacency: i ~ j if dot product is 0 (and i != j)
    edges = []
    adj = set()
    for i in range(n):
        for j in range(i+1, n):
            p, r = points[i], points[j]
            dot = (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
            if dot == 0:
                adj.add((i,j))
                edges.append([i,j])
    
    return n, edges

def _check_c4(n, edges):
    """Check if graph has a C4."""
    adj = [set() for _ in range(n)]
    for u,v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for u in range(n):
        for v in list(adj[u]):
            if v <= u:
                continue
            common = adj[u] & adj[v]
            if len(common) >= 2:
                return False
    return True

def _check_independent_set(n, edges, size):
    """Check if graph has an independent set of given size."""
    adj = [set() for _ in range(n)]
    for u,v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Backtracking search for independent set of given size
    vertices = list(range(n))
    
    def backtrack(remaining, current):
        if len(current) == size:
            return True
        if not remaining:
            return False
        # Pruning: if not enough vertices left
        if len(remaining) < size - len(current):
            return False
        
        # Choose vertex with highest degree in remaining subgraph
        best = max(remaining, key=lambda v: len(adj[v] & set(remaining)))
        # Try including best
        new_remaining = [v for v in remaining if v != best and best not in adj[v]]
        if backtrack(new_remaining, current + [best]):
            return True
        # Try excluding best
        new_remaining = [v for v in remaining if v != best]
        if backtrack(new_remaining, current):
            return True
        return False
    
    return backtrack(vertices, [])

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Try polarity graphs for various primes q ≡ 3 mod 4
    # q=3: 13 vertices, q=7: 57 vertices, q=11: 133 vertices (too many)
    
    best_result = None
    
    # Start with known good: ten triangles (n=30)
    edges30 = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        edges30 += [[a,b],[a,c],[b,c]]
    best_result = {'n': 30, 'edges': edges30}
    
    # Try q=7 polarity graph (57 vertices)
    if time.time() < deadline - 5:
        n, edges = _build_polarity_graph(7)
        # Check C4-free
        if _check_c4(n, edges):
            # Check independence number
            # For polarity graph of PG(2,7), need to verify α ≤ 10
            # This might be expensive, so do a quick check
            if not _check_independent_set(n, edges, 11):
                best_result = {'n': n, 'edges': edges}
    
    if time.time() >= deadline - 5:
        return best_result
    
    # Try to improve: start from smaller C4-free graphs and add vertices
    # Or try q=3 (13 vertices) combined with other structures
    
    # Try a different approach: use known strongly regular graphs
    # The Hoffman-Singleton graph: 50 vertices, C4-free, α=7
    # But we need to verify
    
    # Let's try to build a graph on 39 vertices using a known construction
    # The best known R(C4,K11) > 39 construction
    
    # Alternative: try to find a C4-free graph with α ≤ 10 on up to 43 vertices
    # by starting from polarity graph and removing vertices if needed
    
    if time.time() < deadline - 10:
        # Try to find a subset of the q=7 polarity graph with α ≤ 10
        # Or build a custom graph
        
        # Let's try: take the 13-vertex polarity graph (q=3) and combine
        # Actually, let's try a direct construction
        
        # Use the fact that we can take disjoint union of C4-free graphs
        # and the independence number adds
        # Ten triangles: α=10, n=30
        # Can we add more vertices while keeping C4-free and α ≤ 10?
        
        # Try adding vertices connected to break up independent sets
        # Start with 10 triangles and try to add vertices
        
        base_edges = list(edges30)
        base_n = 30
        current_n = base_n
        current_edges = base_edges
        
        # Try adding vertices one by one
        for new_v in range(43):
            if time.time() >= deadline - 5:
                break
            if current_n >= 43:
                break
            
            # Try to add vertex current_n connected to some existing vertices
            # Must maintain C4-free and α ≤ 10
            adj = [set() for _ in range(current_n)]
            for u,v in current_edges:
                adj[u].add(v)
                adj[v].add(u)
            
            # Try different connection patterns
            found = False
            for mask in range(1 << min(current_n, 10)):  # try connecting to first few
                if time.time() >= deadline - 5:
                    break
                neighbors = set()
                for i in range(min(current_n, 10)):
                    if mask & (1 << i):
                        neighbors.add(i)
                
                # Check C4-free: new vertex can't create C4
                # C4 with new vertex: need two neighbors of new vertex that share a common neighbor
                c4_ok = True
                nb_list = list(neighbors)
                for a_i in range(len(nb_list)):
                    for b_i in range(a_i+1, len(nb_list)):
                        a, b = nb_list[a_i], nb_list[b_i]
                        common = adj[a] & adj[b]
                        if len(common) >= 1:  # a and b share a neighbor, and both connect to new_v
                            # That's a C4: new_v - a - common - b - new_v
                            c4_ok = False
                            break
                    if not c4_ok:
                        break
                
                if not c4_ok:
                    continue
                
                # Check α ≤ 10
                # New independent set: either doesn't include new_v (old α) or includes new_v
                # If includes new_v: can include independent set from non-neighbors of new_v
                non_nb = set(range(current_n)) - neighbors
                # Max independent set in non_nb induced subgraph
                # Quick upper bound: |non_nb|
                if len(non_nb) >= 10:
                    # Need to check if there's an independent set of size 10 in non_nb
                    # This is expensive, so skip for now
                    continue
                
                # If we get here, it might work
                new_edges = current_edges + [[current_n, x] for x in neighbors]
                current_n += 1
                current_edges = new_edges
                found = True
                break
            
            if not found:
                # Try with no edges (isolated vertex) - but that increases α
                # Only if α was < 10
                pass
    
    if current_n > 30:
        best_result = {'n': current_n, 'edges': current_edges}
    
    return best_result
# EVOLVE-BLOCK-END
