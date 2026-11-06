"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a graph with 31 vertices and 91 edges minimizing the number of 4-cycles.
    
    Strategy:
    1. Start with a polarity graph of order 31 (from finite geometry) which is known to have
       ex(31; C4) = 90 edges and exactly 0 four-cycles.
    2. Add one extra edge to reach 91 edges.
    3. Choose the extra edge that minimizes the number of new 4-cycles created.
    
    A polarity graph on q^2+q+1 vertices (q odd) is a C4-free graph with q^2+q+1 vertices.
    For q=5, we have 31 vertices. The standard construction uses the projective plane of order 5.
    
    The polarity graph construction:
    - Vertices are points of the projective plane PG(2, q) plus the line at infinity? 
      Actually, for a polarity graph on q^2+q+1 vertices:
      Vertices correspond to points of PG(2,q). Two vertices are adjacent if the point corresponding
      to one is on the polar line of the other.
    
    Alternatively, a simpler algebraic construction for polarity graphs:
    For q odd, vertices are elements of F_q x F_q plus one extra vertex (infinity).
    Actually, let me use the standard construction.
    
    Standard polarity graph on q^2+q+1 vertices for odd q:
    Let V = {0, 1, ..., q-1} x {0, 1, ..., q-1} union {∞}
    That's q^2 + 1 vertices, not q^2+q+1.
    
    Let me reconsider. The polarity graph from a projective plane of order q has q^2+q+1 vertices.
    
    For q=5, we need 31 vertices. Let me use the finite field construction.
    
    Actually, a well-known C4-free graph with q^2+q+1 vertices for prime power q is the incidence graph 
    of points and lines? No, that's bipartite with 2(q^2+q+1) vertices.
    
    Let me use the standard polarity graph construction:
    - Take the projective plane of order q.
    - A polarity is a bijection from points to lines such that point P is incident to line f(Q) iff Q is incident to line f(P).
    - The polarity graph has vertices = points of the plane, and two vertices P, Q are adjacent iff P is incident to f(Q) (equivalently Q is incident to f(P)).
    
    For q=5, I need to construct a projective plane of order 5 and a polarity.
    
    This is complex. Let me use a simpler approach:
    
    Known result: The Regev-Rödl graph or similar constructions.
    
    Actually, let me just build a random C4-free graph with 90 edges and then add one edge optimally.
    
    Better approach: Use the known construction of a C4-free graph with 31 vertices and 90 edges.
    
    The polarity graph from the projective plane of order 5 has 31 vertices and each vertex has degree 6 (since q+1=6).
    Total edges = 31*6/2 = 93. Wait, that's 93 edges, not 90.
    
    Let me recalculate. In a polarity graph from a projective plane of order q:
    - Each point P is adjacent to all points Q such that Q is on the polar line of P.
    - The polar line of P contains q+1 points, including P itself (since the polarity is a correlation of type (1,1), the point P is on its own polar line).
    - So P is adjacent to q points (excluding itself) on its polar line, plus... wait.
    
    Actually, in a polarity graph, vertex P is connected to vertex Q if P is on the polar line of Q. Since the polarity is symmetric, this is equivalent to Q being on the polar line of P.
    
    The polar line of P contains q+1 points. One of them is P itself. So P is adjacent to q other points.
    But we also need to check if P is on its own polar line - yes, for a polarity of a projective plane, each point lies on its own polar line.
    
    So degree of each vertex = q (since it's adjacent to the other q points on its polar line).
    Total edges = 31 * 5 / 2 = 77.5. That's not an integer, so something is wrong.
    
    Wait, q=5, so degree should be 5? 31*5/2 = 77.5. Not integer. So my understanding is flawed.
    
    Let me look this up differently. For a polarity graph of order q (odd), the number of vertices is q^2+q+1 and the number of edges is q(q^2+q+1)/2 if q is odd... no.
    
    Actually, I recall that the polarity graph has q^2+q+1 vertices and each vertex has degree q+1 or q depending on whether the point is absolute or not.
    
    This is getting complicated. Let me use a different strategy:
    
    1. Generate a random graph with 31 vertices and 90 edges.
    2. Use local search to minimize 4-cycles.
    3. Then add one edge that creates the fewest new 4-cycles.
    
    But this might be slow. Let me think of a better approach.
    
    Actually, let me just use a greedy/randomized approach to build a C4-free graph with 90 edges, then add the best 91st edge.
    
    For the C4-free part, I'll use a greedy algorithm:
    - Start with an empty graph.
    - Add edges in random order, but only add an edge if it doesn't create a C4.
    - Continue until we have 90 edges or can't add more.
    
    Then for the 91st edge, try all possible non-edges and pick the one that creates the fewest C4s.
    
    Let me implement this.
    """
    n = 31
    target_edges = 91
    
    # First, build a C4-free graph with 90 edges using greedy randomized approach
    # We'll try multiple random orders and keep the best one
    
    import random
    from itertools import combinations
    
    def count_c4s(edges, n):
        """Count the number of 4-cycles in the graph."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        c4_count = 0
        # For each pair of vertices, count common neighbors
        common_neighbors = {}
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if len(common) >= 2:
                    c4_count += len(common) * (len(common) - 1) // 2
        return c4_count
    
    def creates_c4(u, v, adj, n):
        """Check if adding edge (u, v) creates a C4."""
        common = adj[u] & adj[v]
        return len(common) >= 2
    
    # Greedy construction of C4-free graph
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_edges = None
    best_c4 = float('inf')
    
    for trial in range(50):
        random.seed(trial)
        random.shuffle(all_pairs)
        
        adj = [set() for _ in range(n)]
        edges = []
        
        for u, v in all_pairs:
            if len(edges) >= 90:
                break
            if not creates_c4(u, v, adj, n):
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
        
        if len(edges) < 90:
            continue
        
        c4_count = count_c4s(edges, n)
        if c4_count < best_c4:
            best_c4 = c4_count
            best_edges = edges[:]
    
    if best_edges is None or len(best_edges) < 90:
        # Fallback: use lexicographic first 91 edges
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]
    
    # Now add the 91st edge that minimizes new C4s
    adj = [set() for _ in range(n)]
    for u, v in best_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    existing_edges = set()
    for u, v in best_edges:
        existing_edges.add((min(u, v), max(u, v)))
    
    best_new_edge = None
    best_new_c4 = float('inf')
    
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in existing_edges:
                continue
            # Count how many C4s would be created by adding (u, v)
            common = adj[u] & adj[v]
            new_c4s = len(common) * (len(common) - 1) // 2
            if new_c4s < best_new_c4:
                best_new_c4 = new_c4s
                best_new_edge = (u, v)
    
    if best_new_edge is not None:
        best_edges.append(best_new_edge)
    
    # Convert to list of lists
    result = [[u, v] for u, v in best_edges]
    
    # Ensure exactly 91 distinct edges
    if len(result) > 91:
        result = result[:91]
    elif len(result) < 91:
        # Add more edges if needed (shouldn't happen)
        existing = set(tuple(sorted(e)) for e in result)
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in existing and len(result) < 91:
                    result.append([i, j])
                    existing.add((i, j))
    
    return result

# EVOLVE-BLOCK-END
