<<<<<<< SEARCH
def _build_voltage_kautz(m, deadline):
    """
    Build a voltage lift of the Kautz(3,2) graph over Z_m.
    Kautz(3,2) has 6 vertices, 3-regular.
    Uses deterministic voltage assignment via primitive root.
    Returns list of edges or None if deadline exceeded.
    """
    # Kautz(3,2) graph: vertices are ordered pairs (i,j) with i != j, i,j in {0,1,2}
    # That's 6 vertices: (0,1),(0,2),(1,0),(1,2),(2,0),(2,1)
    # Edge from (i,j) to (j,k) for k != i, k != j... wait, degree should be 3.
    # Actually Kautz(q,d): vertices are words of length d over q symbols with adjacent symbols different.
    # Kautz(3,2): words (a,b) with a != b. 6 vertices.
    # Edge from (a,b) to (b,c) for c != a and c != b. But c != a and c != b means c is the third symbol.
    # So each vertex has degree 1? That's not right.
    
    # Let me reconsider. Kautz graph K(q,d):
    # Vertices: sequences (x_0, ..., x_{d-1}) with x_i != x_{i+1} for all i.
    # For d=2: (x_0, x_1) with x_0 != x_1. q*(q-1) vertices.
    # Edges: (x_0, ..., x_{d-1}) connected to (x_1, ..., x_{d-1}, x_d) where x_d != x_{d-1}.
    # So from (a,b), edges go to (b,c) for c != b. That's q-1 = 2 edges. Degree 2, not 3.
    
    # For degree 3, we need q-1=3, so q=4.
    # Kautz(4,2): 4*3=12 vertices, degree 3.
    # Or use a different base graph.
    
    # Let's use a simpler approach: take a 3-regular expander base and lift it.
    # Base: Petersen graph (10 vertices, 3-regular, girth 5).
    # Lift over Z_m: n = 10*m.
    # Moore bound 512 => m >= 52.
    
    # Actually, let me use a known 3-regular graph with small girth as base.
    # The complete bipartite K_{3,3} is 3-regular, 6 vertices, girth 4.
    # Better: use the 3-regular graph on 8 vertices (cubic) with girth 5?
    
    # Simplest: use a fixed 3-regular base with good properties.
    # Let's use the 3-regular graph formed by the 1-skeleton of a cube (8 vertices, degree 3, girth 4).
    # Girth 4 is too small.
    
    # Let me just use the Petersen graph as base (10 vertices, 3-regular, girth 5).
    # Petersen graph edges:
    # Outer pentagon: 0-1, 1-2, 2-3, 3-4, 4-0
    # Inner star: 5-7, 7-9, 9-6, 6-8, 8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    
    petersen_edges = [(0,1),(1,2),(2,3),(3,4),(4,0),
                      (5,7),(7,9),(9,6),(6,8),(8,5),
                      (0,5),(1,6),(2,7),(3,8),(4,9)]
    
    # Assign voltages to each edge using a primitive root mod m
    # Voltage of edge (u,v) is g^{hash(u,v)} mod m where g is primitive root
    # For simplicity, use voltage = (u*17 + v*31 + edge_index*7) % m
    
    if m <= 1:
        return None
    
    edges = []
    for idx, (u, v) in enumerate(petersen_edges):
        # Voltage assignment: deterministic based on edge identity
        # Use different primes for different edges to avoid small-order issues
        volt = (idx * 13 + 5) % m
        
        for i in range(m):
            u_lift = u * m + i
            v_lift = v * m + (i + volt) % m
            if u_lift < v_lift:
                edges.append((u_lift, v_lift))
            else:
                edges.append((v_lift, u_lift))
    
    return edges
=======
def _build_voltage_petersen(m):
    """
    Build a voltage lift of the Petersen graph over Z_m.
    Petersen graph: 10 vertices, 3-regular, girth 5.
    Lift over Z_m gives 10*m vertices.
    Uses deterministic voltage assignment to maximize girth.
    """
    if m <= 1:
        return None
    
    # Petersen graph edges (directed for voltage assignment)
    # We need to orient edges consistently for the voltage assignment
    petersen_edges = [(0,1),(1,2),(2,3),(3,4),(4,0),
                      (5,7),(7,9),(9,6),(6,8),(8,5),
                      (0,5),(1,6),(2,7),(3,8),(4,9)]
    
    # Voltage assignment: use different multipliers for each edge
    # to avoid small-order closed walks
    # volt_e = (a_e * i + b_e) mod m for edge e
    # We want the product of voltages around any base cycle to have large order
    
    # For edge (u,v) at position idx, voltage from u to v is:
    # v_uv = (idx * 7 + 3) mod m
    # For the reverse direction (v to u), voltage is -v_uv mod m
    
    edges = []
    for idx, (u, v) in enumerate(petersen_edges):
        volt = (idx * 7 + 3) % m
        
        for i in range(m):
            u_lift = u * m + i
            v_lift = v * m + (i + volt) % m
            if u_lift < v_lift:
                edges.append((u_lift, v_lift))
            else:
                edges.append((v_lift, u_lift))
    
    return edges
>>>>>>> REPLACE
