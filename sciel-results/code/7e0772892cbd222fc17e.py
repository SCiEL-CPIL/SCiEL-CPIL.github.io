"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def sign_edges(graph, budget):
    import numpy as np
    
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    
    # We need to assign signs to edges.
    # The graph is a prism: two m-cycles (top and bottom) connected by a matching.
    # Vertices 0..m-1 are top, m..2m-1 are bottom.
    # Edges are likely ordered as: top cycle, bottom cycle, matching? Or interleaved?
    # The problem statement says: "graph = {'n': 2m, 'edges': ordered list of (u, v)}: two m-cycles... joined by the matching"
    # It doesn't explicitly state the order of edges in the list. 
    # However, standard prism graph representations often list cycle edges first, then matching, or in some other order.
    # We must inspect the edge endpoints to determine their type.
    
    signs = np.ones(n_edges, dtype=int)
    
    for i, (u, v) in enumerate(edges):
        # Determine if this edge is a matching edge or a cycle edge
        # Matching edges connect i and i+m. 
        # Cycle edges connect i and i+1 (mod m) within the same half.
        
        if abs(u - v) == m:
            # This is a matching edge between u and v (where v = u+m or u = v+m)
            # Let's assume u is the top vertex (0..m-1) and v is bottom (m..2m-1) or vice versa.
            # To create a "twisted" lift that might reduce spectral radius, 
            # we can alternate signs on the matching edges.
            # Let's try: sign = +1 if u is even, -1 if u is odd.
            # Or maybe all -1? 
            # Let's think about the spectrum.
            # For a prism graph, if all signs are +1, the lift is two disjoint copies of the prism (if connected? No, +1 on matching connects corresponding vertices, so it's two disjoint prisms? No, the base graph is connected. The lift with all +1 signs is two copies of the base graph, so it's disconnected. The task requires the lift to be connected.
            # Wait, "The lift must be connected... or the whole result is invalid."
            # If all signs are +1, the lift consists of two disjoint copies of the base graph. Thus, disconnected. Invalid.
            # If all signs are -1, the lift is a single connected graph (the double cover with a twist).
            # So we must have at least one -1 to ensure connectivity?
            # Actually, the lift is connected if and only if the product of signs around every cycle is -1? No.
            # The lift of a connected graph is connected if and only if the sign assignment is not "trivial" in the sense that the homomorphism to Z2 is surjective? 
            # Let's recall: The 2-lift defined by signs is connected if and only if there is no non-trivial partition of vertices into two sets such that all edges within a set have sign +1 and all edges between sets have sign -1? No, that's for bipartite.
            # Simpler: The lift is connected iff the sign assignment corresponds to a non-trivial element of H^1(G, Z2). For a connected graph, the lift is connected iff the product of signs along some cycle is -1.
            # The prism graph contains cycles. For example, the "ladder" cycle of length 2m? No, the matching edges and cycle edges form cycles of length 4 (squares) if m>=4? 
            # Let's check the cycle (0, 1, m+1, m, 0). Edges: (0,1), (1,m+1), (m+1,m), (m,0).
            # If we set signs such that the product around this 4-cycle is -1, the lift is connected.
            
            # Strategy: Start with all -1 signs. This ensures connectivity (product around any cycle is (-1)^L, if L is even, product is 1? Wait.
            # If all signs are -1, product around a cycle of length L is (-1)^L.
            # Prism graph cycles:
            # - Top cycle: length m. Product (-1)^m.
            # - Bottom cycle: length m. Product (-1)^m.
            # - Square cycles (if we go up, across, down, across back): length 4. Product (-1)^4 = 1.
            # If m is even, top/bottom cycle product is 1. If m is odd, product is -1.
            # The lift is connected iff there exists a cycle with product -1.
            # If m is odd, all -1 signs yield a connected lift.
            # If m is even, all -1 signs yield a disconnected lift (product 1 on all cycles? No, just the ones we checked. Are there other cycles?
            # In a prism with even m, it is bipartite. All cycles are even length. Product of -1s on even length is 1. So all cycles have product 1. Lift is disconnected.
            # So for even m (4, 8? no m=4, 5, 7, 9. m=4 is even), all -1 is invalid.
            # We need at least one cycle with product -1.
            # For m=4, we can flip one sign to make the product of the 4-cycle be -1.
            
            # Let's try a pattern:
            # For matching edge (i, i+m), set sign = -1 if i is even, +1 if i is odd?
            # Let's test connectivity for this pattern.
            # Cycle (0, 1, m+1, m, 0):
            # Edges: (0,1) [top], (1, m+1) [matching], (m+1, m) [bottom], (m, 0) [matching].
            # Signs:
            # Top edge (0,1): Let's assume cycle edges are +1 for now? Or -1?
            # Let's fix cycle edges to +1 first.
            # Matching (1, m+1): i=1 (odd) -> +1.
            # Matching (m, 0): i=0 (even) -> -1.
            # Product: (+1)*(+1)*(+1)*(-1) = -1. Connected!
            
            # So, strategy:
            # 1. All cycle edges (top and bottom) get sign +1.
            # 2. Matching edges get sign -1 if the top vertex index is even, +1 if odd.
            
            # Let's verify this for m=4, 5, 7, 9.
            # This pattern ensures at least one 4-cycle has product -1, so the lift is connected.
            
            # Which vertex is "top"? u or v?
            # If u < m, u is top. If v < m, v is top.
            top = u if u < m else v
            
            if top % 2 == 0:
                signs[i] = -1
            else:
                signs[i] = +1
        else:
            # Cycle edge
            signs[i] = +1
            
    return signs
# EVOLVE-BLOCK-END
