import time
import sys

def construct():
    """
    Construct a triangle-free graph on 16 vertices with chromatic number at least 4,
    maximizing the number of edges.
    
    Strategy:
    1. Start with the Grötzsch graph (Mycielski graph of C5) on vertices 0..10.
       It has 20 edges, is triangle-free, and has chromatic number 4.
    2. The maximum independent set of the Grötzsch graph has size 5 (e.g., the 5 'u' vertices).
    3. Add 5 new vertices (11..15).
    4. Connect each new vertex to all 5 vertices in the maximum independent set.
    5. This adds 5 * 5 = 25 edges.
    6. Total edges = 20 + 25 = 45.
    7. The graph remains triangle-free because the new vertices are independent from each other
       and their neighbors form an independent set in the original graph.
    8. The chromatic number remains at least 4 because the Grötzsch graph is a subgraph.
    """
    # Grötzsch graph construction
    # Vertices 0-4: Cycle C5
    # Vertices 5-9: Copies u_i
    # Vertex 10: w
    
    edges = []
    
    # Cycle edges (0-4)
    for i in range(5):
        edges.append([i, (i + 1) % 5])
        
    # Edges between v_i and u_j
    # u_i is connected to v_{i-1} and v_{i+1}
    for i in range(5):
        u_i = 5 + i
        v_prev = (i - 1) % 5
        v_next = (i + 1) % 5
        edges.append([u_i, v_prev])
        edges.append([u_i, v_next])
        
    # Edges between u_i and w (10)
    for i in range(5):
        edges.append([5 + i, 10])
        
    # Grötzsch graph has 20 edges.
    # Max independent set is {5, 6, 7, 8, 9}
    max_independent_set = [5, 6, 7, 8, 9]
    
    # Add vertices 11-15
    for v in range(11, 16):
        for u in max_independent_set:
            edges.append([v, u])
            
    return edges
