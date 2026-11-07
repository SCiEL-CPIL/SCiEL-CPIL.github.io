"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # Strategy: Construct two disjoint K_{3,3} components.
    # K_{3,3} is 3-regular, bipartite (girth 4), and has cycles of length 4 and 6 only.
    # Two disjoint copies ensure the minimum degree is 3 for all 12 vertices.
    # The set of distinct cycle lengths in the union is {4, 6}.
    # Raw value = 2.
    # This is the theoretical minimum for min-degree-3 graphs with girth >= 4.
    
    edge_list = []
    
    # Component 1: K_{3,3} on vertices 0..5
    # Partition A: {0, 1, 2}
    # Partition B: {3, 4, 5}
    comp1_a = [0, 1, 2]
    comp1_b = [3, 4, 5]
    for node_a in comp1_a:
        for node_b in comp1_b:
            edge_list.append([int(node_a), int(node_b)])
            
    # Component 2: K_{3,3} on vertices 6..11
    # Partition A: {6, 7, 8}
    # Partition B: {9, 10, 11}
    comp2_a = [6, 7, 8]
    comp2_b = [9, 10, 11]
    for node_a in comp2_a:
        for node_b in comp2_b:
            edge_list.append([int(node_a), int(node_b)])
            
    # Sort edges for consistency
    edge_list.sort()
    
    # Final type check and conversion
    final_edges = []
    for e in edge_list:
        final_edges.append([e[0], e[1]])
        
    # Time guard
    if time.time() - start_time > deadline:
        pass # Return best known, which is this static graph
        
    return final_edges

# EVOLVE-BLOCK-END
