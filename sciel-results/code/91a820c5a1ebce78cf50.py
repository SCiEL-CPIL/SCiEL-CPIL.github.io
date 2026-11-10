import sys

def construct():
    # Construct the Groetzsch graph on vertices 0..10
    # Vertices 0-4 form a 5-cycle
    edges = []
    for i in range(5):
        edges.append([i, (i + 1) % 5])
    
    # Vertices 5-9 are copies of 0-4
    # Vertex 10 is the root
    for i in range(5):
        u = i
        v = (i + 1) % 5
        # Connect u' to v and v' to u
        edges.append([5 + u, v])
        edges.append([5 + v, u])
        
    # Connect root 10 to all 5-9
    for i in range(5):
        edges.append([10, 5 + i])
        
    # Current edge count: 5 (cycle) + 10 (cross) + 5 (root) = 20
    # Add 5 more vertices: 11, 12, 13, 14, 15
    # Connect each to an independent set of the Groetzsch graph
    # An independent set of size 4 in Groetzsch: {0, 2, 5, 7}
    # Let's verify {0, 2, 5, 7} is independent.
    # 0 not connected to 2 (dist 2 in C5).
    # 0 not connected to 5 (5 is copy of 0, connected to neighbors 4,1).
    # 0 not connected to 7 (7 is copy of 2, connected to neighbors 1,3).
    # 2 not connected to 5.
    # 2 not connected to 7.
    # 5 not connected to 7.
    # Yes, {0, 2, 5, 7} is an independent set.
    
    independent_set = [0, 2, 5, 7]
    
    for new_v in range(11, 16):
        for u in independent_set:
            edges.append([new_v, u])
            
    return edges
