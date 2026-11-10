import sys
import time

def construct():
    n = 16
    # The optimal graph is the complete bipartite graph K_{n/2, n/2}.
    # For n=16, this is K_{8,8}.
    # Properties:
    # 1. Diameter is exactly 2.
    # 2. It is triangle-free.
    # 3. Removing any edge (u, v) increases the distance between u and v to 3,
    #    thus increasing the graph diameter to 3.
    # 4. By Mantel's Theorem, this has the maximum number of edges (64) 
    #    for a triangle-free graph on 16 vertices.
    #    Any graph with more edges contains a triangle, which would violate 
    #    the condition that removing an edge must increase the diameter 
    #    (edges in triangles have a common neighbor, so distance stays 2).
    
    # Partition vertices into two sets of size 8
    set_a = set(range(8))
    set_b = set(range(8, 16))
    
    edges = []
    # Generate all edges between set_a and set_b
    for u in set_a:
        for v in set_b:
            edges.append([u, v])
            
    return edges
