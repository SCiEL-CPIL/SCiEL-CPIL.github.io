"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # The task requires:
    # 1. Undirected simple edges on vertices 0..11.
    # 2. Every vertex degree >= 3.
    # 3. Girth >= 4 (triangle-free).
    # 4. Minimize number of distinct simple cycle lengths.
    
    # Strategy:
    # Construct two disjoint copies of the complete bipartite graph K_{3,3}.
    # K_{3,3} has 6 vertices, is 3-regular, and is bipartite (girth 4).
    # The simple cycles in K_{3,3} have lengths 4 and 6 only.
    # Therefore, the union of two K_{3,3}s will have distinct cycle lengths {4, 6}.
    # Raw value = 2.
    # This is a robust construction that satisfies all constraints.
    
    edges = []
    
    # Define a helper to generate edges for K_{m, n} given vertex ranges
    def add_k_mn(part_a, part_b):
        for u in part_a:
            for v in part_b:
                if u < v:
                    edges.append([u, v])
                else:
                    edges.append([v, u])

    # Component 1: Vertices 0..5
    part_a1 = [0, 1, 2]
    part_b1 = [3, 4, 5]
    add_k_mn(part_a1, part_b1)
    
    # Component 2: Vertices 6..11
    part_a2 = [6, 7, 8]
    part_b2 = [9, 10, 11]
    add_k_mn(part_a2, part_b2)

    # Remove duplicates and sort
    unique_edges = set()
    for e in edges:
        if e[0] > e[1]:
            unique_edges.add((e[1], e[0]))
        else:
            unique_edges.add((e[0], e[1]))
            
    # Convert back to list of lists of ints
    result = [[int(u), int(v)] for u, v in sorted(unique_edges)]

    return result

# EVOLVE-BLOCK-END
