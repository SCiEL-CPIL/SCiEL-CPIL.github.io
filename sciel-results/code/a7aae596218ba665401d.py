"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Target: Minimize number of distinct simple cycle lengths.
    # Seed value (3 distinct lengths: 4, 6, 12) -> Score 1.0
    # Goal: 2 distinct lengths (4, 6) -> Score 1.5
    # Construction: Two disjoint copies of K_{3,3}.
    # K_{3,3} is bipartite (triangle-free), min degree 3.
    # Simple cycles in K_{3,3} have lengths 4 and 6 only.
    # Vertices 0-5 form first K_{3,3}, 6-11 form second.
    
    edges = set()
    
    # First K_{3,3} on vertices 0,1,2 (left) and 3,4,5 (right)
    left1 = [0, 1, 2]
    right1 = [3, 4, 5]
    for u in left1:
        for v in right1:
            edges.add(tuple(sorted((u, v))))
            
    # Second K_{3,3} on vertices 6,7,8 (left) and 9,10,11 (right)
    left2 = [6, 7, 8]
    right2 = [9, 10, 11]
    for u in left2:
        for v in right2:
            edges.add(tuple(sorted((u, v))))
            
    # Convert to list of lists of ints
    result = [list(e) for e in sorted(edges)]
    
    return result
# EVOLVE-BLOCK-END
