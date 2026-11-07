"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 12, 'minimum_degree': 3, 'girth': 4}

def construct():
    """
    Construct a graph on vertices 0..11 with:
    - Every vertex has degree at least 3
    - Every simple cycle has length at least 4 (triangle-free)
    - Minimize the number of distinct simple cycle lengths
    
    Strategy: Use K_{6,6} complete bipartite graph.
    - Bipartite graphs have no odd cycles, so girth >= 4 is guaranteed
    - Minimum degree in K_{6,6} is 6, which satisfies degree >= 3
    - The only cycle lengths in K_{6,6} are even: 4, 6, 8, 10, 12
    - We need to verify what cycle lengths actually appear
    
    Actually, let's think more carefully. In K_{6,6}:
    - 4-cycles: yes (any 2 vertices in each part form a 4-cycle)
    - 6-cycles: yes
    - 8-cycles: yes
    - 10-cycles: yes
    - 12-cycles: yes
    
    So K_{6,6} has 5 distinct cycle lengths: {4, 6, 8, 10, 12}
    
    Can we do better? The theoretical minimum is 1 (if all cycles have the same length).
    A graph where all cycles have length 4 would be ideal. But can we have min degree 3 
    with all cycles of length 4?
    
    A 4-regular graph where every cycle is length 4... that's like a disjoint union of 
    complete bipartite graphs K_{k,k} where only 4-cycles exist. But in K_{k,k} for k>=3,
    we get longer cycles too.
    
    What about a graph that's a disjoint union of K_{3,3} components? 
    K_{3,3} has 6 vertices, degree 3, and cycles of length 4 and 6.
    So two copies of K_{3,3} would give us 12 vertices, degree 3, and cycle lengths {4, 6}.
    
    That's 2 distinct cycle lengths, which is better than K_{6,6}'s 5.
    
    Can we get down to 1? We'd need all cycles to be the same length.
    If all cycles are length 4, the graph must be a "4-cycle graph" in some sense.
    But with min degree 3, it's hard to avoid longer cycles.
    
    Actually, let's try two disjoint K_{3,3} graphs.
    """
    n = 12
    edges = set()
    
    # First K_{3,3}: vertices 0,1,2 in part A, 3,4,5 in part B
    for i in range(3):
        for j in range(3):
            u = i
            v = 3 + j
            edges.add(tuple(sorted((u, v))))
    
    # Second K_{3,3}: vertices 6,7,8 in part A, 9,10,11 in part B
    for i in range(3):
        for j in range(3):
            u = 6 + i
            v = 9 + j
            edges.add(tuple(sorted((u, v))))
    
    return [list(e) for e in sorted(edges)]
# EVOLVE-BLOCK-END
