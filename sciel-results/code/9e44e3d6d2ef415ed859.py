"""Construct a C5-free graph with no independent set of size 9, maximizing n."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def construct():
    """
    Returns a simple graph on n vertices (0..n-1) that is:
    - C5-free (no 5-cycle as a subgraph)
    - has no independent set of size 9
    
    Base construction: 8 disjoint copies of K4.
    n = 32.
    alpha(G) = 8 (one vertex from each K4).
    K4 is C5-free, so the disjoint union is C5-free.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n = 32
    edges = []
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    best_n = n
    best_edges = edges
    
    # Try to find a valid graph with n=33 using a heuristic search
    # Strategy: Start from the n=32 graph and try to add one vertex with edges
    # to existing vertices, then try to remove some edges to maintain C5-freeness
    # while keeping alpha < 9.
    
    # Alternative strategy: Direct construction for n=33
    # We need a C5-free graph on 33 vertices with alpha <= 8.
    
    # Let's try a randomized approach to find an edge set for n=33
    # We'll use a simulated annealing-like approach on the edge set.
    
    # But first, let's try a specific known structure or a smarter heuristic.
    # The complement of the graph H must be K9-free and C5-free.
    # So G must be C5-free and its complement must be K9-free.
    # This means G must have no independent set of size 9.
    
    # Let's try to construct G directly.
    # A known extremal graph for R(C5, K9) might be related to blow-ups of C5?
    # No, C5-free means no subgraph C5.
    # What about a blow-up of a C5-free graph?
    # The base graph must be C5-free.
    # K4 is C5-free. The disjoint union of K4s is C5-free.
    # What if we add edges between the K4s?
    # If we add an edge between two K4s, does it create a C5?
    # Not necessarily.
    # But adding edges reduces the independence number? No, adding edges makes it harder to find independent sets.
    # So adding edges helps with the alpha constraint but might violate C5-freeness.
    
    # Let's try adding random edges between the K4s and check for C5.
    # If a C5 is found, remove one of the new edges.
    
    n_try = 33
    # Start with 8 K4s (32 vertices) and add 1 isolated vertex? No, that increases alpha to 9.
    # So we must connect the new vertex to existing vertices.
    # Let's start with the n=32 graph and add vertex 32.
    # Connect vertex 32 to some vertices in the K4s.
    
    # We need alpha(G) <= 8.
    # If we connect v32 to at least one vertex in each K4, then any independent set can contain at most one vertex from each K4 AND cannot contain v32 if it contains its neighbors.
    # Actually, if v32 is connected to a vertex u in K4_i, then an independent set cannot contain both v32 and u.
    # To ensure alpha <= 8, we need to "cover" the independent sets of the K4s.
    
    # Let's try a randomized search for n=33.
    # We will maintain an edge set and try to add/remove edges.
    # Objective: Maximize number of vertices (fixed at 33) subject to C5-free and alpha < 9.
    
    # Since this is a hard search problem, and we have limited time,
    # let's try a few specific constructions.
    
    # Construction 2: 7 K4s and one K5?
    # K5 is C5-free? No, K5 contains C5 as a subgraph (actually as an induced subgraph? No, K5 has all edges, so it contains C5).
    # Wait, a complete graph K_n contains C_k as a subgraph for any k <= n.
    # So K5 contains C5. Thus we cannot use K5.
    # The largest C5-free complete graph is K4.
    
    # So all components must be C5-free.
    # What about C4? C4 is C5-free.
    # What about K_{2,3}? C5-free.
    
    # Let's stick to the 8xK4 construction for now, as it's known to be valid.
    # The prompt says "best known has 32 vertices".
    # So finding n=33 might be very hard or impossible within the time limit.
    
    # I will return the n=32 solution.
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
