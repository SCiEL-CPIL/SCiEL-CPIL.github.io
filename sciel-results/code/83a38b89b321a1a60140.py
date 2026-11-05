import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5 second margin for safety
    
    # Strategy:
    # 1. Start with known strong candidates for triangle-free graphs with small independence number.
    # 2. The complement of the Clebsch graph (n=16, alpha=5) is a good seed, but we need n ~ 39.
    # 3. Known fact: R(3,10) > 39. The graph is often related to the icosahedron or specific SRGs.
    # 4. A strong candidate for n=39 is the complement of the 39-vertex graph derived from 
    #    the icosahedron's line graph or similar structures.
    #    Actually, the known 39-vertex graph for R(3,10) is the complement of the 
    #    "Clebsch graph blow-up" or similar? No, the standard 39-vertex example is 
    #    the graph formed by the vertices of the icosahedron and its edges?
    #    
    #    Let's use a known construction: 
    #    The graph G(39) is the complement of the 39-vertex graph where 
    #    each vertex is a 2-subset of {1..5} plus something? 
    #    
    #    A more robust approach: Use the "Hamming Graph" or "Johnson Graph" logic.
    #    However, the most famous 39-vertex triangle-free graph with alpha < 10 is 
    #    the complement of the 39-vertex strongly regular graph?
    #    
    #    Actually, let's look at the structure of the 39-vertex graph.
    #    It is often constructed as follows:
    #    Take the icosahedron (12 vertices, 30 edges).
    #    This doesn't directly give 39.
    #    
    #    Let's try a different known candidate: The complement of the Clebsch graph has n=16, alpha=5.
    #    If we take the disjoint union of two Clebsch complements, n=32, alpha=10. This is not good (alpha=10).
    #    We need alpha < 10.
    #    
    #    The known 39-vertex graph is the "Clebsch graph" extended?
    #    Actually, the graph is the complement of the 39-vertex graph known as the 
    #    "39-vertex Ramsey graph".
    #    
    #    Let's construct a specific candidate:
    #    The graph is the line graph of K_{5,5} minus a perfect matching?
    #    Line graph of K_{5,5} has 25 vertices.
    #    
    #    Let's try the graph derived from the icosahedron:
    #    Vertices: 12 (icosahedron vertices) + 30 (icosahedron edges) = 42? No.
    #    
    #    There is a known 39-vertex graph. It is the complement of the 39-vertex 
    #    strongly regular graph with parameters (39, 14, 5, 6)? No, that's the complement of the 
    #    "Clebsch" related?
    #    
    #    Let's use a simpler, verifiable approach:
    #    Generate the complement of the 39-vertex graph from the icosahedron's 
    #    "edge-vertex" incidence?
    #    
    #    Actually, the most reliable 39-vertex triangle-free graph with alpha < 10 
    #    is the one constructed from the 5-dimensional hypercube? No.
    #    
    #    Let's use the following known construction for n=39:
    #    The graph is the complement of the 39-vertex graph where 
    #    vertices are the 39 edges of the icosahedron? No, icosahedron has 30 edges.
    #    
    #    Wait, the 39-vertex graph is often cited as the complement of the 
    #    "39-vertex graph" which is the line graph of K_{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}? No.
    #    
    #    Let's try a local search on a random 39-vertex graph, but initialized with 
    #    a structure that is likely to be triangle-free.
    #    
    #    Better: Use the "Blow-up" of the 5-cycle C5.
    #    C5 is triangle-free, alpha(C5)=2.
    #    Blow up each vertex by k vertices (independent set of size k).
    #    n = 5k.
    #    If k=8, n=40.
    #    Alpha of blow-up = 2 * max(blown up vertex size) ? No.
    #    In a blow-up of C5, an independent set can take all vertices from two non-adjacent bags.
    #    If bags have sizes k1..k5, alpha = max_{i != j, i not adj j} (ki + kj).
    #    To minimize alpha, make bags equal. k=8.
    #    n=40. Bags of size 8.
    #    Alpha = 8+8=16. Too large.
    #    
    #    We need a denser graph.
    #    The known 39-vertex graph is NOT a blow-up of C5.
    #    
    #    Let's use the specific 39-vertex graph from the literature:
    #    It is the complement of the 39-vertex graph formed by the 
    #    vertices of the icosahedron (12) and the edges of the icosahedron (30)?
    #    Total 42.
    #    
    #    Actually, the 39-vertex graph is the complement of the graph where 
    #    vertices are the 39 2-subsets of {1..5} with some property?
    #    Number of 2-subsets of 5 is 10.
    #    
    #    Let's try a different known graph: The "Hoffman-Singleton" graph is n=50.
    #    The "Clebsch" graph is n=16.
    #    
    #    Let's use a random 39-vertex graph and use a powerful local search 
    #    to find a triangle-free subgraph with small alpha.
    #    However, finding a triangle-free graph with alpha < 10 is hard.
    #    
    #    Alternative: Use the known 39-vertex graph.
    #    Reference: "A 39-vertex triangle-free graph with independence number 9".
    #    Construction:
    #    Let V = {0, ..., 38}.
    #    This graph is the complement of the 39-vertex strongly regular graph 
    #    with parameters (39, 14, 5, 6)?
    #    Let's check parameters.
    #    If G is SRG(39, 14, 5, 6), then complement G' is SRG(39, 24, 13, 14).
    #    Triangle-free means G' has no triangles.
    #    Does SRG(39, 24, 13, 14) have triangles?
    #    If u, v, w form a triangle in G', they are all connected in G'.
    #    In G, they are independent.
    #    
    #    Actually, the 39-vertex graph for R(3,10) is the complement of the 
    #    39-vertex graph with parameters (39, 14, 5, 6).
    #    Let's verify if such a graph exists.
    #    Yes, it is the complement of the "Clebsch graph" extended? No.
    #    It is the "39-vertex graph" often associated with the icosahedron.
    #    
    #    Let's construct the 39-vertex graph explicitly.
    #    The vertices can be labeled by pairs (i, j) where i in {0..4}, j in {0..7}?
    #    No, 5*8=40.
    #    
    #    Let's use the 40-vertex graph from the 5x8 grid?
    #    If we can find a valid 40-vertex graph, we win.
    #    But R(3,10) <= 41 implies max n is 40.
    #    Is there a 40-vertex triangle-free graph with alpha < 10?
    #    If yes, R(3,10) > 40.
    #    The survey says "39 as the largest known order". This implies no 40-vertex graph is known.
    #    So we should aim for 39.
    #    
    #    Construction of the 39-vertex graph:
    #    It is the complement of the graph where vertices are the 39 edges of the icosahedron?
    #    No, icosahedron has 30 edges.
    #    
    #    Let's use the following explicit construction for n=39:
    #    The graph is the complement of the 39-vertex graph formed by the 
    #    12 vertices of the icosahedron and the 27 edges of the icosahedron 
    #    that are not in a certain Hamiltonian cycle?
    #    
    #    Actually, let's use a simpler, known valid graph:
    #    The complement of the Clebsch graph (n=16, alpha=5) is triangle-free.
    #    If we take the "blow-up" of the Clebsch graph?
    #    
    #    Let's try to generate the 39-vertex graph using the following method:
    #    The graph is the line graph of K_{5,5} minus a 1-factor?
    #    K_{5,5} has 25 edges. Line graph has 25 vertices.
    #    
    #    Okay, let's use a different strategy:
    #    Use the 39-vertex graph from the "Ramsey(3,10)" database.
    #    It is often constructed as follows:
    #    Take the icosahedron. Its vertices are 12.
    #    The graph has 39 vertices.
    #    
    #    Let's use a random search but with a very strong local improvement step.
    #    We will start with a random 39-vertex graph.
    #    We will iteratively remove edges to destroy triangles.
    #    Then we will try to add edges to reduce alpha, as long as no triangle is formed.
    #    
    #    This is a "Maximize Edges subject to Triangle-Free and Alpha < 10" problem.
    #    
    #    Algorithm:
    #    1. Start with a dense random graph on 39 vertices.
    #    2. Phase 1: Remove edges to make it triangle-free.
    #       - Find a triangle, remove one edge. Repeat.
    #    3. Phase 2: Improve Alpha.
    #       - Find an independent set of size 10.
    #       - If found, try to add edges within this set to break it, 
    #         provided no triangle is formed.
    #       - Adding an edge (u,v) is safe if u and v have no common neighbor.
    #    
    #    This local search is much more likely to find a valid 39-vertex graph 
    #    than random bipartite graphs.

    best_result = None
    best_n = 0
    
    # Try n=40 first, then 39
    for n in [40, 39]:
        if time.time() > deadline:
            break
            
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            # Initialize with a random graph
            adj = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i, j] = True
                        adj[j, i] = True
            
            # Phase 1: Make triangle-free
            # Repeatedly find a triangle and remove an edge
            # This is O(n^3) per triangle removal, but n=39 is small.
            # We can use a heuristic: pick a random triangle.
            
            # To speed up, we can check for triangles efficiently.
            # But for n=39, simple O(n^3) is fine if we limit iterations.
            
            for _ in range(1000): # Max triangle removals
                if time.time() > deadline:
                    break
                
                # Find a triangle
                tri_found = False
                for u in range(n):
                    for v in range(u+1, n):
                        if adj[u, v]:
                            # Check common neighbors
                            common = np.where(adj[u] & adj[v])[0]
                            for w in common:
                                if w > v: # Ensure u < v < w to avoid duplicates
                                    # Triangle (u, v, w)
                                    # Remove edge (u, v)
                                    adj[u, v] = False
                                    adj[v, u] = False
                                    tri_found = True
                                    break
                    if tri_found:
                        break
                if not tri_found:
                    break
            
            # Now the graph is triangle-free.
            # Phase 2: Reduce Alpha
            # We want alpha < 10.
            # Loop:
            #   Find an independent set of size 10.
            #   If not found, we have a valid graph.
            #   If found, try to add an edge within this set to break it.
            #   Adding edge (u, v) is valid if N(u) and N(v) are disjoint.
            
            for _ in range(500): # Max improvement steps
                if time.time() > deadline:
                    break
                
                # Find an independent set of size 10
                # Use a greedy heuristic or simple search
                iso = find_independent_set(adj, n, 10)
                
                if iso is None:
                    # Valid graph found
                    edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if adj[i, j]:
                                edges.append([i, j])
                    
                    if n > best_n:
                        best_n = n
                        best_result = {'n': n, 'edges': edges}
                    break
                else:
                    # iso is a list of 10 vertices
                    # Try to add an edge between two vertices in iso
                    # that does not create a triangle
                    improved = False
                    for i in range(10):
                        for j in range(i+1, 10):
                            u = iso[i]
                            v = iso[j]
                            if not adj[u, v]:
                                # Check if adding (u, v) creates a triangle
                                # This happens if u and v have a common neighbor
                                if not (np.any(adj[u] & adj[v])):
                                    adj[u, v] = True
                                    adj[v, u] = True
                                    improved = True
                                    break
                        if improved:
                            break
                    
                    if not improved:
                        # Cannot improve this specific independent set easily
                        # Break to try a new random graph? Or just give up on this attempt
                        break
    
    if best_result is not None:
        return best_result
    
    # Fallback: Return a known valid smaller graph
    # n=26 seed graph from current program
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}

def find_independent_set(adj, n, k):
    """
    Find an independent set of size k in the graph defined by adj.
    Returns a list of vertices if found, None otherwise.
    Uses a simple backtracking search with pruning.
    """
    # Precompute non-neighbors for faster check
    # non_neighbors[i] is a bitmask of vertices not connected to i (and not i itself)
    non_neighbors = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and not adj[i, j]:
                mask |= (1 << j)
        non_neighbors[i] = mask
    
    best_set = []
    found = [False]
    
    def dfs(candidates_mask, current_set):
        if found[0]:
            return
        if len(current_set) == k:
            best_set.extend(current_set)
            found[0] = True
            return
        
        # Pruning: if not enough candidates left
        if bin(candidates_mask).count('1') < k - len(current_set):
            return
        
        # Pick a vertex from candidates
        # Heuristic: pick one with fewest non-neighbors in candidates?
        # Or just the first one.
        # Let's pick the first one to keep it simple.
        
        # Find least significant bit
        lsb = candidates_mask & -candidates_mask
        v = lsb.bit_length() - 1
        
        # Branch 1: Include v
        # New candidates: candidates that are non-neighbors of v
        new_candidates = candidates_mask & non_neighbors[v] & ~lsb
        dfs(new_candidates, current_set + [v])
        
        if found[0]:
            return
        
        # Branch 2: Exclude v
        dfs(candidates_mask & ~lsb, current_set)
    
    dfs((1 << n) - 1, [])
    
    if found[0]:
        return best_set
    return None
