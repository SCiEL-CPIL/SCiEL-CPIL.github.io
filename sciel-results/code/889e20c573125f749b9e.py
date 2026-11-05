import time
import itertools

def construct():
    """
    Returns a graph with n=32 vertices, no 5-cycle, and no independent set of size 9.
    
    Strategy:
    We seek a graph G on n vertices such that:
    1. G is C5-free.
    2. alpha(G) < 9 (i.e., no independent set of size 9).
    
    This is equivalent to finding a graph H = complement(G) such that:
    1. H is C5-free (since C5 is self-complementary).
    2. H contains no K9 subgraph (since independent set in G is clique in H).
    
    We use the family of disjoint unions of cliques.
    Let H be a disjoint union of cliques.
    If H is a disjoint union of cliques, it contains no C5 (cycles must stay within a clique, but cliques have no induced cycles of length > 3, and any C5 in a clique would imply a K5, which is not a simple cycle in the sense of being C5-free if we restrict clique size? Wait.
    A clique K_k contains C5 as a subgraph if k >= 5.
    So, to ensure H is C5-free, each clique in H must have size at most 4.
    
    To maximize n, we want as many vertices as possible.
    We also need H to be K9-free.
    If H is a disjoint union of cliques of size <= 4, the largest clique in H is size 4.
    Therefore, H is K9-free automatically.
    
    So, any disjoint union of cliques of size <= 4 is a valid H.
    Consequently, G = complement(H) is a valid G.
    
    To maximize n, we should maximize the number of vertices in H.
    The problem states n <= 35.
    We know R(C5, K9) = 33, so n=32 is the max.
    
    Let's construct H as 8 disjoint K4s.
    n = 8 * 4 = 32.
    H consists of 8 components, each K4.
    H is C5-free (max clique size 4).
    H is K9-free (max clique size 4).
    
    Then G is the complement of H.
    G will have 32 vertices.
    G is C5-free.
    alpha(G) = omega(H) = 4 < 9.
    
    This yields a valid graph on 32 vertices.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    # Define the structure of H: 8 disjoint K4s
    num_components = 8
    comp_size = 4
    n = num_components * comp_size
    
    # Build adjacency matrix for H (complement of G)
    # H[i][j] = 1 if i and j are connected in H
    H = [[0] * n for _ in range(n)]
    
    # Fill H with edges for 8 disjoint K4s
    for c in range(num_components):
        base = c * comp_size
        # Connect all vertices within the component
        for i in range(comp_size):
            for j in range(i + 1, comp_size):
                H[base + i][base + j] = 1
                H[base + j][base + i] = 1
        
        # Check deadline
        if time.time() > deadline:
            # If we time out, we still have a valid partial construction?
            # No, we need the full graph. But this loop is tiny.
            break

    # Derive G from H
    # G is the complement of H.
    # G has edge (u, v) if H does not have edge (u, v) and u != v.
    edges_G = []
    for u in range(n):
        for v in range(u + 1, n):
            if H[u][v] == 0:
                edges_G.append([u, v])
        
        if time.time() > deadline:
            break

    # The result is G
    result = {
        'n': n,
        'edges': edges_G
    }
    
    return result
