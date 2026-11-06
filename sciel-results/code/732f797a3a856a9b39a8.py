# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a graph on 32 vertices consisting of 8 disjoint copies of K4.
    
    This construction is optimal within the family of disjoint unions of cliques:
    1. Each component must be a clique K_m with m <= 4 to be C5-free (K5 contains C5).
    2. The independence number of a disjoint union is the sum of the independence numbers
       of its components. alpha(K_m) = 1 for m >= 1.
    3. To avoid an independent set of size 9, we need sum(alpha) < 9, i.e., at most 8 components.
    4. To maximize n, we maximize the size of each component (m=4) and the number of components (k=8).
    5. n = 8 * 4 = 32.
    """
    n = 32
    edges = []
    
    # Generate edges for 8 disjoint K4s
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
