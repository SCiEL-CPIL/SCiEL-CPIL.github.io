import itertools

def construct():
    """
    Returns a graph with n=32 vertices, no 5-cycle, and no independent set of size 9.
    
    Construction:
    The graph is the disjoint union of 8 copies of the complete graph K4.
    
    Properties:
    1. C5-free: K4 contains no cycles of length 5. Since the components are disjoint,
       no cycle can span multiple components. Therefore, the entire graph is C5-free.
    2. Independence Number: The independence number of a disjoint union of cliques
       is the sum of the independence numbers of the cliques. For a K4, alpha=1.
       Thus, alpha(G) = 8 * 1 = 8. This satisfies the condition alpha(G) < 9.
    3. Vertex Count: n = 8 * 4 = 32.
    
    This construction certifies R(C5, K9) > 32, which is the best known lower bound
    for this Ramsey number via simple constructive methods.
    """
    n = 32
    edges = []
    
    # The graph consists of 8 disjoint blocks of 4 vertices each (K4)
    num_blocks = 8
    block_size = 4
    
    for b in range(num_blocks):
        # Vertices in this block
        block_vertices = range(b * block_size, (b + 1) * block_size)
        
        # Add all edges within the block to form a K4
        for u, v in itertools.combinations(block_vertices, 2):
            edges.append([u, v])
            
    return {'n': n, 'edges': edges}
