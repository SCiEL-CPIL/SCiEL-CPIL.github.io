import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Construction: 4 disjoint copies of the complement of C8
    # C8 is triangle-free and C5-free.
    # Its complement G is C5-free and has alpha(G) = 2.
    # 4 copies give n=32 and alpha=8.
    
    n_comp = 8
    num_copies = 4
    n = n_comp * num_copies
    
    edges = []
    
    # Precompute C8 edges for the component
    c8_edges = set([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0)])
    
    for c in range(num_copies):
        offset = c * n_comp
        # Generate edges of K_n_comp
        for i in range(n_comp):
            for j in range(i + 1, n_comp):
                # Check if (i, j) is an edge in C8
                if (i, j) not in c8_edges and (j, i) not in c8_edges:
                    edges.append([offset + i, offset + j])
                    
    return {'n': n, 'edges': edges}
