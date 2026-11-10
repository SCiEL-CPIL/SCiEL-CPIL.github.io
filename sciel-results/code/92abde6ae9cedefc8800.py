import itertools

def construct():
    # Initialize with Grötzsch graph (vertices 0..10)
    # This graph is triangle-free and 4-chromatic.
    edge_set = set()
    
    # Cycle 0-1-2-3-4-0
    cycle = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    for u, v in cycle:
        edge_set.add((u, v))
        edge_set.add((v, u))
        
    # Mycielski copies 5..9
    # 5 connected to 1, 4
    # 6 connected to 0, 2
    # 7 connected to 1, 3
    # 8 connected to 2, 4
    # 9 connected to 3, 0
    copy_edges = [
        (5, 1), (5, 4),
        (6, 0), (6, 2),
        (7, 1), (7, 3),
        (8, 2), (8, 4),
        (9, 3), (9, 0)
    ]
    for u, v in copy_edges:
        edge_set.add((u, v))
        edge_set.add((v, u))
        
    # Apex 10 connected to 5, 6, 7, 8, 9
    for v in range(5, 10):
        edge_set.add((10, v))
        edge_set.add((v, 10))
        
    current_vertices = list(range(11))
    
    # Helper to find Maximum Independent Set (MIS) using brute force for small n
    def find_mis(vertices, e_set):
        best_set = []
        # Try largest subsets first
        for r in range(len(vertices), -1, -1):
            for subset in itertools.combinations(vertices, r):
                is_indep = True
                # Check all pairs in subset
                for i in range(len(subset)):
                    for j in range(i + 1, len(subset)):
                        u, v = subset[i], subset[j]
                        if (u, v) in e_set or (v, u) in e_set:
                            is_indep = False
                            break
                    if not is_indep:
                        break
                if is_indep:
                    return list(subset)
        return best_set

    # Greedily add vertices 11 to 15
    for new_v in range(11, 16):
        # Find the Maximum Independent Set of the current graph
        mis = find_mis(current_vertices, edge_set)
        
        # Connect new_v to all vertices in the MIS
        for u in mis:
            edge_set.add((new_v, u))
            edge_set.add((u, new_v))
            
        # Add new vertex to the list
        current_vertices.append(new_v)
        
    # Convert edge_set to list of lists [u, v] with u < v
    final_edges = []
    for u, v in edge_set:
        if u < v:
            final_edges.append([u, v])
            
    return final_edges
