def construct():
    edges = []
    # Parts of size 4
    V1 = [0, 1, 2, 3]
    V2 = [4, 5, 6, 7]
    V3 = [8, 9, 10, 11]
    V4 = [12, 13, 14, 15]
    
    parts = [V1, V2, V3, V4]
    
    # For each pair of parts, add 14 edges
    # Remove 2 edges incident to the first vertex of the first part
    for i in range(4):
        for j in range(i + 1, 4):
            u_part = parts[i]
            v_part = parts[j]
            # Generate all 16 edges
            all_edges = [[u, v] for u in u_part for v in v_part]
            # Remove first 2 edges (incident to u_part[0])
            # u_part[0] is connected to v_part[0] and v_part[1]
            # So remove [u_part[0], v_part[0]] and [u_part[0], v_part[1]]
            e1 = [u_part[0], v_part[0]]
            e2 = [u_part[0], v_part[1]]
            
            for e in all_edges:
                if e != e1 and e != e2:
                    edges.append(e)
                    
    return edges
