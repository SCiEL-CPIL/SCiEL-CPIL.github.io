def construct():
    import time
    start_time = time.time()
    deadline = start_time + 59.0  # Safety margin before 90s limit

    # Construct Fano Plane from PG(2,2)
    # Points are non-zero vectors in GF(2)^3
    # Index 0: 100, 1: 010, 2: 001, 3: 110, 4: 101, 5: 011, 6: 111
    vectors = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1)
    ]
    
    # Map vector to index for quick lookup
    vec_to_idx = {v: i for i, v in enumerate(vectors)}
    
    edges_set = set()
    for i in range(7):
        for j in range(i + 1, 7):
            # Sum of vectors i and j in GF(2) is XOR
            v_sum = (vectors[i][0] ^ vectors[j][0], 
                     vectors[i][1] ^ vectors[j][1], 
                     vectors[i][2] ^ vectors[j][2])
            if v_sum in vec_to_idx:
                k = vec_to_idx[v_sum]
                if k != i and k != j:
                    # Add edge {i, j, k} as a sorted tuple to avoid duplicates
                    edge = tuple(sorted((i, j, k)))
                    edges_set.add(edge)
    
    # Convert to list of lists
    edges = [list(e) for e in edges_set]
    
    # Verify 3-chromaticity by checking all 2^7 binary colorings
    # If no 2-coloring works, it is 3-chromatic (assuming it's 3-colorable, which Fano is)
    is_2_colorable = False
    for mask in range(128):
        colors_2 = [(mask >> i) & 1 for i in range(7)]
        valid_2 = True
        for e in edges:
            if colors_2[e[0]] == colors_2[e[1]] == colors_2[e[2]]:
                valid_2 = False
                break
        if valid_2:
            is_2_colorable = True
            break
            
    if is_2_colorable:
        # This should never happen for the Fano plane
        pass

    # Provide a valid 3-coloring
    # The Fano plane is 3-chromatic. A valid 3-coloring is:
    # 0: 0, 1: 1, 2: 2, 3: 1, 4: 2, 5: 0, 6: 1
    colors_3 = [0, 1, 2, 1, 2, 0, 1]
    
    # Verify the 3-coloring is valid (no monochromatic edges)
    valid_3 = True
    for e in edges:
        if colors_3[e[0]] == colors_3[e[1]] == colors_3[e[2]]:
            valid_3 = False
            break
            
    if not valid_3:
        # Fallback if hardcoded coloring fails (shouldn't happen)
        pass

    # Vertices 7 and 8 are isolated
    colors = colors_3 + [0, 0]

    if time.time() > deadline:
        return {'edges': edges, 'colors': colors}

    return {'edges': edges, 'colors': colors}
