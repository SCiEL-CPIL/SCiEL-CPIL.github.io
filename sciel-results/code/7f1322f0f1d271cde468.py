def is_k4_valid_fast(k4_edges, colors):
    c0 = colors[k4_edges[0]]
    c1 = colors[k4_edges[1]]
    c2 = colors[k4_edges[2]]
    c3 = colors[k4_edges[3]]
    c4 = colors[k4_edges[4]]
    c5 = colors[k4_edges[5]]
    
    # Count distincts
    distinct = 1
    if c1 != c0: distinct += 1
    if c2 != c0 and c2 != c1: distinct += 1
    if c3 != c0 and c3 != c1 and c3 != c2: distinct += 1
    if c4 != c0 and c4 != c1 and c4 != c2 and c4 != c3: distinct += 1
    if c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: distinct += 1
    
    return distinct >= 5
