    valid = True
    for k4_idx in edge_to_k4s[e_idx]:
        e1, e2, e3, e4, e5, e6 = k4_edges[k4_idx]
        c1, c2, c3, c4, c5, c6 = current[e1], current[e2], current[e3], current[e4], current[e5], current[e6]
        # Check if distinct < 5
        # This is true if there are duplicates such that unique count < 5.
        # Equivalent to:
        # if (c1==c2 and c1==c3) or ... too many cases.
        # Just use a small set check but optimized?
        # Or: 
        # distinct = 6
        # if c1==c2: distinct-=1
        # if c1==c3: distinct-=1
        # ...
        # This is 15 comparisons. 30 K4s * 15 = 450 comparisons.
        # Set creation is likely slower.
    