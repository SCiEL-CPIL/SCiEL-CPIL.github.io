    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    for i, spec in enumerate(specs):
        if time.time() > deadline - 10: # 10s margin
             # Fallback for remaining
             for j in range(i, 3):
                 if j >= len(results):
                     e = instance(*specs[j])
                     results.append([[list(edge)] for edge in e])
             break
        edges = instance(*spec)
        paths = decompose_graph(edges)
        results.append(paths)
    
    return results
