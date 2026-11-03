    # Deduplicate edges
    unique_edges = set()
    for u, v in edges:
        if u == v:
            continue
        if u < v:
            unique_edges.add((u, v))
        else:
            unique_edges.add((v, u))
    edges = list(unique_edges)
