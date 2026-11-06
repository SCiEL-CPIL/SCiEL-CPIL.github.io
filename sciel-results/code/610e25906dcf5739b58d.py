    cycle = [(i, (i + 1) % 5) for i in range(5)]
    edges = [list(e) for e in cycle] # 5 edges
    for u, w in cycle:
        edges += [[u, 5 + w], [w, 5 + u]] # 10 edges
    return edges + [[5 + u, 10] for u in range(5)] # 5 edges
