def _reduce_hypergraph(n, edges, deadline):
    edges_set = set(tuple(e) for e in edges)
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        for e in list(edges_set):
            if time.time() > deadline:
                break
            if _is_not_2_colorable(n, edges_set - {e}):
                edges_set.remove(e)
                changed = True
    return [list(e) for e in edges_set]
