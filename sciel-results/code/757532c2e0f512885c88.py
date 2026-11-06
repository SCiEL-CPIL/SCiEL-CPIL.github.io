    # Topological order (Kahn)
    indeg = {v: 0 for v in all_vertices}
    for v, ps in parents_map.items():
        for p in ps:
            indeg[v] = indeg.get(v, 0) + 1
    # Actually indegree = number of parents
    q = deque([v for v in all_vertices if indeg.get(v, 0) == 0])
    topo = []
    while q:
        v = q.popleft()
        topo.append(v)
        for c in children_map.get(v, []):
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    if len(topo) != n:
        # cycle or missing; just use all
        topo = all_vertices
