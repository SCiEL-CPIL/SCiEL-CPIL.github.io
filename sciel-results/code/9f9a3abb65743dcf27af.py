for i, v in enumerate(topological_order):
    if v in parents:
        for p in parents[v]:
            last_use[p] = max(last_use[p], i)
