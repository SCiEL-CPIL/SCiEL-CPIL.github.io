    if hasattr(dag, "keys"):
        items = list(dag.items())
    else:
        # maybe list of (vertex, parents)
        items = list(dag)

    parents = {}
    for v, p in items:
        parents[v] = list(p)
