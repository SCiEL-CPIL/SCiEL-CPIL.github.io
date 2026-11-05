    if not vertices and isinstance(dag, (list, tuple)):
        for item in dag:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                v = item[0]
                p = item[1]
                vertices.append(v)
                parents[v] = list(p) if p else []
