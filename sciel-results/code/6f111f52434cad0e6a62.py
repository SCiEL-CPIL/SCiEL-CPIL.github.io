    def normalize_vertex(v):
        if isinstance(v, list):
            return tuple(v)
        return v

    def normalize_parents(parents):
        if parents is None:
            return []
        if not isinstance(parents, (list, tuple)):
            parents = [parents]
        norm = []
        for p in parents:
            if isinstance(p, list):
                p = tuple(p)
            norm.append(p)
        return norm
   