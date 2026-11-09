    def has_path_len_4(u, v):
        for a in get_neighbors(u):
            if a == v: continue # v not in N(u) anyway
            for b in get_neighbors(a):
                if b == u or b == v: continue
                for c in get_neighbors(b):
                    if c == u or c == a or c == v: continue
                    for d in get_neighbors(c):
                        if d == u or d == a or d == b or d == v: continue
                        if (adj[d] >> v) & 1:
                            return True
        return False
