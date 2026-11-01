<<<<<<< SEARCH
    if k == 4 and g == 10:
        try:
            return _cage_4_10()
        except Exception:
            pass
=======
    if k == 4 and g == 10:
        try:
            result = _cage_4_10()
            if result is not None:
                return result
        except Exception:
            pass
        # Fallback: try the L-U-W construction directly
        try:
            q = 4
            d = 5
            mul_table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            def f4_add(a,b): return a^b
            def f4_mul(a,b): return mul_table[a][b]
            def f4_sub(a,b): return a^b
            rule = {2:(1,1), 3:(2,1), 4:(1,2), 5:(3,1)}
            def neighbours(vertex):
                side, x = vertex
                for first in range(q):
                    y = [None, first] + [0]*(d-1)
                    for c in range(2, d+1):
                        a, b = rule[c]
                        if side == 0:
                            y[c] = f4_add(x[c], f4_mul(y[a], x[b]))
                        else:
                            y[c] = f4_sub(x[c], f4_mul(x[a], y[b]))
                    yield 1-side, tuple(y)
            start = (0, (None,)+(0,)*d)
            index = {start: 0}
            order = [start]
            edges = []
            for vertex in order:
                for other in neighbours(vertex):
                    if other not in index:
                        index[other] = len(order)
                        order.append(other)
                    if vertex[0] == 0:
                        edges.append((index[vertex], index[other]))
            n = len(order)
            adj = [[] for _ in range(n)]
            for u,v in edges:
                adj[u].append(v)
                adj[v].append(u)
            if all(len(adj[i])==4 for i in range(n)) and _check_girth_fast(adj, n, 10):
                return n, edges
        except Exception:
            pass
>>>>>>> REPLACE
