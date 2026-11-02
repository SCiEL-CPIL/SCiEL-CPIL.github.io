<<<<<<< SEARCH
    def build_graph(q_val, d_val):
        r = get_rule(d_val)
        def neighbours(vertex):
            side, x = vertex
            for first in range(q_val):
                y = [None, first] + [0] * (d_val - 1)
                for c in range(2, d_val + 1):
                    a, b = r[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d_val)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges
=======
    def build_graph(q_val, d_val):
        r = get_rule(d_val)
        # Precompute rule as tuple for faster access
        r_tuple = tuple(r[c] for c in range(2, d_val + 1))
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q_val):
                y_list = [None, first] + [0] * (d_val - 1)
                for idx, c in enumerate(range(2, d_val + 1)):
                    a, b = r_tuple[idx]
                    if side == 0:
                        y_list[c] = add(x[c], mul(y_list[a], x[b]))
                    else:
                        y_list[c] = sub(x[c], mul(x[a], y_list[b]))
                yield 1 - side, tuple(y_list)

        start = (0, (None,) + (0,) * d_val)
        index = {start: 0}
        order = [start]
        edges = []
        for i, vertex in enumerate(order):
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((i, index[other]))
        return len(order), edges
>>>>>>> REPLACE
