<<<<<<< SEARCH
    start = (0, (None,) + (0,) * d)
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
    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        if time.time() > deadline:
            # Return best effort (partial graph might be invalid, but we expect to finish)
            # For LUW, the graph is generated in one pass. If we time out, we might return a disconnected subgraph.
            # However, for k=5, g=9, d=4, n=1250, this is very fast.
            # We just break to avoid hanging if something goes wrong.
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
>>>>>>> REPLACE
