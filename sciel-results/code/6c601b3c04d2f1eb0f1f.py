        unassigned = [v for v in range(n) if colors[v] == -1]
        # Sort by number of edges containing v
        unassigned.sort(key=lambda x: len(vertex_edges[x]), reverse=True)
        v = unassigned[0]
