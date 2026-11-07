    while available_edges:
        cycle = find_shortest_cycle(available_graph)
        if not cycle: break
        result.append(cycle)
        remove_edges(cycle, available_graph)
    