<<<<<<< SEARCH
    # Strategy 1: Greedy with random edge orderings
    for trial in range(20):
        if time.time() > deadline:
            break
        order = list(all_edges)
        random.shuffle(order)
        adj, count = greedy_build(order)
        adj, count = local_search(adj, count)
        if count > best_edges:
            best_edges = count
            best_adj = list(adj)
=======
    # Strategy 1: Greedy with random edge orderings
    for trial in range(40):
        if time.time() > deadline:
            break
        order = list(all_edges)
        random.shuffle(order)
        adj, count = greedy_build(order)
        adj, count = local_search(adj, count)
        if count > best_edges:
            best_edges = count
            best_adj = list(adj)
>>>>>>> REPLACE

