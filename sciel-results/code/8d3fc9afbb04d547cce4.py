<<<<<<< SEARCH
    best = set()
    
    # Try multiple random orderings for greedy
    random.seed(42)
    for attempt in range(50):
        if time.time() > deadline:
            break
        order = valid_nodes[:]
        random.shuffle(order)
        chosen = greedy_mis(order)
        chosen = local_search(chosen, max_iters=5000)
        if len(chosen) > len(best):
            best = chosen
    
    # Also try sorted orders
    for order in [valid_nodes, sorted(valid_nodes, reverse=True), sorted(valid_nodes, key=lambda x: len(conflicts[x]))]:
        if time.time() > deadline:
            break
        chosen = greedy_mis(order)
        chosen = local_search(chosen, max_iters=5000)
        if len(chosen) > len(best):
            best = chosen
=======
    best = set()
    
    # Try multiple random orderings for greedy
    random.seed(42)
    for attempt in range(80):
        if time.time() > deadline:
            break
        order = valid_nodes[:]
        random.shuffle(order)
        chosen = greedy_mis(order)
        chosen = local_search(chosen, max_iters=5000)
        if len(chosen) > len(best):
            best = chosen
    
    # Also try sorted orders with various heuristics
    orders = [
        valid_nodes,
        sorted(valid_nodes, reverse=True),
        sorted(valid_nodes, key=lambda x: len(conflicts[x])),
        sorted(valid_nodes, key=lambda x: (-len(conflicts[x]), x)),
        sorted(valid_nodes, key=lambda x: (x % 7, len(conflicts[x]))),
    ]
    for order in orders:
        if time.time() > deadline:
            break
        chosen = greedy_mis(order)
        chosen = local_search(chosen, max_iters=5000)
        if len(chosen) > len(best):
            best = chosen
    
    # Final polish: run local search on best with more iterations
    if time.time() < deadline - 5:
        best = local_search(best, max_iters=20000)
>>>>>>> REPLACE
