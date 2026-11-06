    while time.time() < deadline:
        if random.random() < 0.7:
            # Local search from current best
            start_net = list(best)
        else:
            # Perturb seed
            ...
            start_net = _greedy_reduce(start_net, min(deadline, time.time() + 3))
        
        candidate = _memetic_local_search(start_net, deadline)
        ...
