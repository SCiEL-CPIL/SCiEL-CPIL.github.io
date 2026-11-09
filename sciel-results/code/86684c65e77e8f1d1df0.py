<<<<<<< SEARCH
    # Evaluate initial orders
    for order in initial_orders:
        size = greedy_mis_size(order)
        update_best(order, size)
    
    # Local search: Simulated Annealing with swaps
    # We'll use a time-based loop to ensure we finish in time.
    # For each initial order, we'll do local search.
    
    num_restarts = len(initial_orders)
    steps_per_restart = 8000
    
    # If we have time, we can do more restarts or steps.
    # Let's use a global time budget.
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        order = initial_orders[r].copy()
        current_size = greedy_mis_size(order)
        update_best(order, current_size)
        
        # Simulated Annealing
        # Start with high temperature, cool down over time
        initial_temp = 2.0
        cooling_rate = 0.9995
        
        temperature = initial_temp
        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break
            
            temperature *= cooling_rate
            
            # Pick two random indices to swap
            i, j = rng.integers(0, n, 2)
            if i == j:
                continue
                
            order[i], order[j] = order[j], order[i]
            
            new_size = greedy_mis_size(order)
            
            delta = new_size - current_size
            
            if delta >= 0:
                # Accept better or equal solutions
                current_size = new_size
                update_best(order, current_size)
            else:
                # Accept worse solutions with probability exp(delta / temperature)
                if temperature > 1e-6:
                    prob = np.exp(delta / temperature)
                    if rng.random() < prob:
                        current_size = new_size
                    else:
                        # Revert
                        order[i], order[j] = order[j], order[i]
                else:
                    # Revert
                    order[i], order[j] = order[j], order[i]
=======
    # Evaluate initial orders
    for order in initial_orders:
        size = greedy_mis_size(order)
        update_best(order, size)
    
    # Local search: Simulated Annealing with swaps
    # Use time-based budget to maximize iterations within deadline
    
    num_restarts = len(initial_orders)
    
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        order = initial_orders[r].copy()
        current_size = greedy_mis_size(order)
        update_best(order, current_size)
        
        # Simulated Annealing
        initial_temp = 2.0
        cooling_rate = 0.9998  # Slower cooling for more exploration
        
        temperature = initial_temp
        step = 0
        while time.time() < deadline - 20:
            step += 1
            temperature *= cooling_rate
            
            # Pick two random indices to swap
            i, j = rng.integers(0, n, 2)
            if i == j:
                continue
                
            order[i], order[j] = order[j], order[i]
            
            new_size = greedy_mis_size(order)
            
            delta = new_size - current_size
            
            if delta >= 0:
                current_size = new_size
                update_best(order, current_size)
            else:
                if temperature > 1e-6:
                    prob = np.exp(delta / temperature)
                    if rng.random() < prob:
                        current_size = new_size
                    else:
                        order[i], order[j] = order[j], order[i]
                else:
                    order[i], order[j] = order[j], order[i]
    
    # Final polish pass: try targeted improvements on the best solution
    if best_order is not None and time.time() < deadline - 10:
        polish_order = best_order.copy()
        polish_size = best_size
        
        # Try swapping each vertex with nearby positions
        for i in range(n):
            if time.time() > deadline - 10:
                break
            for j in range(max(0, i-5), min(n, i+6)):
                if i == j:
                    continue
                polish_order[i], polish_order[j] = polish_order[j], polish_order[i]
                new_size = greedy_mis_size(polish_order)
                if new_size > polish_size:
                    polish_size = new_size
                    best_size = polish_size
                    best_order = polish_order.copy()
                else:
                    polish_order[i], polish_order[j] = polish_order[j], polish_order[i]
>>>>>>> REPLACE

