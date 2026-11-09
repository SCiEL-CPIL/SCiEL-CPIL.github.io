import numpy as np
import time
from collections import deque

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n = adjacency.shape[0]
    adjacency = adjacency.astype(bool)
    
    # Precompute neighbor lists
    neighbor_lists = [np.where(adjacency[i])[0] for i in range(n)]
    degrees = np.array([len(nbrs) for nbrs in neighbor_lists], dtype=float)
    
    # Fast simulation function
    def simulate_greedy(order):
        available = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if available[v]:
                count += 1
                available[v] = False
                nbrs = neighbor_lists[v]
                if len(nbrs) > 0:
                    available[nbrs] = False
        return count

    # Compute K-core numbers for better heuristics
    def k_core_decomposition():
        core_number = np.zeros(n, dtype=int)
        removed = np.zeros(n, dtype=bool)
        current_degree = degrees.astype(int).copy()
        
        # Standard peeling
        while True:
            min_deg = n
            min_v = -1
            for v in range(n):
                if not removed[v] and current_degree[v] < min_deg:
                    min_deg = current_degree[v]
                    min_v = v
            if min_v == -1:
                break
            removed[min_v] = True
            core_number[min_v] = min_deg
            for nbr in neighbor_lists[min_v]:
                if not removed[nbr]:
                    current_degree[nbr] -= 1
        for v in range(n):
            if not removed[v]:
                core_number[v] = current_degree[v]
        return core_number

    core_numbers = k_core_decomposition()
    
    # Normalize features for mixing
    deg_norm = degrees / (degrees.max() + 1e-9)
    core_norm = core_numbers.astype(float) / (core_numbers.max() + 1e-9)
    
    rng = np.random.RandomState(42)
    
    best_order = None
    best_size = 0
    
    # Generate diverse initial permutations
    initial_orders = []
    
    # 1. Degree ascending (Low degree first)
    initial_orders.append(np.argsort(degrees))
    
    # 2. Core number ascending (Peripheral first)
    initial_orders.append(np.argsort(core_numbers))
    
    # 3. Combined: 0.7 * degree + 0.3 * core
    combo1 = 0.7 * deg_norm + 0.3 * core_norm
    initial_orders.append(np.argsort(combo1))
    
    # 4. Combined: 0.3 * degree + 0.7 * core
    combo2 = 0.3 * deg_norm + 0.7 * core_norm
    initial_orders.append(np.argsort(combo2))
    
    # 5. Random
    initial_orders.append(rng.permutation(n))
    
    # 6. Degree with noise
    deg_noise = degrees + rng.uniform(0, 1, n) * 0.5
    initial_orders.append(np.argsort(deg_noise))
    
    # 7. Core with noise
    core_noise = core_numbers + rng.uniform(0, 1, n) * 0.5
    initial_orders.append(np.argsort(core_noise))
    
    # Local search function
    def local_search(init_order, time_limit):
        curr_order = init_order.copy()
        curr_size = simulate_greedy(curr_order)
        
        # Phase 1: Random Swap Hill Climbing
        improved = True
        no_improve_count = 0
        max_no_improve = 100
        
        while time.time() < time_limit and no_improve_count < max_no_improve:
            improved = False
            # Try a random swap
            i, j = rng.randint(0, n, 2)
            if i != j:
                new_order = curr_order.copy()
                new_order[i], new_order[j] = new_order[j], new_order[i]
                new_size = simulate_greedy(new_order)
                if new_size > curr_size:
                    curr_size = new_size
                    curr_order = new_order
                    improved = True
                    no_improve_count = 0
                elif new_size == curr_size:
                    # Accept equal to escape plateaus? No, stick to strict improvement for now
                    pass
                else:
                    no_improve_count += 1
            
            # If stuck, try adjacent swaps in a sweep
            if no_improve_count > 50:
                for i in range(n - 1):
                    if time.time() > time_limit:
                        break
                    new_order = curr_order.copy()
                    new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
                    new_size = simulate_greedy(new_order)
                    if new_size > curr_size:
                        curr_size = new_size
                        curr_order = new_order
                        improved = True
                        no_improve_count = 0
                        break
                    else:
                        no_improve_count += 1
        
        # Phase 2: Insertion Move Hill Climbing
        # Moving a vertex to a different position is a stronger move
        improved = True
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < time_limit and no_improve_count < max_no_improve:
            improved = False
            i, j = rng.randint(0, n, 2)
            if i != j:
                # Move element at i to position j
                v = curr_order[i]
                new_order = np.delete(curr_order, i)
                if j > i:
                    j -= 1
                new_order = np.insert(new_order, j, v)
                
                new_size = simulate_greedy(new_order)
                if new_size > curr_size:
                    curr_size = new_size
                    curr_order = new_order
                    improved = True
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        
        return curr_order, curr_size

    # Run local search for each initial order, allocating time based on remaining budget
    # We have ~145s. Let's spend 10s on each of the first 7 orders? No, let's be adaptive.
    
    # First, quickly evaluate all initial orders to find the best start
    for init_order in initial_orders:
        if time.time() > deadline - 5:
            break
        size = simulate_greedy(init_order)
        if size > best_size:
            best_size = size
            best_order = init_order
    
    # Now, perform local search on the best initial order found, and also try a few others
    # We will run local search on the top 3 initial orders
    
    # Sort initial orders by their initial size
    init_evals = []
    for init_order in initial_orders:
        size = simulate_greedy(init_order)
        init_evals.append((size, init_order))
    init_evals.sort(key=lambda x: x[0], reverse=True)
    
    # Run local search on the best 3 starts
    for size, init_order in init_evals[:3]:
        if time.time() > deadline - 10:
            break
        
        # Allocate remaining time to this search, but leave 5s buffer
        search_deadline = deadline - 5
        # If we have multiple searches, split the time
        # For simplicity, just run until near deadline for the first, then less for others?
        # Better: Run one long search on the best, then if time remains, try others.
        
        # Let's just run a robust local search on the single best initial order
        # and use the remaining time to refine it further.
        
        refined_order, refined_size = local_search(init_order, search_deadline)
        
        if refined_size > best_size:
            best_size = refined_size
            best_order = refined_order
            
        # If we still have time, do a final polish on the best order found so far
        if time.time() < deadline - 2 and best_order is not None:
            # Final polish: adjacent swaps
            improved = True
            while improved and time.time() < deadline - 1:
                improved = False
                for i in range(n - 1):
                    if time.time() > deadline - 1:
                        break
                    new_order = best_order.copy()
                    new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
                    new_size = simulate_greedy(new_order)
                    if new_size > best_size:
                        best_size = new_size
                        best_order = new_order
                        improved = True

    # Convert best order to scores
    scores = np.zeros(n)
    if best_order is not None:
        for pos, v in enumerate(best_order):
            scores[v] = -pos
    else:
        # Fallback: random scores
        scores = rng.uniform(0, 1, n) * -1

    return scores
