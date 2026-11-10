import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    k = 2
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    # Forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        val = pow(x, k, m)
        if val != 0:
            forbidden.add(val)

    # Build conflict matrix
    # conflict[i, j] is True if i and j cannot be in the set together
    conflict = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if d in forbidden:
                conflict[i, j] = True
            else:
                d2 = (j - i) % m
                if d2 in forbidden:
                    conflict[i, j] = True

    # Initial greedy solution
    # We want to maximize the number of elements.
    # Greedy: pick node with minimum degree in the remaining graph?
    # Or just random greedy.
    
    def greedy_init(order=None):
        if order is None:
            order = list(range(m))
            random.shuffle(order)
        selected = np.zeros(m, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                # Check if v conflicts with any selected
                if not np.any(conflict[v] & selected):
                    selected[v] = True
                    count += 1
        return selected, count

    # Get a good initial solution
    best_sel, best_count = greedy_init()
    current_sel, current_count = best_sel.copy(), best_count
    current_set_indices = np.where(current_sel)[0]

    # Local Search: Try to add any possible node
    def try_add_all(sel):
        changed = True
        while changed:
            changed = False
            # Find all nodes that can be added
            # A node v can be added if not sel[v] and not np.any(conflict[v] & sel)
            # Vectorized: 
            # conflicts_with_selected = np.any(conflict & sel[None, :], axis=1)
            # candidates = ~sel & ~conflicts_with_selected
            # But we need to be careful: adding one might block another.
            # Greedy fill:
            candidates = np.where(~sel)[0]
            random.shuffle(candidates) # Add randomness to break ties
            for v in candidates:
                if not sel[v]:
                    if not np.any(conflict[v] & sel):
                        sel[v] = True
                        changed = True
        return sel, np.sum(sel)

    current_sel, current_count = try_add_all(current_sel)
    if current_count > best_count:
        best_count = current_count
        best_sel = current_sel.copy()

    # Main Search Loop: Simulated Annealing / Iterated Local Search
    # Move: Remove a random element, then greedily refill.
    # If the new size is larger, accept.
    # If the new size is equal, accept with probability p (to explore).
    # If smaller, accept with probability p (annealing).
    
    # We will use a simple restart/annealing scheme.
    # Since N is small, we can try many iterations.
    
    rng = random.Random(42)
    temperature = 1.0
    cooling = 0.999
    
    # Track best
    global_best_count = best_count
    global_best_sel = best_sel.copy()
    
    # Precompute degrees for potential heuristic? Not strictly necessary with random search.
    
    while time.time() < deadline:
        # 1. Try to add any possible nodes (Hill Climbing)
        current_sel, current_count = try_add_all(current_sel)
        
        if current_count > global_best_count:
            global_best_count = current_count
            global_best_sel = current_sel.copy()
            # Reset temperature on improvement?
            # temperature = 1.0 

        if current_count == m: # Max possible
            break

        # 2. Generate a neighbor: Remove a random node from current set
        if current_count > 0:
            selected_indices = np.where(current_sel)[0]
            if len(selected_indices) > 0:
                rem_idx = rng.choice(selected_indices)
                neighbor_sel = current_sel.copy()
                neighbor_sel[rem_idx] = False
                neighbor_count = current_count - 1
                
                # 3. Greedily refill the neighbor
                neighbor_sel, neighbor_count = try_add_all(neighbor_sel)
                
                delta = neighbor_count - current_count
                
                # Acceptance criterion
                accept = False
                if delta > 0:
                    accept = True
                elif delta == 0:
                    # Accept neutral moves to explore
                    if rng.random() < 0.5: 
                        accept = True
                else:
                    # Accept worse moves with annealing probability
                    if temperature > 0:
                        prob = np.exp(delta / temperature)
                        if rng.random() < prob:
                            accept = True
                
                if accept:
                    current_sel = neighbor_sel
                    current_count = neighbor_count
                    
        # Cool down
        temperature *= cooling
        
        # Occasionally restart from best to avoid getting stuck in a bad basin?
        # Or just keep going. The "remove and refill" is quite powerful.
        
        # Check time frequently
        if time.time() > deadline:
            break

    # Final polish: try to add any remaining nodes
    final_sel = global_best_sel.copy()
    final_sel, final_count = try_add_all(final_sel)
    
    # One more check: try random removals from the global best one last time
    # (In case the main loop ended just before a good move was tried)
    # Actually, the loop runs until deadline, so this is redundant but safe.
    
    result_indices = np.where(final_sel)[0]
    result = np.array(sorted(result_indices), dtype=np.int64)
    
    return result
