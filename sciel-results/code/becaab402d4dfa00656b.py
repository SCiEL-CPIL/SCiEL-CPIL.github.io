def construct():
    import time
    import random
    from fractions import Fraction
    
    t0 = time.time()
    DEADLINE = 55.0  # Safety margin under 60s limit

    def elapsed_ok():
        return (time.time() - t0) < DEADLINE

    # Denominators: 2, 3, ..., 48 -> indices 0..46
    denoms = list(range(2, 49))
    n = len(denoms)
    
    # Precompute float values for fast evaluation during search
    reciprocals = [1.0 / d for d in denoms]
    
    best_deficit = None
    best_indices = None
    best_sum_frac = None

    def update_best(indices, sum_frac):
        nonlocal best_deficit, best_indices, best_sum_frac
        if not indices:
            return
        if sum_frac < 1:
            d = 1 - sum_frac
            if best_deficit is None or d < best_deficit:
                best_deficit = d
                best_indices = list(indices)
                best_sum_frac = sum_frac

    # 1. Greedy Initialization
    # Sort by reciprocal value descending (smallest denominator first) to pack tight
    indices_sorted = sorted(range(n), key=lambda i: reciprocals[i], reverse=True)
    
    current_indices = []
    current_sum = Fraction(0)
    for i in indices_sorted:
        if current_sum + Fraction(1, denoms[i]) < 1:
            current_indices.append(i)
            current_sum += Fraction(1, denoms[i])
    
    update_best(current_indices, current_sum)
    
    if not elapsed_ok():
        return best_indices

    # 2. Local Search (Hill Climbing / Simulated Annealing style)
    # We maintain a set of selected indices for O(1) add/remove checks
    selected_set = set(current_indices)
    current_sum_frac = current_sum
    current_sum_float = sum(reciprocals[i] for i in current_indices)
    
    # Random seed for reproducibility in case of ties, though not strictly required
    random.seed(42)
    
    while elapsed_ok():
        # Randomly pick a move type
        move_type = random.randint(0, 2)
        
        if move_type == 0:
            # Try adding a random unselected item
            unselected = [i for i in range(n) if i not in selected_set]
            if not unselected:
                continue
            i = random.choice(unselected)
            new_sum_float = current_sum_float + reciprocals[i]
            if new_sum_float < 1.0:
                # Verify with Fraction to be exact
                new_sum_frac = current_sum_frac + Fraction(1, denoms[i])
                if new_sum_frac < 1:
                    # Accept if it improves deficit (smaller 1-sum) or just accept to explore?
                    # Strictly, we want to minimize deficit. 
                    # If we add, sum increases, deficit decreases. So adding is always an improvement 
                    # if valid.
                    selected_set.add(i)
                    current_indices.append(i)
                    current_sum_frac = new_sum_frac
                    current_sum_float = new_sum_float
                    update_best(current_indices, current_sum_frac)
                    
        elif move_type == 1:
            # Try removing a random selected item
            if not current_indices:
                continue
            i = random.choice(current_indices)
            # Removing increases deficit, so this is only useful if it allows adding a larger item later?
            # Or if we are stuck? 
            # Actually, removing makes the sum smaller, deficit larger. This is a "downhill" move.
            # We use this to escape local minima if we allow simulated annealing, but here we stick to 
            # strict improvement or swaps. 
            # However, sometimes removing a small item allows adding a larger one? No, set is fixed.
            # Let's skip pure removals as they worsen the objective directly.
            pass
            
        else:
            # Try swapping: remove one, add another
            if not current_indices:
                continue
            i_remove = random.choice(current_indices)
            unselected = [i for i in range(n) if i not in selected_set]
            if not unselected:
                continue
            i_add = random.choice(unselected)
            
            # Check validity
            delta_float = reciprocals[i_add] - reciprocals[i_remove]
            new_sum_float = current_sum_float + delta_float
            if new_sum_float < 1.0:
                new_sum_frac = current_sum_frac - Fraction(1, denoms[i_remove]) + Fraction(1, denoms[i_add])
                if new_sum_frac < 1:
                    # Check if it improves deficit
                    old_deficit = 1 - current_sum_frac
                    new_deficit = 1 - new_sum_frac
                    if new_deficit < old_deficit:
                        # Accept swap
                        selected_set.remove(i_remove)
                        current_indices.remove(i_remove)
                        selected_set.add(i_add)
                        current_indices.append(i_add)
                        current_sum_frac = new_sum_frac
                        current_sum_float = new_sum_float
                        update_best(current_indices, current_sum_frac)
                    else:
                        # Even if it doesn't improve deficit, it might be a valid state.
                        # But we want to minimize deficit. If it's worse, we revert.
                        # However, sometimes a worse state leads to a better one? 
                        # With swaps, if new_deficit > old_deficit, we revert.
                        pass

        # Occasionally, try a "random restart" or "perturbation" if stuck?
        # For now, the swap mechanism is sufficient for this small N.
        
        # Optimization: If we find a deficit of 0 (impossible) or very small, we can stop early?
        # The task asks for minimal positive deficit. 
        # If we reach a point where no single add or swap improves it, we might be at a local minimum.
        # To ensure global optimality or near-optimality, we can run multiple restarts if time permits.
        
        # Simple heuristic: If no improvement in last 10000 steps, restart with random init?
        # Given N=47, the search space is small enough that greedy + swaps should find the optimal or near-optimal quickly.
        
        # Let's add a counter to break out of infinite loops if no progress is made?
        # The while loop is bounded by time.

    return best_indices
