# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 295.0  # Safety margin
    
    # Precompute coordinates
    xs = np.array([i // N for i in range(N * N)], dtype=np.int32)
    ys = np.array([i % N for i in range(N * N)], dtype=np.int32)
    
    all_points = list(range(N * N))
    
    # Precompute all pairwise squared distances
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist_matrix = (dx * dx + dy * dy).astype(np.int32)
    
    def solve_once(seed, time_limit):
        start = time.time()
        limit = start + time_limit
        
        rng = random.Random(seed)
        
        # Data structures
        selected = set()
        selected_list = []
        # dist_sets[i] = set of squared distances from i to other selected points
        dist_sets = {}
        
        def can_add(p):
            if p in selected:
                return False
            
            if not selected_list:
                return True

            # Optimization: Check unique distances first using numpy if list is large enough
            # For small lists, python loop is fine.
            # Let's just use the python loop for now, it's simple and correct.
            # But we can break early.
            
            seen_dists = set()
            for q in selected_list:
                d = dist_matrix[p, q]
                
                # Check if d is already seen (x as apex)
                if d in seen_dists:
                    return False
                seen_dists.add(d)
                
                # Check if d is in dist_sets[q] (q as apex)
                if d in dist_sets[q]:
                    return False
                    
            return True

        def add(p):
            selected.add(p)
            selected_list.append(p)
            dist_sets[p] = set()
            # Update dist_sets for p and existing points
            for q in selected_list[:-1]:
                d = dist_matrix[p, q]
                dist_sets[p].add(d)
                dist_sets[q].add(d)

        def remove(p):
            if p not in selected:
                return
            selected.discard(p)
            # O(N) remove
            selected_list.remove(p)
            # Update dist_sets
            for q in selected_list:
                d = dist_matrix[p, q]
                dist_sets[q].discard(d)
            del dist_sets[p]

        # Greedy Initialization
        orders = []
        
        # 1. Random
        ord1 = all_points.copy()
        rng.shuffle(ord1)
        orders.append(ord1)
        
        # 2. Center-out
        center = N // 2
        ord2 = sorted(all_points, key=lambda i: (xs[i]-center)**2 + (ys[i]-center)**2)
        orders.append(ord2)
        
        # 3. Corner-in
        ord3 = sorted(all_points, key=lambda i: min(xs[i], N-1-xs[i]) + min(ys[i], N-1-ys[i]))
        orders.append(ord3)
        
        # 4. Diagonal
        ord4 = sorted(all_points, key=lambda i: abs(xs[i] - ys[i]))
        orders.append(ord4)
        
        # 5. Sparse grid pattern (e.g., step 2)
        ord5 = [i for i in all_points if (i // N) % 2 == 0 and (i % N) % 2 == 0]
        rng.shuffle(ord5)
        # Add rest
        rest = [i for i in all_points if i not in ord5]
        rng.shuffle(rest)
        orders.append(ord5 + rest)

        best_greedy_size = 0
        best_greedy_selected = set()
        best_greedy_dist_sets = {}
        best_greedy_list = []

        for order in orders:
            selected.clear()
            selected_list.clear()
            dist_sets.clear()
            
            for p in order:
                if can_add(p):
                    add(p)
            
            if len(selected) > best_greedy_size:
                best_greedy_size = len(selected)
                best_greedy_selected = set(selected)
                best_greedy_dist_sets = {k: set(v) for k, v in dist_sets.items()}
                best_greedy_list = list(selected_list)

        # Restore best greedy state
        selected = best_greedy_selected
        selected_list = best_greedy_list
        dist_sets = best_greedy_dist_sets
        
        unselected = [p for p in all_points if p not in selected]
        
        best_size = len(selected)
        best_solution = [(i // N, i % N) for i in selected]
        
        no_improve_count = 0
        max_no_improve = 100 # Increased patience
        
        while time.time() < limit:
            improved = False
            
            # 1. Try to add a random unselected point
            if unselected:
                # Pick a random candidate to avoid shuffling the whole list
                idx = rng.randint(0, len(unselected) - 1)
                p = unselected[idx]
                if can_add(p):
                    add(p)
                    unselected.pop(idx)
                    improved = True
            
            if improved:
                no_improve_count = 0
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_solution = [(i // N, i % N) for i in selected]
                continue
            
            # 2. Try Swap (1-out, 1-in)
            if selected_list and unselected:
                # Pick a random point to remove
                p_out_idx = rng.randint(0, len(selected_list) - 1)
                p_out = selected_list[p_out_idx]
                
                found_swap = False
                # Try a few random points to add
                # To be efficient, we can try up to 10 random points
                attempts = min(10, len(unselected))
                for _ in range(attempts):
                    if not unselected:
                        break
                    p_in_idx = rng.randint(0, len(unselected) - 1)
                    p_in = unselected[p_in_idx]
                    
                    # Check if swap is possible
                    remove(p_out)
                    if can_add(p_in):
                        add(p_in)
                        unselected.pop(p_in_idx)
                        unselected.append(p_out)
                        found_swap = True
                        improved = True
                        break
                    else:
                        # Restore p_out
                        # Note: add(p_out) might fail if state changed?
                        # No, we only removed p_out and didn't add anything else yet.
                        # So state is S \ {p_out}. Adding p_out back restores S.
                        add(p_out)
                
                if found_swap:
                    no_improve_count = 0
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_solution = [(i // N, i % N) for i in selected]
                    continue
            
            # 3. Kick (Perturbation)
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                no_improve_count = 0
                # Kick: remove k random points
                k = rng.randint(2, 5)
                if len(selected_list) > k:
                    to_remove_indices = rng.sample(range(len(selected_list)), k)
                    # Sort indices to remove from end to start to maintain index validity
                    to_remove_indices.sort(reverse=True)
                    for idx in to_remove_indices:
                        p = selected_list[idx]
                        remove(p)
                        unselected.append(p)
                    
                    # Refill greedily
                    # Shuffle a subset of unselected to avoid bias
                    # Or just iterate through unselected
                    # To speed up, we can just try to add any point that fits
                    # We do a single pass through unselected
                    # But unselected is large.
                    # Let's pick random points to try to add
                    refill_attempts = min(50, len(unselected))
                    for _ in range(refill_attempts):
                        if not unselected:
                            break
                        p_idx = rng.randint(0, len(unselected) - 1)
                        p = unselected[p_idx]
                        if can_add(p):
                            add(p)
                            unselected.pop(p_idx)

        return best_solution

    best_result = []
    best_size = 0
    
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222, 333]
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - time.time()
        runs_left = len(seeds) - seeds.index(seed)
        # Allocate more time to later seeds? Or equal?
        # Let's give each seed a fair share, but ensure we finish.
        time_per_run = min(15.0, max(5.0, (remaining - 5.0) / runs_left))
        
        if time.time() + time_per_run > deadline - 2:
            time_per_run = max(1.0, deadline - time.time() - 2)
            
        result = solve_once(seed, time_per_run)
        
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
            
    return best_result

# EVOLVE-BLOCK-END
