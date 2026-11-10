import time
import math
import random
import numpy as np

N = 64

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
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
        
        # Data structures for the current set
        selected = set()
        selected_list = []
        dist_sets = {} # dist_sets[i] = set of squared distances from i to other selected points
        
        def can_add(p):
            if p in selected:
                return False
            
            # Check if distances from p to all selected points are unique
            # And if any of these distances already exist in the dist_set of the target point
            
            # Optimization: Early exit if possible
            # We need to check against all q in selected_list
            seen_dists = set()
            for q in selected_list:
                d = int(dist_matrix[p, q])
                if d in seen_dists:
                    return False
                if d in dist_sets[q]:
                    return False
                seen_dists.add(d)
            return True

        def add(p):
            selected.add(p)
            selected_list.append(p)
            dist_sets[p] = set()
            for q in selected_list[:-1]:
                d = int(dist_matrix[p, q])
                dist_sets[p].add(d)
                dist_sets[q].add(d)

        def remove(p):
            if p not in selected:
                return
            selected.discard(p)
            selected_list.remove(p) # O(N) but N is small enough (<= 100)
            for q in selected_list:
                d = int(dist_matrix[p, q])
                dist_sets[q].discard(d)
            del dist_sets[p]

        # Greedy Initialization
        # Try different orders
        orders = []
        
        # Random order
        ord1 = all_points.copy()
        rng.shuffle(ord1)
        orders.append(ord1)
        
        # Center-out spiral
        center = N // 2
        ord2 = sorted(all_points, key=lambda i: (xs[i]-center)**2 + (ys[i]-center)**2)
        orders.append(ord2)
        
        # Corner-in
        ord3 = sorted(all_points, key=lambda i: min(xs[i], N-1-xs[i]) + min(ys[i], N-1-ys[i]))
        orders.append(ord3)

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
        
        # Local Search / ILS
        best_size = len(selected)
        best_solution = [(i // N, i % N) for i in selected]
        
        no_improve_count = 0
        max_no_improve = 50 # Reset after this many failed kicks
        
        while time.time() < limit:
            improved = False
            
            # 1. Try to add any unselected point
            # Shuffle unselected to randomize order
            if unselected:
                rng.shuffle(unselected)
                for p in unselected:
                    if can_add(p):
                        add(p)
                        unselected.remove(p)
                        improved = True
                        # Don't break, try to add more if possible
                        # But can_add is expensive, so maybe break after first few?
                        # Let's try to add all possible in one pass? No, adding one changes validity of others.
                        # So we just add one and restart the check?
                        # Actually, if we add p, other points might become invalid? No, adding a point only adds constraints.
                        # It doesn't invalidate existing points. It might invalidate OTHER unselected points.
                        # So we can keep trying to add from unselected.
                        # But we removed p from unselected.
                        # Let's break to re-evaluate the whole set? 
                        # No, just continue the loop.
                        break # Restart the outer while loop to re-shuffle and try again? 
                        # Or just continue the for loop? 
                        # If we continue, we might miss opportunities because the order was fixed.
                        # Let's break and let the outer while loop handle it.
                        # Actually, just adding one is a good step.
            
            if improved:
                no_improve_count = 0
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_solution = [(i // N, i % N) for i in selected]
                continue
            
            # 2. If no improvement, try a Swap (1-out, 1-in)
            if selected_list and unselected:
                # Pick a random point to remove
                p_out = rng.choice(selected_list)
                # Try to find a point to add
                found_swap = False
                # Try a few random candidates
                for _ in range(10):
                    if not unselected:
                        break
                    p_in = rng.choice(unselected)
                    
                    # Check if swap is possible
                    # 1. Remove p_out
                    remove(p_out)
                    # 2. Check if p_in can be added
                    if can_add(p_in):
                        add(p_in)
                        unselected.remove(p_in)
                        unselected.append(p_out)
                        found_swap = True
                        improved = True
                        break
                    else:
                        # Restore p_out
                        add(p_out)
                
                if found_swap:
                    no_improve_count = 0
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_solution = [(i // N, i % N) for i in selected]
                    continue
            
            # 3. If stuck, perform a Kick (Perturbation)
            # Remove k random points and try to refill
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                no_improve_count = 0
                # Kick: remove 2-5 random points
                k = rng.randint(2, 5)
                if len(selected_list) > k:
                    to_remove = rng.sample(selected_list, k)
                    for p in to_remove:
                        remove(p)
                        unselected.append(p)
                    # Try to refill
                    for _ in range(100): # Limited refill attempts
                        if not unselected:
                            break
                        p = rng.choice(unselected)
                        if can_add(p):
                            add(p)
                            unselected.remove(p)
                        else:
                            continue

        return best_solution

    best_result = []
    best_size = 0
    
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222, 333]
    
    for seed in seeds:
        if time.time() > deadline - 15:
            break
        
        remaining = deadline - time.time()
        runs_left = len(seeds) - seeds.index(seed)
        time_per_run = min(20.0, max(5.0, remaining / runs_left - 5.0))
        if time.time() + time_per_run > deadline - 5:
            time_per_run = max(2.0, deadline - time.time() - 5)
            
        result = solve_once(seed, time_per_run)
        
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
            
    return best_result

# EVOLVE-BLOCK-END
