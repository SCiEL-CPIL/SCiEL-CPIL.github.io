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
        dist_sets = {}  # dist_sets[i] = set of squared distances from i to other selected points
        
        def can_add(p):
            if p in selected:
                return False
            
            # Check if distances from p to all selected points are unique
            # And if any of these distances already exist in the dist_set of the target point
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
            selected_list.remove(p)
            for q in selected_list:
                d = int(dist_matrix[p, q])
                dist_sets[q].discard(d)
            del dist_sets[p]

        # Greedy Initialization
        orders = []
        
        # Random order
        ord1 = all_points.copy()
        rng.shuffle(ord1)
        orders.append(ord1)
        
        # Center-out spiral
        center = N // 2
        ord2 = sorted(all_points, key=lambda i: (xs[i] - center) ** 2 + (ys[i] - center) ** 2)
        orders.append(ord2)
        
        # Corner-in
        ord3 = sorted(all_points, key=lambda i: min(xs[i], N - 1 - xs[i]) + min(ys[i], N - 1 - ys[i]))
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
        
        best_size = len(selected)
        best_solution = [(i // N, i % N) for i in selected]
        
        # Conflict-driven local search
        # For each unselected point, compute which selected points block it
        def get_blockers(p):
            """Return set of selected points that block p from being added"""
            if p in selected:
                return set()
            
            seen_dists = set()
            blockers = set()
            for q in selected_list:
                d = int(dist_matrix[p, q])
                if d in seen_dists:
                    # Distance collision - q blocks p (and all points with same dist to p)
                    blockers.add(q)
                if d in dist_sets[q]:
                    # q already has a point at distance d
                    blockers.add(q)
                seen_dists.add(d)
            return blockers
        
        # Iterative improvement with targeted removal
        T = 1000.0
        T_min = 0.01
        cooling_rate = 0.999
        max_iter = 100000
        no_improve_count = 0
        stagnation_limit = 5000
        
        current_size = len(selected)
        
        for iteration in range(max_iter):
            if time.time() > limit - 1.0:
                break
            
            if T < T_min:
                break
            
            # Choose move type based on temperature
            move_type = rng.random()
            
            if move_type < 0.3:
                # Single add attempt
                if not unselected:
                    continue
                p = rng.choice(unselected)
                if can_add(p):
                    add(p)
                    unselected.remove(p)
                    current_size += 1
                    no_improve_count = 0
                    if current_size > best_size:
                        best_size = current_size
                        best_solution = [(i // N, i % N) for i in selected]
                else:
                    no_improve_count += 1
                    
            elif move_type < 0.6:
                # Targeted removal: find a point that blocks many unselected points
                if not selected_list or not unselected:
                    continue
                
                # Sample a few unselected points and find common blockers
                sample_size = min(10, len(unselected))
                sample = rng.sample(unselected, sample_size)
                
                blocker_counts = {}
                for p in sample:
                    blockers = get_blockers(p)
                    for b in blockers:
                        blocker_counts[b] = blocker_counts.get(b, 0) + 1
                
                if not blocker_counts:
                    no_improve_count += 1
                    continue
                
                # Remove the most-blocked point
                p_out = max(blocker_counts, key=blocker_counts.get)
                remove(p_out)
                current_size -= 1
                unselected.append(p_out)
                
                # Try to add back as many points as possible
                added_count = 0
                for p in sample:
                    if p not in selected and can_add(p):
                        add(p)
                        unselected.remove(p)
                        current_size += 1
                        added_count += 1
                
                if added_count > 1:
                    no_improve_count = 0
                    if current_size > best_size:
                        best_size = current_size
                        best_solution = [(i // N, i % N) for i in selected]
                else:
                    no_improve_count += 1
                    
            elif move_type < 0.85:
                # Pair swap: remove one, try to add two
                if not selected_list or len(unselected) < 2:
                    continue
                
                p_out = rng.choice(selected_list)
                remove(p_out)
                current_size -= 1
                unselected.append(p_out)
                
                # Try to add two points
                added = 0
                for _ in range(2):
                    if not unselected:
                        break
                    p = rng.choice(unselected)
                    if can_add(p):
                        add(p)
                        unselected.remove(p)
                        current_size += 1
                        added += 1
                
                if added >= 2:
                    no_improve_count = 0
                    if current_size > best_size:
                        best_size = current_size
                        best_solution = [(i // N, i % N) for i in selected]
                else:
                    # If we didn't add enough, restore
                    if added < 2:
                        # Try to restore p_out if possible
                        if can_add(p_out):
                            add(p_out)
                            unselected.remove(p_out)
                            current_size += 1
                    no_improve_count += 1
                    
            else:
                # Kick: remove k random points and try to refill aggressively
                k = rng.randint(2, 5)
                if len(selected_list) <= k:
                    continue
                
                removed = []
                for _ in range(k):
                    if not selected_list:
                        break
                    p = rng.choice(selected_list)
                    remove(p)
                    unselected.append(p)
                    removed.append(p)
                    current_size -= 1
                
                # Try to refill with greedy approach on unselected
                refill_count = 0
                for p in unselected[:]:
                    if refill_count >= k:
                        break
                    if can_add(p):
                        add(p)
                        unselected.remove(p)
                        current_size += 1
                        refill_count += 1
                
                if current_size > best_size:
                    best_size = current_size
                    best_solution = [(i // N, i % N) for i in selected]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            
            # Cool down
            T *= cooling_rate
            
            # If stagnating for too long, do a bigger kick
            if no_improve_count > stagnation_limit:
                no_improve_count = 0
                # Remove a larger random chunk
                kick_size = rng.randint(5, 10)
                if len(selected_list) > kick_size:
                    for _ in range(kick_size):
                        if not selected_list:
                            break
                        p = rng.choice(selected_list)
                        remove(p)
                        unselected.append(p)
                        current_size -= 1
                    
                    # Refill greedily
                    for p in unselected[:]:
                        if can_add(p):
                            add(p)
                            unselected.remove(p)
                            current_size += 1
                    
                    if current_size > best_size:
                        best_size = current_size
                        best_solution = [(i // N, i % N) for i in selected]
        
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
