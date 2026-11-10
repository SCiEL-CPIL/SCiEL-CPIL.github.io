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
    # dist_matrix[i, j] = squared distance between point i and point j
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist_matrix = (dx * dx + dy * dy).astype(np.int32)
    
    def dist2(i, j):
        return int(dist_matrix[i, j])
    
    def solve_once(seed, time_limit):
        start = time.time()
        limit = start + time_limit
        
        rng = random.Random(seed)
        
        # Try multiple greedy strategies
        best_greedy = set()
        best_greedy_size = 0
        best_dist_sets = {}
        
        # Strategy 1: Random
        def run_greedy(order):
            selected = set()
            dist_sets = {}
            selected_list = []
            
            def can_add(p):
                if p in selected:
                    return False
                seen_from_p = set()
                for q in selected_list:
                    d = int(dist_matrix[p, q])
                    if d in seen_from_p:
                        return False
                    if d in dist_sets[q]:
                        return False
                    seen_from_p.add(d)
                return True

            def add(p):
                selected.add(p)
                selected_list.append(p)
                dist_sets[p] = set()
                for q in selected_list[:-1]:
                    d = int(dist_matrix[p, q])
                    dist_sets[p].add(d)
                    dist_sets[q].add(d)

            for p in order:
                if time.time() > limit - 10: # Reserve time for local search
                    break
                if can_add(p):
                    add(p)
            return selected, dist_sets

        # Generate orders
        order_random = all_points.copy()
        rng.shuffle(order_random)
        
        # Spiral order (center out)
        center = N // 2
        points_by_dist = sorted(all_points, key=lambda i: (abs(xs[i]-center)**2 + abs(ys[i]-center)**2))
        order_spiral = points_by_dist
        
        # Corner order
        points_by_corner = sorted(all_points, key=lambda i: (min(xs[i], N-1-xs[i]), min(ys[i], N-1-ys[i])))
        order_corner = points_by_corner
        
        candidates = [
            ("random", order_random),
            ("spiral", order_spiral),
            ("corner", order_corner)
        ]
        
        for name, order in candidates:
            if time.time() > limit - 20:
                break
            sel, ds = run_greedy(order)
            if len(sel) > best_greedy_size:
                best_greedy_size = len(sel)
                best_greedy = sel
                best_dist_sets = ds
        
        selected = best_greedy
        dist_sets = best_dist_sets
        unselected = [p for p in all_points if p not in selected]
        
        if not selected:
            return []
            
        def remove(p):
            if p not in selected:
                return
            selected.discard(p)
            if p not in dist_sets:
                return
            for q in list(selected):
                d = int(dist_matrix[p, q])
                if d in dist_sets[q]:
                    dist_sets[q].discard(d)
            del dist_sets[p]
            
        def add(p):
            selected.add(p)
            dist_sets[p] = set()
            for q in selected:
                if q == p:
                    continue
                d = int(dist_matrix[p, q])
                dist_sets[p].add(d)
                dist_sets[q].add(d)

        def try_swap(p_out, p_in):
            if p_in in selected or p_out not in selected:
                return False
            
            # 1. Remove p_out
            remove(p_out)
            
            # 2. Check if p_in can be added
            # Re-implement can_add logic for p_in against current selected/dist_sets
            seen_from_pin = set()
            valid = True
            for r in selected:
                d = int(dist_matrix[p_in, r])
                if d in seen_from_pin:
                    valid = False
                    break
                if d in dist_sets[r]:
                    valid = False
                    break
                seen_from_pin.add(d)
            
            if valid:
                add(p_in)
                return True
            else:
                # 3. Restore p_out if invalid
                add(p_out)
                return False

        # Local Search
        # Use a mix of random swaps and systematic attempts
        no_improvement_count = 0
        max_no_improvement = 1000
        
        selected_list = list(selected)
        
        while time.time() < limit:
            if no_improvement_count > max_no_improvement:
                break
                
            improved = False
            
            # Random swap
            if unselected and selected_list:
                p_out = rng.choice(selected_list)
                p_in = rng.choice(unselected)
                if try_swap(p_out, p_in):
                    unselected.remove(p_in)
                    unselected.append(p_out)
                    selected_list.remove(p_out)
                    selected_list.append(p_in)
                    improved = True
            
            # Systematic swap: try to replace a random p_out with best p_in
            if not improved and selected_list and unselected:
                p_out = rng.choice(selected_list)
                # Try a few random p_in's
                for _ in range(20):
                    if not unselected:
                        break
                    p_in = rng.choice(unselected)
                    if try_swap(p_out, p_in):
                        unselected.remove(p_in)
                        unselected.append(p_out)
                        selected_list.remove(p_out)
                        selected_list.append(p_in)
                        improved = True
                        break
            
            if improved:
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
        return [(i // N, i % N) for i in selected]

    best_result = []
    best_size = 0
    
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808]
    
    for seed in seeds:
        if time.time() > deadline - 15:
            break
        
        # Estimate time per run
        remaining = deadline - time.time()
        runs_left = len(seeds) - seeds.index(seed)
        time_per_run = min(25.0, max(10.0, remaining / runs_left - 5.0))
        if time.time() + time_per_run > deadline - 5:
            time_per_run = max(5.0, deadline - time.time() - 5)
            
        result = solve_once(seed, time_per_run)
        
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
            
    return best_result

# EVOLVE-BLOCK-END
