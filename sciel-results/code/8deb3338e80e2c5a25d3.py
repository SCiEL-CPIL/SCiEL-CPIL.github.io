import time
import math
import random

N = 64

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Precompute coordinates
    xs = [i // N for i in range(N * N)]
    ys = [i % N for i in range(N * N)]
    
    all_points = list(range(N * N))
    
    def dist2(i, j):
        dx = xs[i] - xs[j]
        dy = ys[i] - ys[j]
        return dx * dx + dy * dy
    
    def run_greedy(seed):
        """Run greedy construction with given seed, return (selected_set, dist_sets)"""
        selected = set()
        dist_sets = {}
        
        def can_add(p):
            if p in selected:
                return False
            seen_from_p = set()
            for q in selected:
                d = dist2(p, q)
                if d in seen_from_p:
                    return False
                if d in dist_sets[q]:
                    return False
                seen_from_p.add(d)
            return True

        def add(p):
            selected.add(p)
            dist_sets[p] = set()
            for q in selected:
                if q == p:
                    continue
                d = dist2(p, q)
                dist_sets[p].add(d)
                dist_sets[q].add(d)
        
        rng = random.Random(seed)
        shuffled_points = all_points.copy()
        rng.shuffle(shuffled_points)
        
        for p in shuffled_points:
            if time.time() > deadline - 120:
                break
            if can_add(p):
                add(p)
        
        return selected, dist_sets
    
    def run_local_search(selected, dist_sets, seed, time_limit):
        """Run local search on given state, return improved (selected, dist_sets)"""
        unselected = [p for p in all_points if p not in selected]
        rng = random.Random(seed)
        
        def can_add(p):
            if p in selected:
                return False
            seen_from_p = set()
            for q in selected:
                d = dist2(p, q)
                if d in seen_from_p:
                    return False
                if d in dist_sets[q]:
                    return False
                seen_from_p.add(d)
            return True

        def add(p):
            selected.add(p)
            dist_sets[p] = set()
            for q in selected:
                if q == p:
                    continue
                d = dist2(p, q)
                dist_sets[p].add(d)
                dist_sets[q].add(d)

        def remove(p):
            selected.discard(p)
            if p not in dist_sets:
                return
            for q in selected:
                if q == p:
                    continue
                d = dist2(p, q)
                if d in dist_sets[q]:
                    dist_sets[q].discard(d)
            del dist_sets[p]
        
        def try_swap(p_out, p_in):
            if p_in in selected or p_out not in selected:
                return False
            
            seen_from_pin = set()
            for r in selected:
                if r == p_out:
                    continue
                d = dist2(p_in, r)
                if d in seen_from_pin:
                    return False
                seen_from_pin.add(d)
                
            for r in selected:
                if r == p_out:
                    continue
                d_pr = dist2(p_in, r)
                d_pout_r = dist2(p_out, r)
                
                if d_pr in dist_sets[r]:
                    if d_pout_r != d_pr:
                        return False
            
            remove(p_out)
            add(p_in)
            return True
        
        while time.time() < time_limit:
            if rng.random() < 0.4:
                # Try Add
                if not unselected:
                    break
                p = rng.choice(unselected)
                if can_add(p):
                    add(p)
                    unselected.remove(p)
            else:
                # Try Swap
                if not selected or not unselected:
                    continue
                p_out = rng.choice(list(selected))
                p_in = rng.choice(unselected)
                
                if try_swap(p_out, p_in):
                    unselected.remove(p_in)
                    unselected.append(p_out)
        
        return selected, dist_sets
    
    # Multi-start strategy: try several different initializations
    best_size = 0
    best_selected = set()
    best_dist_sets = {}
    
    # Phase 1: Try multiple greedy starts with different seeds
    seeds = [42, 123, 456, 789, 1000, 2024]
    
    for seed in seeds:
        if time.time() > deadline - 90:
            break
        sel, ds = run_greedy(seed)
        if len(sel) > best_size:
            best_size = len(sel)
            best_selected = sel
            best_dist_sets = ds
    
    # Phase 2: Run local search on the best state
    if time.time() < deadline - 30:
        best_selected, best_dist_sets = run_local_search(
            best_selected, best_dist_sets, 42, deadline - 30
        )
    
    # Phase 3: Try a few more local searches with different seeds if time allows
    if time.time() < deadline - 15:
        for extra_seed in [777, 888, 999]:
            if time.time() > deadline - 15:
                break
            sel, ds = run_greedy(extra_seed)
            sel, ds = run_local_search(sel, ds, extra_seed, deadline - 15)
            if len(sel) > best_size:
                best_size = len(sel)
                best_selected = sel
                best_dist_sets = ds
    
    result = [(i // N, i % N) for i in best_selected]
    return result

# EVOLVE-BLOCK-END
