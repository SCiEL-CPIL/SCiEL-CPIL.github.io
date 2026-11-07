"""
Combined approach: Simulated Annealing co-optimization with structured initializations.
Uses the structural idea of Simulated Annealing from the inspiration (co-optimizing values and edges)
and the refinement/initialization ideas from the current program (structured value sets).
"""
import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    n = 14
    m = 35
    max_val = 10000
    
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)
    
    def evaluate(values, edges):
        """Fast evaluation of max(distinct sums, distinct products)."""
        sums = set()
        prods = set()
        # Local variables for speed
        v = values
        for u, v_idx in edges:
            sums.add(v[u] + v[v_idx])
            prods.add(v[u] * v[v_idx])
        ns = len(sums)
        np_ = len(prods)
        return max(ns, np_)

    def greedy_select_edges(values):
        """Select m edges to minimize max(distinct sums, distinct products) using greedy."""
        # Precompute sums and products for all pairs for this value set
        pair_data = []
        for u, v in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            pair_data.append((u, v, s, p))
        
        selected = []
        selected_set = set()
        current_sums = set()
        current_prods = set()
        
        for _ in range(m):
            best_edge = None
            best_score_edge = float('inf')
            best_s = None
            best_p = None
            
            for u, v, s, p in pair_data:
                if (u, v) in selected_set:
                    continue
                
                add_s = 1 if s not in current_sums else 0
                add_p = 1 if p not in current_prods else 0
                
                new_num_sums = len(current_sums) + add_s
                new_num_prods = len(current_prods) + add_p
                score = max(new_num_sums, new_num_prods)
                
                # Tie-break: prefer edges that add to both (more "efficient" in filling sets?) 
                # Or prefer edges that add to neither? 
                # Actually, we want to MINIMIZE the max count. 
                # If score is same, prefer one that adds 0 to both (reuses existing).
                tie_break = add_s + add_p
                
                if (score, tie_break) < (best_score_edge, float('inf')):
                    best_score_edge = score
                    best_edge = (u, v)
                    best_s = s
                    best_p = p
            
            if best_edge:
                selected.append(best_edge)
                selected_set.add(best_edge)
                current_sums.add(best_s)
                current_prods.add(best_p)
        
        return selected

    def local_search_polish(values, edges, time_limit):
        """Fast local search to polish the result."""
        current_values = list(values)
        current_edges = list(edges)
        current_score = evaluate(current_values, current_edges)
        
        improved = True
        while improved and (time.time() < time_limit):
            improved = False
            
            # 1. Try swapping edges
            edges_set = set(tuple(e) for e in current_edges)
            for i in range(len(current_edges)):
                if (time.time() > time_limit):
                    break
                u1, v1 = current_edges[i]
                
                # Temporarily remove edge i
                temp_sums = set()
                temp_prods = set()
                for j, (u, v) in enumerate(current_edges):
                    if j != i:
                        temp_sums.add(current_values[u] + current_values[v])
                        temp_prods.add(current_values[u] * current_values[v])
                
                # Try adding each unselected edge
                for u2, v2 in all_edges:
                    if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                        continue
                    if (u2, v2) in edges_set:
                        continue
                    
                    s2 = current_values[u2] + current_values[v2]
                    p2 = current_values[u2] * current_values[v2]
                    
                    # Check if adding this improves
                    # We can do a quick check: if s2 in temp_sums and p2 in temp_prods, score might drop or stay same
                    # But calculating full set is safer.
                    new_sums = temp_sums | {s2}
                    new_prods = temp_prods | {p2}
                    new_score = max(len(new_sums), len(new_prods))
                    
                    if new_score < current_score:
                        current_edges[i] = (u2, v2)
                        edges_set.discard((u1, v1))
                        edges_set.add((u2, v2))
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            # 2. Try changing values
            if (time.time() > time_limit):
                break
                
            for idx in range(n):
                if (time.time() > time_limit):
                    break
                old_val = current_values[idx]
                
                # Generate candidates: nearby values
                candidates = set()
                for delta in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 10, -10, 20, -20, 50, -50]:
                    candidates.add(old_val + delta)
                # Add some random ones
                for _ in range(5):
                    candidates.add(random.randint(1, max_val))
                
                for new_val in candidates:
                    if new_val < 1 or new_val > max_val:
                        continue
                    if new_val in current_values:
                        continue
                    
                    old_v = current_values[idx]
                    current_values[idx] = new_val
                    new_score = evaluate(current_values, current_edges)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        current_values[idx] = old_v
                
                if improved:
                    break
        
        return current_values, current_edges, current_score

    def simulated_annealing(initial_values, time_budget, rng):
        """Simulated Annealing co-optimizing values and edges."""
        deadline = time.time() + time_budget
        
        current_values = list(initial_values)
        current_edges = greedy_select_edges(current_values)
        current_score = evaluate(current_values, current_edges)
        
        best_values = list(current_values)
        best_edges = list(current_edges)
        best_score = current_score
        
        T = 5.0  # Initial temperature
        cooling = 0.998
        
        while time.time() < deadline:
            # Choose perturbation type
            r = rng.random()
            
            if r < 0.4:
                # Perturb a value
                idx = rng.randint(0, n - 1)
                # Mix of local and global moves
                if rng.random() < 0.7:
                    delta = rng.choice([-5, -3, -2, -1, 1, 2, 3, 5, -10, 10, -20, 20, -50, 50])
                    new_val = current_values[idx] + delta
                else:
                    new_val = rng.randint(1, max_val)
                
                if new_val < 1 or new_val > max_val or new_val in current_values:
                    continue
                new_values = current_values.copy()
                new_values[idx] = new_val
                new_edges = current_edges
            elif r < 0.7:
                # Swap two values (permutation)
                i, j = rng.sample(range(n), 2)
                new_values = current_values.copy()
                new_values[i], new_values[j] = new_values[j], new_values[i]
                new_edges = current_edges
            else:
                # Perturb edges: swap one edge
                i = rng.randint(0, len(current_edges) - 1)
                old_u, old_v = current_edges[i]
                # Pick a random unselected edge
                # To speed this up, we can just pick random until we find one not in set, 
                # but that might be slow if graph is dense. 
                # m=35, total edges=91. Density is ~38%. 
                # Random picking is fine.
                attempts = 0
                while attempts < 10:
                    new_u, new_v = rng.choice(all_edges)
                    if (new_u, new_v) != (old_u, old_v) and (new_u, new_v) != (old_v, old_u):
                        # Check if already in current_edges
                        # This check is O(m). 
                        is_in = False
                        for e in current_edges:
                            if e == (new_u, new_v):
                                is_in = True
                                break
                        if not is_in:
                            break
                    attempts += 1
                
                if attempts == 10:
                    continue
                    
                new_values = current_values
                new_edges = list(current_edges)
                new_edges[i] = (new_u, new_v)
            
            new_score = evaluate(new_values, new_edges)
            
            # Accept if better or with probability exp(-(new_score - current_score)/T)
            if new_score <= current_score:
                current_values = new_values
                current_edges = new_edges
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_values = list(new_values)
                    best_edges = list(new_edges)
            else:
                delta = new_score - current_score
                # Avoid log(0) if delta is huge, but exp(-huge) is 0 anyway
                if delta < 100: # Optimization: skip if too bad
                    if rng.random() < math.exp(-delta / T):
                        current_values = new_values
                        current_edges = new_edges
                        current_score = new_score
            
            T *= cooling
        
        return best_values, best_edges, best_score

    # Value strategies from current program + some random ones
    value_strategies = [
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 20, 24, 30, 36],
        [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56],
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], # Powers of 2
        [3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049, 177147, 531441, 1594323, 4782969], # Powers of 3 (some > 10000, will be clipped/replaced in SA)
    ]
    
    # Ensure distinctness and range for initial strategies
    processed_strategies = []
    for vs in value_strategies:
        # Clean up: remove duplicates, out of range
        clean = list(set(vs))
        clean = [x for x in clean if 1 <= x <= max_val]
        if len(clean) >= n:
            # Shuffle to break symmetry
            random.Random(0).shuffle(clean)
            processed_strategies.append(clean[:n])
        else:
            # Pad with random if not enough
            while len(clean) < n:
                clean.append(random.randint(1, max_val))
            processed_strategies.append(clean)
    
    # Add a few random initializations
    for _ in range(3):
        random.Random(100+_).shuffle(processed_strategies[-1]) # Just to mix up
        rand_vals = [random.randint(1, max_val) for _ in range(n)]
        # Ensure distinct
        while len(set(rand_vals)) < n:
            rand_vals[random.randint(0, n-1)] = random.randint(1, max_val)
        processed_strategies.append(rand_vals)
        
    best_overall_score = float('inf')
    best_result = None
    
    # Allocate time to each strategy
    # We have ~80s. 9 strategies. ~8s each?
    # Let's run SA for each strategy with a fixed time budget, and keep the best.
    # Or run them sequentially and stop if we find a very good one?
    # Better: run all, but with a tight time budget per run, then polish the best.
    
    time_per_run = 5.0 # 5s per SA run * 9 runs = 45s. Leaves 40s for polishing.
    
    rng = random.Random(42)
    
    for init_vals in processed_strategies:
        if (time.time() - start_time) > deadline - 10.0:
            break
            
        # Run SA
        v, e, s = simulated_annealing(init_vals, time_per_run, rng)
        
        if s < best_overall_score:
            best_overall_score = s
            best_result = {'values': v, 'edges': e}
            
    # Final Polish on the best result found
    if best_result is not None and (time.time() - start_time) < deadline - 2.0:
        v, e, s = local_search_polish(best_result['values'], best_result['edges'], deadline - 2.0)
        if s < best_overall_score:
            best_overall_score = s
            best_result = {'values': v, 'edges': e}
            
    # Fallback
    if best_result is None:
        best_result = {
            'values': list(range(1, n+1)),
            'edges': [list(e) for e in list(all_edges)[:m]]
        }
    
    # Ensure types are correct
    values = [int(v) for v in best_result['values']]
    edges = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return {'values': values, 'edges': edges}
