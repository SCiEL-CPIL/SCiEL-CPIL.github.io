import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin
    
    n = 14
    m = 35
    max_val = 10000
    
    # Precompute all possible edges
    all_edges = list(combinations(range(n), 2))
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    def greedy_select_edges(values):
        """Select m edges to minimize max(distinct sums, distinct products) using greedy + local improvement."""
        # Precompute sums and products for all pairs
        pair_data = []
        for u, v in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            pair_data.append((u, v, s, p))
        
        # Greedy selection: pick edge that minimizes new max(distinct_sums, distinct_prods)
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
                if (u, v) in selected_set or (v, u) in selected_set:
                    continue
                
                add_s = 1 if s not in current_sums else 0
                add_p = 1 if p not in current_prods else 0
                
                new_num_sums = len(current_sums) + add_s
                new_num_prods = len(current_prods) + add_p
                score = max(new_num_sums, new_num_prods)
                
                # Tie-break: prefer edges that add to both (more "efficient")
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
        
        # Local improvement: try swapping one edge for another to reduce distinct counts
        improved = True
        while improved:
            improved = False
            current_sums = set()
            current_prods = set()
            for u, v in selected:
                current_sums.add(values[u] + values[v])
                current_prods.add(values[u] * values[v])
            current_score = max(len(current_sums), len(current_prods))
            
            for i in range(len(selected)):
                u1, v1 = selected[i]
                s1 = values[u1] + values[v1]
                p1 = values[u1] * values[v1]
                
                # Remove edge i temporarily
                temp_sums = set()
                temp_prods = set()
                for j, (u, v) in enumerate(selected):
                    if j != i:
                        temp_sums.add(values[u] + values[v])
                        temp_prods.add(values[u] * values[v])
                
                # Try adding each unselected edge
                for u2, v2, s2, p2 in pair_data:
                    if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                        continue
                    if (u2, v2) in selected_set or (v2, u2) in selected_set:
                        continue
                    
                    # Check if this edge is already in selected (at any position)
                    already_in = False
                    for (u, v) in selected:
                        if (u == u2 and v == v2) or (u == v2 and v == u2):
                            already_in = True
                            break
                    if already_in:
                        continue
                    
                    new_sums = temp_sums | {s2}
                    new_prods = temp_prods | {p2}
                    new_score = max(len(new_sums), len(new_prods))
                    
                    if new_score < current_score:
                        selected[i] = (u2, v2)
                        selected_set.discard((u1, v1))
                        selected_set.discard((v1, u1))
                        selected_set.add((u2, v2))
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
        
        return selected
    
    # Try multiple value configurations
    candidate_value_sets = []
    
    # Consecutive integers starting at different offsets
    for start in [1, 2, 3, 5, 8, 10, 15, 20]:
        candidate_value_sets.append(list(range(start, start + n)))
    
    # Multiples of small numbers (to increase product collisions)
    for base in [2, 3, 4, 5, 6]:
        vals = []
        v = base
        while len(vals) < n:
            if v <= max_val and v not in vals:
                vals.append(v)
            v += base
        if len(vals) == n:
            candidate_value_sets.append(vals)
    
    # Mixed: some small, some clustered
    candidate_value_sets.append(list(range(1, 8)) + list(range(10, 17)))
    candidate_value_sets.append(list(range(1, 6)) + list(range(8, 15)))
    candidate_value_sets.append(list(range(2, 9)) + list(range(12, 19)))
    
    # Remove duplicates and ensure all values are distinct positive integers <= 10000
    unique_candidates = []
    seen = set()
    for vals in candidate_value_sets:
        if len(set(vals)) != n:
            continue
        if any(v < 1 or v > max_val for v in vals):
            continue
        key = tuple(sorted(vals))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(vals)
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    for vals in unique_candidates:
        if time.time() - start_time > timeout - 5:
            break
        
        edges = greedy_select_edges(vals)
        score = evaluate(vals, edges)
        
        if score < best_score:
            best_score = score
            best_values = vals
            best_edges = edges
            
            # Early exit if we hit a very good score
            if best_score <= 12:
                break
    
    # If we haven't found anything, fall back to 1..14
    if best_values is None:
        best_values = list(range(1, n + 1))
        best_edges = greedy_select_edges(best_values)
        best_score = evaluate(best_values, best_edges)
    
    # Final polish: try small value perturbations on the best solution
    random.seed(42)
    polish_deadline = time.time() + (timeout - (time.time() - start_time) - 2)
    
    while time.time() < polish_deadline:
        idx = random.randint(0, n - 1)
        old_val = best_values[idx]
        
        # Try nearby values
        for delta in [-3, -2, -1, 1, 2, 3, -5, 5, -10, 10]:
            new_val = old_val + delta
            if new_val < 1 or new_val > max_val:
                continue
            if new_val in best_values:
                continue
            
            new_values = best_values.copy()
            new_values[idx] = new_val
            
            new_edges = greedy_select_edges(new_values)
            new_score = evaluate(new_values, new_edges)
            
            if new_score < best_score:
                best_score = new_score
                best_values = new_values
                best_edges = new_edges
                break
        else:
            # No improvement from nearby, try a random value
            new_val = random.randint(1, 50)
            if new_val == old_val or new_val in best_values:
                continue
            
            new_values = best_values.copy()
            new_values[idx] = new_val
            new_edges = greedy_select_edges(new_values)
            new_score = evaluate(new_values, new_edges)
            
            if new_score < best_score:
                best_score = new_score
                best_values = new_values
                best_edges = new_edges
    
    # Ensure types are correct
    return {
        'values': [int(v) for v in best_values],
        'edges': [list(e) for e in best_edges]
    }
