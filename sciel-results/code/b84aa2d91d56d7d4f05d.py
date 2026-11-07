import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin
    
    n = 14
    m = 35
    max_val = 10000
    
    # Precompute all possible edges
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    def evaluate_detailed(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return len(sums), len(prods), max(len(sums), len(prods))
    
    def greedy_select_edges(values):
        """Select m edges to minimize max(distinct sums, distinct products) using greedy."""
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
        
        return selected
    
    def local_improve_edges(values, edges):
        """Try swapping one edge for another to reduce distinct counts."""
        selected = list(edges)
        selected_set = set(selected)
        
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
                
                # Remove edge i temporarily
                temp_sums = set()
                temp_prods = set()
                for j, (u, v) in enumerate(selected):
                    if j != i:
                        temp_sums.add(values[u] + values[v])
                        temp_prods.add(values[u] * values[v])
                
                # Try adding each unselected edge
                for u2, v2 in all_edges:
                    if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                        continue
                    if (u2, v2) in selected_set or (v2, u2) in selected_set:
                        continue
                    
                    s2 = values[u2] + values[v2]
                    p2 = values[u2] * values[v2]
                    
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
    
    def simulated_annealing(values, edges, time_budget):
        """Simulated annealing that co-optimizes values and edges."""
        random.seed(42)
        deadline = time.time() + time_budget
        
        current_values = list(values)
        current_edges = list(edges)
        _, _, current_score = evaluate_detailed(current_values, current_edges)
        
        best_values = list(current_values)
        best_edges = list(current_edges)
        best_score = current_score
        
        T = 5.0  # Initial temperature
        cooling = 0.995
        
        while time.time() < deadline:
            # Choose perturbation type
            r = random.random()
            
            if r < 0.4:
                # Perturb a value
                idx = random.randint(0, n - 1)
                delta = random.choice([-5, -3, -2, -1, 1, 2, 3, 5, -10, 10, -20, 20])
                new_val = current_values[idx] + delta
                if new_val < 1 or new_val > max_val or new_val in current_values:
                    continue
                new_values = current_values.copy()
                new_values[idx] = new_val
                new_edges = current_edges
            elif r < 0.7:
                # Swap two values
                i, j = random.sample(range(n), 2)
                new_values = current_values.copy()
                new_values[i], new_values[j] = new_values[j], new_values[i]
                new_edges = current_edges
            else:
                # Perturb edges: swap one edge
                i = random.randint(0, len(current_edges) - 1)
                old_u, old_v = current_edges[i]
                # Pick a random unselected edge
                candidates = [e for e in all_edges if e != (old_u, old_v) and e != (old_v, old_u) and e not in current_edges and tuple(reversed(e)) not in current_edges]
                if not candidates:
                    continue
                new_u, new_v = random.choice(candidates)
                new_values = current_values
                new_edges = list(current_edges)
                new_edges[i] = (new_u, new_v)
            
            _, _, new_score = evaluate_detailed(new_values, new_edges)
            
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
                if random.random() < math.exp(-delta / T):
                    current_values = new_values
                    current_edges = new_edges
                    current_score = new_score
            
            T *= cooling
            if T < 0.01:
                T = 0.01
        
        return best_values, best_edges, best_score
    
    # Generate diverse initial value configurations
    candidate_value_sets = []
    
    # Consecutive integers
    for start in [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]:
        candidate_value_sets.append(list(range(start, start + n)))
    
    # Multiples of small numbers
    for base in [2, 3, 4, 5, 6, 7, 8]:
        vals = []
        v = base
        while len(vals) < n:
            if v <= max_val and v not in vals:
                vals.append(v)
            v += base
        if len(vals) == n:
            candidate_value_sets.append(vals)
    
    # Geometric-like sequences
    for ratio in [1.2, 1.3, 1.5]:
        vals = [1]
        while len(vals) < n:
            next_val = int(round(vals[-1] * ratio))
            if next_val > max_val or next_val in vals:
                next_val = vals[-1] + 1
            if next_val <= max_val and next_val not in vals:
                vals.append(next_val)
            else:
                break
        if len(vals) == n:
            candidate_value_sets.append(vals)
    
    # Mixed clusters
    candidate_value_sets.append(list(range(1, 8)) + list(range(10, 17)))
    candidate_value_sets.append(list(range(1, 6)) + list(range(8, 15)))
    candidate_value_sets.append(list(range(2, 9)) + list(range(12, 19)))
    candidate_value_sets.append(list(range(1, 5)) + list(range(7, 12)) + list(range(14, 19)))
    
    # Random small values
    random.seed(123)
    for _ in range(5):
        vals = random.sample(range(1, 60), n)
        candidate_value_sets.append(sorted(vals))
    
    # Remove duplicates and ensure validity
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
    
    # Phase 1: Greedy edge selection for each candidate
    for vals in unique_candidates:
        if time.time() - start_time > timeout - 15:
            break
        
        edges = greedy_select_edges(vals)
        edges = local_improve_edges(vals, edges)
        score = evaluate(vals, edges)
        
        if score < best_score:
            best_score = score
            best_values = vals
            best_edges = edges
            
            if best_score <= 12:
                break
    
    # Phase 2: Simulated annealing with co-optimization
    if best_values is None:
        best_values = list(range(1, n + 1))
        best_edges = greedy_select_edges(best_values)
        best_score = evaluate(best_values, best_edges)
    
    remaining_time = timeout - (time.time() - start_time) - 3
    if remaining_time > 5:
        # Run SA on the best solution found
        sa_values, sa_edges, sa_score = simulated_annealing(best_values, best_edges, remaining_time * 0.6)
        if sa_score < best_score:
            best_score = sa_score
            best_values = sa_values
            best_edges = sa_edges
        
        # Try SA on a few other promising candidates
        remaining_time = timeout - (time.time() - start_time) - 2
        if remaining_time > 5:
            for vals in unique_candidates[:5]:
                if time.time() - start_time > timeout - 3:
                    break
                if vals == best_values:
                    continue
                edges = greedy_select_edges(vals)
                sa_v, sa_e, sa_s = simulated_annealing(vals, edges, min(remaining_time * 0.3, 8))
                if sa_s < best_score:
                    best_score = sa_s
                    best_values = sa_v
                    best_edges = sa_e
    
    # Final local improvement
    if time.time() - start_time < timeout - 2:
        best_edges = local_improve_edges(best_values, best_edges)
        best_score = evaluate(best_values, best_edges)
    
    # Ensure types are correct
    return {
        'values': [int(v) for v in best_values],
        'edges': [list(e) for e in best_edges]
    }
