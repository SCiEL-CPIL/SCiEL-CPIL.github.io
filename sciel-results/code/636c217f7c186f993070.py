"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 14
    m = 35
    maximum = 10000
    
    def _raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            v = values[i]
            w = values[j]
            sums.add(v + w)
            prods.add(v * w)
        return max(len(sums), len(prods))
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Generate a diverse pool of value sets with high collision potential
    value_sets = []
    value_sets.append(list(range(1, n+1)))
    
    # Divisors of 2520 (highly composite, good product collisions)
    value_sets.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24])
    
    # Arithmetic Progression
    value_sets.append(list(range(2, 2*n+1, 2)))
    
    # Mixed small values
    value_sets.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    
    # Geometric-ish
    value_sets.append([1, 2, 4, 8, 3, 6, 12, 24, 5, 10, 20, 40, 7, 14])
    
    # Additional structured sets
    value_sets.append([2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24])
    value_sets.append([1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 14, 15, 18, 21])
    value_sets.append([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30])
    
    # Random but structured - sample from small range for collisions
    random.seed(42)
    for _ in range(8):
        if time.time() > deadline - 30:
            break
        vs = random.sample(range(1, 25), n)
        value_sets.append(sorted(vs))

    # Validate value sets
    valid_value_sets = []
    for vs in value_sets:
        if len(set(vs)) == n and all(isinstance(v, int) and v > 0 and v <= maximum for v in vs):
            valid_value_sets.append(list(vs))

    all_edges = list(combinations(range(n), 2))
    num_all_edges = len(all_edges)
    
    # Map edge index to pair
    edge_to_pair = {idx: e for idx, e in enumerate(all_edges)}

    def greedy_edge_selection(values, edges_to_select=m):
        """Greedy selection: pick edges that maximize sum/product collision."""
        selected = set()
        edge_sums = {}
        edge_prods = {}
        sum_counts = {}
        prod_counts = {}
        
        for idx, (i, j) in enumerate(all_edges):
            s = values[i] + values[j]
            p = values[i] * values[j]
            edge_sums[idx] = s
            edge_prods[idx] = p
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        
        # Sort edges by combined collision potential
        def edge_score(idx):
            s = edge_sums[idx]
            p = edge_prods[idx]
            return sum_counts[s] * prod_counts[p]
        
        sorted_edges = sorted(range(num_all_edges), key=edge_score, reverse=True)
        
        result = []
        for idx in sorted_edges:
            if len(result) >= edges_to_select:
                break
            if idx not in selected:
                selected.add(idx)
                result.append(edge_to_pair[idx])
        
        # If we didn't get enough, fill with remaining
        if len(result) < edges_to_select:
            for idx in range(num_all_edges):
                if len(result) >= edges_to_select:
                    break
                if idx not in selected:
                    selected.add(idx)
                    result.append(edge_to_pair[idx])
        
        return result[:edges_to_select]
    
    def local_search(values, edges, max_iter=500, time_budget=5.0):
        """Local search with 1-edge swaps to improve the objective."""
        current_edges = list(edges)
        current_score = _raw_value(values, current_edges)
        edge_set = set(tuple(sorted(e)) for e in current_edges)
        
        # Precompute sum and prod for each edge
        edge_info = {}
        for idx, (i, j) in enumerate(all_edges):
            s = values[i] + values[j]
            p = values[i] * values[j]
            edge_info[(i, j)] = (s, p)
            edge_info[(j, i)] = (s, p)
        
        def get_sums_prods(edge_list):
            sums = set()
            prods = set()
            for i, j in edge_list:
                s, p = edge_info[(i, j)]
                sums.add(s)
                prods.add(p)
            return sums, prods
        
        best_edges = current_edges
        best_score = current_score
        no_improve = 0
        
        for iteration in range(max_iter):
            if time.time() > start_time + time_budget:
                break
            if no_improve > 50:
                break
            
            current_sums, current_prods = get_sums_prods(current_edges)
            improved = False
            
            # Try swapping one edge
            for pos in range(len(current_edges)):
                if time.time() > start_time + time_budget:
                    break
                old_edge = current_edges[pos]
                old_key = tuple(sorted(old_edge))
                
                for idx, new_edge in enumerate(all_edges):
                    new_key = tuple(sorted(new_edge))
                    if new_key == old_key or new_key in edge_set:
                        continue
                    
                    # Compute new sums and prods
                    new_sums = set(current_sums)
                    new_prods = set(current_prods)
                    
                    # Remove old edge contribution
                    old_s, old_p = edge_info[old_key]
                    # Check if old_s is only from this edge
                    only_old_s = True
                    only_old_p = True
                    for e in current_edges:
                        if e == old_edge:
                            continue
                        ek = tuple(sorted(e))
                        es, ep = edge_info[ek]
                        if es == old_s:
                            only_old_s = False
                        if ep == old_p:
                            only_old_p = False
                        if not only_old_s and not only_old_p:
                            break
                    
                    if only_old_s:
                        new_sums.discard(old_s)
                    if only_old_p:
                        new_prods.discard(old_p)
                    
                    # Add new edge contribution
                    new_s, new_p = edge_info[new_key]
                    new_sums.add(new_s)
                    new_prods.add(new_p)
                    
                    new_score = max(len(new_sums), len(new_prods))
                    if new_score < best_score:
                        # Accept improvement
                        current_edges[pos] = new_edge
                        edge_set.discard(old_key)
                        edge_set.add(new_key)
                        current_sums = new_sums
                        current_prods = new_prods
                        best_score = new_score
                        best_edges = list(current_edges)
                        improved = True
                        no_improve = 0
                        break
                    elif new_score == best_score and random.random() < 0.1:
                        # Occasionally accept equal to escape local optima
                        current_edges[pos] = new_edge
                        edge_set.discard(old_key)
                        edge_set.add(new_key)
                        current_sums = new_sums
                        current_prods = new_prods
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
            
            if not improved:
                no_improve += 1
        
        return best_edges, best_score

    def solve_with_ilp(values, time_limit=8.0):
        """
        Use ILP to minimize the number of distinct sums and products.
        """
        # Identify unique sums and products for these values
        unique_sums = set()
        unique_prods = set()
        edge_sum_map = {}
        edge_prod_map = {}
        
        for idx, (i, j) in enumerate(all_edges):
            s = values[i] + values[j]
            p = values[i] * values[j]
            unique_sums.add(s)
            unique_prods.add(p)
            edge_sum_map[idx] = s
            edge_prod_map[idx] = p
        
        sums_list = sorted(list(unique_sums))
        prods_list = sorted(list(unique_prods))
        
        num_sums = len(sums_list)
        num_prods = len(prods_list)
        
        # Index mappings
        sum_index = {s: i for i, s in enumerate(sums_list)}
        prod_index = {p: i for i, p in enumerate(prods_list)}
        
        # Total variables:
        # x: num_all_edges
        # y: num_sums
        # z: num_prods
        # K: 1
        N_x = num_all_edges
        N_y = num_sums
        N_z = num_prods
        N_K = 1
        
        total_vars = N_x + N_y + N_z + N_K
        
        # Objective: Minimize K
        c = np.zeros(total_vars)
        c[-1] = 1.0
        
        # Constraints
        A_rows = []
        lb_rows = []
        ub_rows = []
        
        # 1. Sum(x) = m
        row = np.zeros(total_vars)
        row[:N_x] = 1.0
        A_rows.append(row)
        lb_rows.append(m)
        ub_rows.append(m)
        
        # 2. Sum(y) - K <= 0
        row = np.zeros(total_vars)
        row[N_x:N_x+N_y] = 1.0
        row[-1] = -1.0
        A_rows.append(row)
        lb_rows.append(-np.inf)
        ub_rows.append(0.0)
        
        # 3. Sum(z) - K <= 0
        row = np.zeros(total_vars)
        row[N_x+N_y:N_x+N_y+N_z] = 1.0
        row[-1] = -1.0
        A_rows.append(row)
        lb_rows.append(-np.inf)
        ub_rows.append(0.0)
        
        # 4 & 5. Linking constraints for each edge
        for e_idx in range(N_x):
            s = edge_sum_map[e_idx]
            p = edge_prod_map[e_idx]
            si = sum_index[s]
            pi = prod_index[p]
            
            # x_e - y_{s_e} <= 0
            row = np.zeros(total_vars)
            row[e_idx] = 1.0
            row[N_x + si] = -1.0
            A_rows.append(row)
            lb_rows.append(-np.inf)
            ub_rows.append(0.0)
            
            # x_e - z_{p_e} <= 0
            row = np.zeros(total_vars)
            row[e_idx] = 1.0
            row[N_x + N_y + pi] = -1.0
            A_rows.append(row)
            lb_rows.append(-np.inf)
            ub_rows.append(0.0)
        
        A = np.array(A_rows)
        lb = np.array(lb_rows)
        ub = np.array(ub_rows)
        
        constraints = LinearConstraint(A, lb, ub)
        
        # Bounds for variables
        integrality = np.zeros(total_vars)
        integrality[:N_x] = 1.0
        integrality[N_x:N_x+N_y] = 1.0
        integrality[N_x+N_y:N_x+N_y+N_z] = 1.0
        
        bounds = Bounds(lb=np.zeros(total_vars), ub=np.ones(total_vars))
        bounds.ub[-1] = float('inf')
        
        try:
            res = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds
            )
            
            if res.status == 0:  # Optimal
                x_sol = res.x[:N_x]
                selected_edges = [edge_to_pair[i] for i in range(N_x) if x_sol[i] > 0.5]
                
                if len(selected_edges) == m:
                    score = _raw_value(values, selected_edges)
                    return selected_edges, score
            elif res.status == 1:  # Locally optimal
                x_sol = res.x[:N_x]
                selected_edges = [edge_to_pair[i] for i in range(N_x) if x_sol[i] > 0.5]
                if len(selected_edges) == m:
                    score = _raw_value(values, selected_edges)
                    return selected_edges, score
                    
        except Exception as e:
            pass
            
        return None, float('inf')

    # Phase 1: Quick greedy + local search on all value sets
    for idx, vs in enumerate(valid_value_sets):
        if time.time() > deadline - 20:
            break
        
        # Greedy selection
        greedy_edges = greedy_edge_selection(vs)
        greedy_score = _raw_value(vs, greedy_edges)
        
        # Local search refinement
        if time.time() < deadline - 15:
            polished_edges, polished_score = local_search(vs, greedy_edges, max_iter=200, time_budget=3.0)
            if polished_score < greedy_score:
                greedy_edges = polished_edges
                greedy_score = polished_score
        
        if greedy_score < best_score:
            best_score = greedy_score
            best_values = vs
            best_edges = greedy_edges

    # Phase 2: ILP on top value sets
    # Sort value sets by their greedy score
    scored_vs = []
    for vs in valid_value_sets:
        if time.time() > deadline - 15:
            break
        greedy_edges = greedy_edge_selection(vs)
        score = _raw_value(vs, greedy_edges)
        scored_vs.append((score, vs))
    
    scored_vs.sort(key=lambda x: x[0])
    
    for score, vs in scored_vs[:5]:
        if time.time() > deadline - 10:
            break
        
        edges, ilp_score = solve_with_ilp(vs, time_limit=8.0)
        
        if edges is not None and ilp_score < best_score:
            best_score = ilp_score
            best_values = vs
            best_edges = edges
        elif edges is None and score < best_score:
            # ILP failed, use greedy result
            best_score = score
            best_values = vs
            best_edges = greedy_edge_selection(vs)

    # Phase 3: Final local search polish on best result
    if best_edges is not None and time.time() < deadline - 5:
        polished_edges, polished_score = local_search(best_values, best_edges, max_iter=300, time_budget=5.0)
        if polished_score < best_score:
            best_score = polished_score
            best_edges = polished_edges

    # Fallback
    if best_edges is None:
        best_values = list(range(1, n+1))
        best_edges = all_edges[:m]
        best_score = _raw_value(best_values, best_edges)

    # Final validation
    assert len(best_values) == n
    assert len(set(best_values)) == n
    assert all(isinstance(v, int) and v > 0 and v <= maximum for v in best_values)
    assert len(best_edges) == m
    assert all(len(e) == 2 for e in best_edges)
    assert all(e[0] != e[1] for e in best_edges)
    assert len(set(tuple(sorted(e)) for e in best_edges)) == m
    
    return {
        'values': [int(v) for v in best_values],
        'edges': [[int(i), int(j)] for i, j in best_edges]
    }

# EVOLVE-BLOCK-END
