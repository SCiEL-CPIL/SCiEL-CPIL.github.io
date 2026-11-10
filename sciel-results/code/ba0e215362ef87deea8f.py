"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from collections import Counter

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
    
    # Generate value sets optimized for sum/product collisions
    value_sets = []
    
    # Set 1: Small consecutive integers (many sum collisions)
    value_sets.append(list(range(1, n+1)))
    
    # Set 2: Divisors of 2520 (highly composite - good product collisions)
    value_sets.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24])
    
    # Set 3: Powers of 2 mixed with small numbers
    value_sets.append([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24])
    
    # Set 4: Arithmetic progression with step 2
    value_sets.append(list(range(2, 2*n+1, 2)))
    
    # Set 5: Mixed small values with high multiplicative structure
    value_sets.append([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30])
    
    # Set 6: Numbers with many common factors
    value_sets.append([2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24])
    
    # Set 7: Near-geometric with adjustments
    value_sets.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 20, 24, 27, 32])
    
    # Set 8: All even numbers (products will be multiples of 4)
    value_sets.append([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
    
    # Set 9: Multiples of 3
    value_sets.append([3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42])
    
    # Set 10: Small primes and composites
    value_sets.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15])
    
    # Set 11: Highly structured for product collisions
    value_sets.append([1, 2, 4, 8, 3, 6, 12, 24, 5, 10, 20, 40, 7, 14])
    
    # Set 12: Balanced small range
    value_sets.append([1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 14, 15, 18, 21])
    
    # Add some random but structured sets
    random.seed(42)
    for trial in range(6):
        if time.time() > deadline - 30:
            break
        # Sample from a small range to maximize collisions
        base = random.randint(1, 5)
        step = random.randint(1, 3)
        vs = []
        for i in range(n):
            v = base + i * step + random.randint(-1, 1)
            v = max(1, min(v, maximum))
            if v not in vs:
                vs.append(v)
        while len(vs) < n:
            v = random.randint(1, 30)
            if v not in vs:
                vs.append(v)
        value_sets.append(sorted(vs[:n]))

    # Validate value sets
    valid_value_sets = []
    for vs in value_sets:
        if len(set(vs)) == n and all(isinstance(v, int) and v > 0 and v <= maximum for v in vs):
            valid_value_sets.append(list(vs))

    all_edges = list(combinations(range(n), 2))
    num_all_edges = len(all_edges)
    
    # Map edge index to pair
    edge_to_pair = {idx: e for idx, e in enumerate(all_edges)}

    def fast_greedy_edge_selection(values):
        """Fast greedy: pick edges that maximize collision potential."""
        # Precompute sum and product for each edge
        edge_data = []
        sum_counts = Counter()
        prod_counts = Counter()
        
        for idx, (i, j) in enumerate(all_edges):
            s = values[i] + values[j]
            p = values[i] * values[j]
            edge_data.append((s, p))
            sum_counts[s] += 1
            prod_counts[p] += 1
        
        # Score each edge by combined collision potential
        def edge_score(idx):
            s, p = edge_data[idx]
            return sum_counts[s] * prod_counts[p]
        
        sorted_edges = sorted(range(num_all_edges), key=edge_score, reverse=True)
        
        result = []
        selected = set()
        for idx in sorted_edges:
            if len(result) >= m:
                break
            if idx not in selected:
                selected.add(idx)
                result.append(edge_to_pair[idx])
        
        # Fill remaining if needed
        if len(result) < m:
            for idx in range(num_all_edges):
                if len(result) >= m:
                    break
                if idx not in selected:
                    selected.add(idx)
                    result.append(edge_to_pair[idx])
        
        return result[:m]
    
    def improved_local_search(values, edges, max_iter=1000, time_budget=8.0):
        """Improved local search with precomputed counts for speed."""
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
        
        def compute_counts(edge_list):
            """Compute sum and product counts for a list of edges."""
            sum_counts = Counter()
            prod_counts = Counter()
            for i, j in edge_list:
                s, p = edge_info[(i, j)]
                sum_counts[s] += 1
                prod_counts[p] += 1
            return sum_counts, prod_counts
        
        def get_score_from_counts(sum_counts, prod_counts):
            return max(len(sum_counts), len(prod_counts))
        
        best_edges = current_edges
        best_score = current_score
        no_improve = 0
        current_sum_counts, current_prod_counts = compute_counts(current_edges)
        
        for iteration in range(max_iter):
            if time.time() > start_time + time_budget:
                break
            if no_improve > 80:
                break
            
            improved = False
            
            # Try swapping one edge
            for pos in range(len(current_edges)):
                if time.time() > start_time + time_budget:
                    break
                
                old_edge = current_edges[pos]
                old_key = tuple(sorted(old_edge))
                old_s, old_p = edge_info[old_key]
                
                # Try candidate new edges
                candidates = []
                for idx, new_edge in enumerate(all_edges):
                    new_key = tuple(sorted(new_edge))
                    if new_key == old_key or new_key in edge_set:
                        continue
                    new_s, new_p = edge_info[new_key]
                    
                    # Estimate improvement: how much does this change the counts?
                    # If old_s has count 1, removing it reduces distinct sums by 1
                    # If new_s already exists, adding it doesn't increase distinct sums
                    delta_sums = 0
                    delta_prods = 0
                    
                    if current_sum_counts[old_s] == 1:
                        delta_sums -= 1
                    if current_prod_counts[old_p] == 1:
                        delta_prods -= 1
                    
                    if new_s not in current_sum_counts:
                        delta_sums += 1
                    if new_p not in current_prod_counts:
                        delta_prods += 1
                    
                    # Only consider if it might improve
                    if delta_sums < 0 or delta_prods < 0:
                        candidates.append((delta_sums + delta_prods, idx, new_key, new_s, new_p))
                
                # Sort candidates by estimated improvement
                candidates.sort(key=lambda x: x[0])
                
                for est_delta, idx, new_key, new_s, new_p in candidates[:5]:
                    # Compute exact new score
                    new_sum_counts = Counter(current_sum_counts)
                    new_prod_counts = Counter(current_prod_counts)
                    
                    new_sum_counts[old_s] -= 1
                    if new_sum_counts[old_s] == 0:
                        del new_sum_counts[old_s]
                    new_prod_counts[old_p] -= 1
                    if new_prod_counts[old_p] == 0:
                        del new_prod_counts[old_p]
                    
                    new_sum_counts[new_s] += 1
                    new_prod_counts[new_p] += 1
                    
                    new_score = get_score_from_counts(new_sum_counts, new_prod_counts)
                    
                    if new_score < best_score:
                        # Accept improvement
                        current_edges[pos] = edge_to_pair[idx]
                        edge_set.discard(old_key)
                        edge_set.add(new_key)
                        current_sum_counts = new_sum_counts
                        current_prod_counts = new_prod_counts
                        best_score = new_score
                        best_edges = list(current_edges)
                        improved = True
                        no_improve = 0
                        break
                    elif new_score == best_score and random.random() < 0.15:
                        # Occasionally accept equal to escape local optima
                        current_edges[pos] = edge_to_pair[idx]
                        edge_set.discard(old_key)
                        edge_set.add(new_key)
                        current_sum_counts = new_sum_counts
                        current_prod_counts = new_prod_counts
                        improved = True
                        no_improve = 0
                        break
                
                if improved:
                    break
            
            if not improved:
                no_improve += 1
        
        return best_edges, best_score

    def solve_with_ilp(values, time_limit=6.0):
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
            
            if res.status in [0, 1]:  # Optimal or locally optimal
                x_sol = res.x[:N_x]
                selected_edges = [edge_to_pair[i] for i in range(N_x) if x_sol[i] > 0.5]
                
                if len(selected_edges) == m:
                    score = _raw_value(values, selected_edges)
                    return selected_edges, score
                    
        except Exception:
            pass
            
        return None, float('inf')

    # Phase 1: Quick greedy + improved local search on all value sets
    for idx, vs in enumerate(valid_value_sets):
        if time.time() > deadline - 25:
            break
        
        # Greedy selection
        greedy_edges = fast_greedy_edge_selection(vs)
        greedy_score = _raw_value(vs, greedy_edges)
        
        # Local search refinement with more budget
        if time.time() < deadline - 18:
            polished_edges, polished_score = improved_local_search(vs, greedy_edges, max_iter=500, time_budget=4.0)
            if polished_score < greedy_score:
                greedy_edges = polished_edges
                greedy_score = polished_score
        
        if greedy_score < best_score:
            best_score = greedy_score
            best_values = vs
            best_edges = greedy_edges

    # Phase 2: ILP on top value sets
    scored_vs = []
    for vs in valid_value_sets:
        if time.time() > deadline - 15:
            break
        greedy_edges = fast_greedy_edge_selection(vs)
        score = _raw_value(vs, greedy_edges)
        scored_vs.append((score, vs))
    
    scored_vs.sort(key=lambda x: x[0])
    
    for score, vs in scored_vs[:4]:
        if time.time() > deadline - 12:
            break
        
        edges, ilp_score = solve_with_ilp(vs, time_limit=6.0)
        
        if edges is not None and ilp_score < best_score:
            best_score = ilp_score
            best_values = vs
            best_edges = edges

    # Phase 3: Final local search polish on best result
    if best_edges is not None and time.time() < deadline - 8:
        polished_edges, polished_score = improved_local_search(best_values, best_edges, max_iter=800, time_budget=6.0)
        if polished_score < best_score:
            best_score = polished_score
            best_edges = polished_edges

    # Phase 4: Try a few more value sets with local search if time allows
    if time.time() < deadline - 10:
        for vs in valid_value_sets[:6]:
            if time.time() > deadline - 5:
                break
            greedy_edges = fast_greedy_edge_selection(vs)
            polished_edges, polished_score = improved_local_search(vs, greedy_edges, max_iter=300, time_budget=3.0)
            if polished_score < best_score:
                best_score = polished_score
                best_values = vs
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
