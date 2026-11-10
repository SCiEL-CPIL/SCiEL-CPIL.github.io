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
    
    all_edges = list(combinations(range(n), 2))
    num_all_edges = len(all_edges)
    edge_to_pair = {idx: e for idx, e in enumerate(all_edges)}

    def solve_with_ilp(values, time_limit=2.0):
        """
        Use ILP to minimize the number of distinct sums and products.
        """
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
        
        sum_index = {s: i for i, s in enumerate(sums_list)}
        prod_index = {p: i for i, p in enumerate(prods_list)}
        
        N_x = num_all_edges
        N_y = num_sums
        N_z = num_prods
        N_K = 1
        total_vars = N_x + N_y + N_z + N_K
        
        c = np.zeros(total_vars)
        c[-1] = 1.0
        
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
        
        # 4 & 5. Linking constraints
        for e_idx in range(N_x):
            s = edge_sum_map[e_idx]
            p = edge_prod_map[e_idx]
            si = sum_index[s]
            pi = prod_index[p]
            
            row = np.zeros(total_vars)
            row[e_idx] = 1.0
            row[N_x + si] = -1.0
            A_rows.append(row)
            lb_rows.append(-np.inf)
            ub_rows.append(0.0)
            
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
                bounds=bounds,
                options={"time_limit": time_limit}
            )
            
            if res.status in [0, 1]:
                x_sol = res.x[:N_x]
                selected_edges = [edge_to_pair[i] for i in range(N_x) if x_sol[i] > 0.5]
                
                if len(selected_edges) == m:
                    score = _raw_value(values, selected_edges)
                    return selected_edges, score
        except Exception:
            pass
            
        return None, float('inf')

    # Initial value sets
    value_sets = [
        list(range(1, n+1)),
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24],
        list(range(2, 2*n+1, 2)),
        [1, 2, 4, 8, 3, 6, 12, 24, 5, 10, 20, 40, 7, 14]
    ]
    
    # Evaluate initial sets
    for vs in value_sets:
        if time.time() > deadline - 20:
            break
        edges, score = solve_with_ilp(vs)
        if edges is not None and score < best_score:
            best_score = score
            best_values = vs
            best_edges = edges
            
    if best_values is None:
        best_values = list(range(1, n+1))
        edges, score = solve_with_ilp(best_values)
        if edges:
            best_edges = edges
            best_score = score
        else:
            # Fallback greedy
            all_edges_list = list(combinations(range(n), 2))
            random.shuffle(all_edges_list)
            best_edges = all_edges_list[:m]
            best_score = _raw_value(best_values, best_edges)

    # Local search on values
    # We try to improve the value set by changing one element at a time
    current_values = list(best_values)
    current_score = best_score
    
    # Search space: 1..100
    search_range = range(1, 101)
    
    while time.time() < deadline - 10:
        # Pick a random index to change
        i = random.randint(0, n-1)
        # Pick a random new value
        new_v = random.choice(search_range)
        
        # Check distinctness
        if new_v in current_values and current_values[i] != new_v:
            continue
            
        # Create new values
        new_values = current_values[:]
        new_values[i] = new_v
        
        # Evaluate
        edges, score = solve_with_ilp(new_values, time_limit=1.0)
        
        if edges is not None and score < current_score:
            current_values = new_values
            current_score = score
            best_values = new_values
            best_edges = edges
            best_score = score
        elif edges is not None and score == current_score:
            # Accept with some probability to escape local minima? 
            # Or just accept to explore. Let's accept if it's not worse.
            # Actually, just accepting equal scores helps traverse the plateau.
            current_values = new_values
            best_values = new_values
            # Don't update best_edges unless we find a better one, 
            # but we should probably re-run ILP on the final best_values to ensure edges match.
            
    # Final run to ensure edges correspond to best_values
    if time.time() < deadline - 2:
        final_edges, final_score = solve_with_ilp(best_values, time_limit=5.0)
        if final_edges is not None:
            best_edges = final_edges
            best_score = final_score

    # Final validation
    assert len(best_values) == n
    assert len(set(best_values)) == n
    assert all(isinstance(v, int) and v > 0 and v <= maximum for v in best_values)
    assert len(best_edges) == m
    assert all(len(e) == 2 for e in best_edges)
    assert all(e[0] != e[1] for e in best_edges)
    assert len(set(tuple(sorted(e)) for e in best_edges)) == m
    
    return {
        'values': list(best_values),
        'edges': [list(e) for e in best_edges]
    }

# EVOLVE-BLOCK-END
