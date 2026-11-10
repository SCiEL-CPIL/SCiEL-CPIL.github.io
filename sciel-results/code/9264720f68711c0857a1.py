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
    # 1. Small integers (good sum collisions)
    value_sets = []
    value_sets.append(list(range(1, n+1)))
    
    # 2. Divisors of 2520 (highly composite, good product collisions)
    # 2520 = 2^3 * 3^2 * 5 * 7
    # Divisors: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18, 20, 21, 24, 28, 30, 35, 36, 40, 42, 45, 56, 60, 63, 70, 72, 84, 90, 105, 120, 126, 140, 168, 180, 210, 252, 280, 315, 360, 420, 504, 630, 840, 1260, 2520
    # We need 14 distinct. Let's pick a mix of small ones for sum collisions and some larger for product variety if needed, but usually small is better for sums.
    # Actually, for product collisions, we want a*b = c*d.
    # Let's try: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24
    value_sets.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24])
    
    # 3. Arithmetic Progression
    value_sets.append(list(range(2, 2*n+1, 2))) # 2, 4, ..., 28
    
    # 4. Mixed: Small primes and composites
    value_sets.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    
    # 5. Geometric-ish
    value_sets.append([1, 2, 4, 8, 3, 6, 12, 24, 5, 10, 20, 40, 7, 14])
    
    # 6. Random but structured
    for _ in range(5):
        if time.time() > deadline - 20:
            break
        # Sample from 1..20 to ensure collisions
        vs = random.sample(range(1, 21), n)
        value_sets.append(vs)

    # Validate value sets
    valid_value_sets = []
    for vs in value_sets:
        if len(set(vs)) == n and all(isinstance(v, int) and v > 0 and v <= maximum for v in vs):
            valid_value_sets.append(vs)

    all_edges = list(combinations(range(n), 2))
    num_all_edges = len(all_edges)
    
    # Map edge index to pair
    edge_to_pair = {idx: e for idx, e in enumerate(all_edges)}

    def solve_with_ilp(values, time_limit=5.0):
        """
        Use ILP to minimize the number of distinct sums and products.
        Variables:
        x_e: 1 if edge e is selected (m edges total)
        y_s: 1 if sum s is used
        z_p: 1 if product p is used
        
        Objective: Minimize max(sum(y_s), sum(z_p))
        Since MILP minimizes linear objectives, we can't directly minimize max(A, B) easily without an extra variable K.
        Minimize K
        s.t.
        sum(y_s) <= K
        sum(z_p) <= K
        sum(x_e) = m
        For each edge e with sum s_e and prod p_e:
           x_e <= y_{s_e}
           x_e <= z_{p_e}
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
        # Total = num_all_edges + num_sums + num_prods + 1
        
        N_x = num_all_edges
        N_y = num_sums
        N_z = num_prods
        N_K = 1
        
        total_vars = N_x + N_y + N_z + N_K
        
        # Objective: Minimize K
        # c vector: [0...0 (N_x), 0...0 (N_y), 0...0 (N_z), 1]
        c = np.zeros(total_vars)
        c[-1] = 1.0
        
        # Constraints:
        # 1. Sum(x) = m
        # 2. Sum(y) <= K  => Sum(y) - K <= 0
        # 3. Sum(z) <= K  => Sum(z) - K <= 0
        # 4. For each edge e: x_e - y_{s_e} <= 0
        # 5. For each edge e: x_e - z_{p_e} <= 0
        
        # We will build A_ub and b_ub for scipy.milp (which handles inequalities A_ub @ x <= b_ub)
        # But milp also takes LinearConstraint. Let's use LinearConstraint for clarity.
        
        # Constraint 1: Sum(x) = m
        # This is an equality. LinearConstraint can handle bounds (lb, ub).
        
        # Let's assemble all constraints into a single LinearConstraint object or list.
        # scipy.milp takes:
        # constraints: LinearConstraint(A, lb, ub)
        # where A is 2D, lb and ub are 1D.
        
        # We'll construct A, lb, ub
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
        # x_e: [0, 1]
        # y_s: [0, 1]
        # z_p: [0, 1]
        # K: [0, inf] (but practically <= N_x)
        
        integrality = np.zeros(total_vars)
        integrality[:N_x] = 1.0 # x is binary
        integrality[N_x:N_x+N_y] = 1.0 # y is binary
        integrality[N_x+N_y:N_x+N_y+N_z] = 1.0 # z is binary
        # K is continuous (or integer, but continuous is fine for minimization of sum of binaries)
        
        bounds = Bounds(lb=np.zeros(total_vars), ub=np.ones(total_vars))
        bounds.ub[-1] = float('inf') # K can be large
        
        try:
            res = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds
            )
            
            if res.status == 0: # Optimal
                x_sol = res.x[:N_x]
                # Find selected edges
                selected_edges = [edge_to_pair[i] for i in range(N_x) if x_sol[i] > 0.5]
                
                if len(selected_edges) == m:
                    score = _raw_value(values, selected_edges)
                    return selected_edges, score
            elif res.status == 1: # Locally optimal
                 # Try to interpret if it's good enough
                 x_sol = res.x[:N_x]
                 selected_edges = [edge_to_pair[i] for i in range(N_x) if x_sol[i] > 0.5]
                 if len(selected_edges) == m:
                     score = _raw_value(values, selected_edges)
                     return selected_edges, score
                     
        except Exception as e:
            pass
            
        return None, float('inf')

    # Evaluate value sets with ILP
    for idx, vs in enumerate(valid_value_sets):
        if time.time() > deadline - 10:
            break
        
        edges, score = solve_with_ilp(vs, time_limit=5.0)
        
        if edges is not None and score < best_score:
            best_score = score
            best_values = vs
            best_edges = edges
            
    # Fallback if ILP fails or finds nothing
    if best_edges is None:
        # Use a simple greedy fallback
        best_values = list(range(1, n+1))
        # Greedy edge selection
        all_edges_list = list(combinations(range(n), 2))
        random.shuffle(all_edges_list)
        best_edges = all_edges_list[:m]
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
        'values': list(best_values),
        'edges': [list(e) for e in best_edges]
    }

# EVOLVE-BLOCK-END
