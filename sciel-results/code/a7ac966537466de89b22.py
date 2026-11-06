"""Improved K2,3-free graph construction for n=20 vertices using ILP."""
import time
import random
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 20
    
    # Precompute edge indices
    edge_indices = []
    edge_to_idx = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_to_idx[(u, v)] = idx
            edge_indices.append((u, v))
            idx += 1
    num_edges = len(edge_indices)
    
    # Variables:
    # 0 .. num_edges-1 : x_uv (binary)
    # num_edges .. num_edges + num_y_vars - 1 : y_uvw (binary)
    
    # y_uvw represents x_uw * x_vw for u < v, w != u, v
    # We need to be careful with the definition.
    # Constraint for pair (u, v): sum_{w != u, v} x_uw * x_vw <= 2
    
    y_vars_info = []
    y_idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(n):
                if w == u or w == v:
                    continue
                # y variable for x_uw and x_vw
                # Note: x_uw is edge (min(u,w), max(u,w))
                # x_vw is edge (min(v,w), max(v,w))
                e_uw = (min(u, w), max(u, w))
                e_vw = (min(v, w), max(v, w))
                y_vars_info.append((u, v, w, edge_to_idx[e_uw], edge_to_idx[e_vw], y_idx + num_edges))
                y_idx += 1
                
    num_y_vars = y_idx
    total_vars = num_edges + num_y_vars
    
    # Objective: Maximize sum x_uv
    c = np.zeros(total_vars)
    c[:num_edges] = 1.0
    
    # Bounds: All variables in [0, 1]
    lb = np.zeros(total_vars)
    ub = np.ones(total_vars)
    bounds = Bounds(lb, ub)
    
    # Integrality: All binary
    integrality = np.ones(total_vars)
    
    # Constraints
    # 1. Linearization constraints for each y_uvw
    # y <= x_uw
    # y <= x_vw
    # y >= x_uw + x_vw - 1
    
    A_rows = []
    lb_rows = []
    ub_rows = []
    
    for (u, v, w, idx_uw, idx_vw, idx_y) in y_vars_info:
        # y - x_uw <= 0  =>  -x_uw + y <= 0
        row = np.zeros(total_vars)
        row[idx_uw] = -1.0
        row[idx_y] = 1.0
        A_rows.append(row)
        lb_rows.append(-np.inf)
        ub_rows.append(0.0)
        
        # y - x_vw <= 0  =>  -x_vw + y <= 0
        row = np.zeros(total_vars)
        row[idx_vw] = -1.0
        row[idx_y] = 1.0
        A_rows.append(row)
        lb_rows.append(-np.inf)
        ub_rows.append(0.0)
        
        # x_uw + x_vw - y <= 1  =>  x_uw + x_vw - y <= 1
        row = np.zeros(total_vars)
        row[idx_uw] = 1.0
        row[idx_vw] = 1.0
        row[idx_y] = -1.0
        A_rows.append(row)
        lb_rows.append(-np.inf)
        ub_rows.append(1.0)
        
    # 2. K2,3 constraints: For each pair (u, v), sum_{w} y_uvw <= 2
    # Group y variables by (u, v)
    pair_to_y_indices = {}
    for (u, v, w, idx_uw, idx_vw, idx_y) in y_vars_info:
        if (u, v) not in pair_to_y_indices:
            pair_to_y_indices[(u, v)] = []
        pair_to_y_indices[(u, v)].append(idx_y)
        
    for (u, v), y_idxs in pair_to_y_indices.items():
        row = np.zeros(total_vars)
        for idx_y in y_idxs:
            row[idx_y] = 1.0
        A_rows.append(row)
        lb_rows.append(-np.inf)
        ub_rows.append(2.0)
        
    if A_rows:
        A = np.array(A_rows)
        lb = np.array(lb_rows)
        ub = np.array(ub_rows)
        constraints = LinearConstraint(A, lb, ub)
    else:
        constraints = LinearConstraint(np.zeros((0, total_vars)), -np.inf, np.inf)
        
    # Solve MILP
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options={"time_limit": 120})
    
    if res.status == 0 and res.x is not None:
        x_vals = res.x[:num_edges]
        edges = []
        for i, (u, v) in enumerate(edge_indices):
            if x_vals[i] > 0.5:
                edges.append([u, v])
        return edges
    else:
        # Fallback to heuristic if ILP fails or times out
        # Re-use a simple greedy from the previous logic or similar
        def is_valid_addition(adj, common, u, v):
            if common[u][v] >= 3:
                return False
            for w in range(n):
                if w == u or w == v:
                    continue
                if adj[v][w] and common[u][w] >= 2:
                    return False
            for w in range(n):
                if w == u or w == v:
                    continue
                if adj[u][w] and common[v][w] >= 2:
                    return False
            return True

        def add_edge(adj, common, u, v):
            adj[u][v] = True
            adj[v][u] = True
            for w in range(n):
                if w == u or w == v:
                    continue
                if adj[v][w]:
                    common[u][w] += 1
                    common[w][u] += 1
                if adj[u][w]:
                    common[v][w] += 1
                    common[w][v] += 1

        def get_edge_list(adj):
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        edges.append([i, j])
            return edges

        def greedy_build(seed):
            rng = random.Random(seed)
            adj = [[False]*n for _ in range(n)]
            common = [[0]*n for _ in range(n)]
            all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            rng.shuffle(all_pairs)
            for u, v in all_pairs:
                if time.time() > deadline:
                    break
                if is_valid_addition(adj, common, u, v):
                    add_edge(adj, common, u, v)
            return adj

        best_adj = None
        best_count = 0
        for seed in range(10):
            if time.time() > deadline:
                break
            adj = greedy_build(seed)
            count = len(get_edge_list(adj))
            if count > best_count:
                best_count = count
                best_adj = adj
                
        if best_adj is not None:
            return get_edge_list(best_adj)
            
        # Final fallback: empty graph or simple structure
        return []

