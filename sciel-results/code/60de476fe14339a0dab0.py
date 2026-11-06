"""Rewritten program using MILP to find densest K2,3-free graph on 20 vertices."""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp

def construct():
    n = 20
    # List of all edges
    edges = []
    edge_index = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edges.append([i, j])
            edge_index[(i, j)] = idx
            idx += 1
    
    num_edges = len(edges)
    
    # List of all (i, j, k) triplets where i < j and k != i, j
    # y variables correspond to these
    triplets = []
    triplet_index = {}
    t_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                if k == i or k == j:
                    continue
                triplets.append((i, j, k))
                triplet_index[(i, j, k)] = t_idx
                t_idx += 1
                
    num_triplets = len(triplets)
    num_vars = num_edges + num_triplets
    
    # Objective: Minimize -sum(x)
    c = np.zeros(num_vars)
    for i in range(num_edges):
        c[i] = -1.0
        
    # Constraints
    rows = []
    cols = []
    vals = []
    b_list = []
    
    # 1. y_{i,j,k} <= x_{i,k}
    # y - x_{i,k} <= 0
    for t, (i, j, k) in enumerate(triplets):
        y_var = num_edges + t
        if i < k:
            x_ik_idx = edge_index[(i, k)]
        else:
            x_ik_idx = edge_index[(k, i)]
            
        rows.append(len(b_list))
        cols.append(y_var)
        vals.append(1.0)
        
        rows.append(len(b_list))
        cols.append(x_ik_idx)
        vals.append(-1.0)
        
        b_list.append(0.0)
        
    # 2. y_{i,j,k} <= x_{j,k}
    # y - x_{j,k} <= 0
    for t, (i, j, k) in enumerate(triplets):
        y_var = num_edges + t
        if j < k:
            x_jk_idx = edge_index[(j, k)]
        else:
            x_jk_idx = edge_index[(k, j)]
            
        rows.append(len(b_list))
        cols.append(y_var)
        vals.append(1.0)
        
        rows.append(len(b_list))
        cols.append(x_jk_idx)
        vals.append(-1.0)
        
        b_list.append(0.0)
        
    # 3. x_{i,k} + x_{j,k} - y_{i,j,k} <= 1
    for t, (i, j, k) in enumerate(triplets):
        y_var = num_edges + t
        if i < k:
            x_ik_idx = edge_index[(i, k)]
        else:
            x_ik_idx = edge_index[(k, i)]
            
        if j < k:
            x_jk_idx = edge_index[(j, k)]
        else:
            x_jk_idx = edge_index[(k, j)]
            
        rows.append(len(b_list))
        cols.append(x_ik_idx)
        vals.append(1.0)
        
        rows.append(len(b_list))
        cols.append(x_jk_idx)
        vals.append(1.0)
        
        rows.append(len(b_list))
        cols.append(y_var)
        vals.append(-1.0)
        
        b_list.append(1.0)
        
    # 4. Sum_{k != i, j} y_{i,j,k} <= 2 for each pair (i, j)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                if k == i or k == j:
                    continue
                t = triplet_index[(i, j, k)]
                y_var = num_edges + t
                rows.append(len(b_list))
                cols.append(y_var)
                vals.append(1.0)
            b_list.append(2.0)
            
    A = sp.csr_matrix((vals, (rows, cols)), shape=(len(b_list), num_vars))
    b = np.array(b_list)
    
    constraints = LinearConstraint(A, -np.inf, b)
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(num_vars)
    
    try:
        res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
        if res.success:
            x_vals = res.x[:num_edges]
            final_edges = []
            for i in range(num_edges):
                if x_vals[i] > 0.5:
                    final_edges.append(edges[i])
            return final_edges
    except Exception:
        pass

    # Fallback: 5 disjoint K4s
    fallback_edges = []
    for k in range(5):
        fallback_edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
    return fallback_edges
