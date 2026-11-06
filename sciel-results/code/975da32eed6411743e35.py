import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp
import time
import random

def construct():
    n = 20
    # Indices for x_ij (i < j)
    x_indices = {}
    x_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            x_indices[(i, j)] = x_count
            x_count += 1
    
    # Indices for z_ijk (i < j, k != i, j)
    z_indices = {}
    z_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                if k != i and k != j:
                    z_indices[(i, j, k)] = z_count
                    z_count += 1
    
    total_vars = x_count + z_count
    
    # Objective: Minimize -sum(x_ij)
    c = np.zeros(total_vars)
    for (i, j), idx in x_indices.items():
        c[idx] = -1.0
        
    # Constraints
    # We will use a list of (row, col, value) for sparse matrix
    rows = []
    cols = []
    vals = []
    
    row_idx = 0
    
    # 1. z_ijk <= x_ik
    # 2. z_ijk <= x_jk
    # 3. -z_ijk + x_ik + x_jk <= 1
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                if k == i or k == j:
                    continue
                    
                z_idx = z_indices[(i, j, k)]
                
                # Get x indices for (i, k) and (j, k)
                ik = (min(i, k), max(i, k))
                jk = (min(j, k), max(j, k))
                x_ik_idx = x_indices[ik]
                x_jk_idx = x_indices[jk]
                
                # Constraint 1: z - x_ik <= 0
                rows.append(row_idx); cols.append(z_idx); vals.append(1.0)
                rows.append(row_idx); cols.append(x_ik_idx); vals.append(-1.0)
                
                # Constraint 2: z - x_jk <= 0
                rows.append(row_idx + 1); cols.append(z_idx); vals.append(1.0)
                rows.append(row_idx + 1); cols.append(x_jk_idx); vals.append(-1.0)
                
                # Constraint 3: -z + x_ik + x_jk <= 1
                rows.append(row_idx + 2); cols.append(z_idx); vals.append(-1.0)
                rows.append(row_idx + 2); cols.append(x_ik_idx); vals.append(1.0)
                rows.append(row_idx + 2); cols.append(x_jk_idx); vals.append(1.0)
                
                row_idx += 3
                
    # 4. sum(z_ijk) <= 2 for each pair (i, j)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                if k == i or k == j:
                    continue
                z_idx = z_indices[(i, j, k)]
                rows.append(row_idx); cols.append(z_idx); vals.append(1.0)
            row_idx += 1
            
    A = sp.csc_matrix((vals, (rows, cols)), shape=(row_idx, total_vars))
    
    # Bounds
    lb = np.zeros(total_vars)
    ub = np.ones(total_vars)
    
    # Integrality
    integrality = np.ones(total_vars) # All integer
    
    # Solve
    start_time = time.time()
    try:
        res = opt.milp(c, constraints=A, integrality=integrality, bounds=(lb, ub))
        
        # Check time
        elapsed = time.time() - start_time
        if res.status == 0: # Optimal
            x_vals = res.x[:x_count]
            edges = []
            for (i, j), idx in x_indices.items():
                if x_vals[idx] > 0.5:
                    edges.append([i, j])
            
            # Validate
            if is_valid(n, edges):
                return edges
            else:
                # Fallback to greedy if validation fails (should not happen)
                return greedy_construct(n)
        else:
            # Fallback
            return greedy_construct(n)
            
    except Exception as e:
        return greedy_construct(n)

def is_valid(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    for i in range(n):
        for j in range(i + 1, n):
            if len(adj[i].intersection(adj[j])) > 2:
                return False
    return True

def greedy_construct(n):
    # Simple greedy fallback
    adj = [set() for _ in range(n)]
    edges = []
    
    # Random order
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        # Check if adding (u, v) is valid
        # 1. |N(u) ∩ N(v)| <= 2
        common = adj[u].intersection(adj[v])
        if len(common) > 2:
            continue
            
        # 2. For w in N(v), |N(u) ∩ N(w)| <= 1
        valid = True
        for w in adj[v]:
            if len(adj[u].intersection(adj[w])) > 1:
                valid = False
                break
        if not valid:
            continue
            
        # 3. For w in N(u), |N(v) ∩ N(w)| <= 1
        for w in adj[u]:
            if len(adj[v].intersection(adj[w])) > 1:
                valid = False
                break
        if not valid:
            continue
            
        # Add edge
        adj[u].add(v)
        adj[v].add(u)
        edges.append([u, v])
        
    return edges
