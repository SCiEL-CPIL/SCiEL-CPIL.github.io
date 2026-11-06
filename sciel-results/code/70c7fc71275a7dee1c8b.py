import numpy as np
import time
import sys

def construct():
    n = 20
    # Indices for edges
    # We need to map (i, j) to an index in the variable array
    edge_indices = {}
    edge_list = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_indices[(i, j)] = idx
            edge_list.append([i, j])
            idx += 1
    num_edges = len(edge_list)
    
    # Auxiliary variables y_{u, v, w}
    # u < v, w != u, w != v
    aux_indices = {}
    aux_list = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(n):
                if w != u and w != v:
                    aux_indices[(u, v, w)] = idx
                    aux_list.append((u, v, w))
                    idx += 1
    num_aux = len(aux_list)
    
    total_vars = num_edges + num_aux
    
    # Objective: Maximize sum of x_ij
    # milp minimizes c @ x, so we minimize -sum(x)
    c = np.zeros(total_vars)
    for i in range(num_edges):
        c[i] = -1.0
    
    # Bounds: All binary [0, 1]
    integrality = np.ones(total_vars)
    bounds = [(0, 1)] * total_vars
    
    # Constraints
    # A_ub @ x <= b_ub
    
    # 1. Linearization: y_{uvw} <= x_{uw}
    # 2. Linearization: y_{uvw} <= x_{vw}
    # 3. Linearization: y_{uvw} >= x_{uw} + x_{vw} - 1  => -y_{uvw} + x_{uw} + x_{vw} >= 1 => y_{uvw} - x_{uw} - x_{vw} <= -1
    # 4. Codegree: sum_w y_{uvw} <= 2
    
    # We can build A_ub as a sparse matrix or dense.
    # 10k constraints, 3.6k vars. Dense is 36M floats, ~288MB. Might be tight but okay.
    # Let's use scipy.sparse to be safe and fast.
    
    from scipy.sparse import lil_matrix, csr_matrix
    
    A_ub = lil_matrix((num_aux * 3 + num_edges, total_vars), dtype=float)
    b_ub = np.zeros(num_aux * 3 + num_edges)
    
    row = 0
    # Constraints 1-3 for each aux var
    for (u, v, w), aux_idx in aux_indices.items():
        # Get edge indices for (u,w) and (v,w)
        # Note: edge index expects i < j
        i1, j1 = sorted((u, w))
        i2, j2 = sorted((v, w))
        e_idx_1 = edge_indices[(i1, j1)]
        e_idx_2 = edge_indices[(i2, j2)]
        
        # 1. y <= x_uw  =>  y - x_uw <= 0
        A_ub[row, aux_idx] = 1.0
        A_ub[row, e_idx_1] = -1.0
        b_ub[row] = 0.0
        row += 1
        
        # 2. y <= x_vw  =>  y - x_vw <= 0
        A_ub[row, aux_idx] = 1.0
        A_ub[row, e_idx_2] = -1.0
        b_ub[row] = 0.0
        row += 1
        
        # 3. y >= x_uw + x_vw - 1  =>  y - x_uw - x_vw <= -1
        A_ub[row, aux_idx] = 1.0
        A_ub[row, e_idx_1] = -1.0
        A_ub[row, e_idx_2] = -1.0
        b_ub[row] = -1.0
        row += 1
        
    # Constraints 4: For each pair (u,v), sum_w y_{uvw} <= 2
    # We can group aux vars by (u,v)
    from collections import defaultdict
    pairs_to_aux = defaultdict(list)
    for (u, v, w), aux_idx in aux_indices.items():
        pairs_to_aux[(u, v)].append(aux_idx)
        
    for (u, v), aux_idxs in pairs_to_aux.items():
        for idx in aux_idxs:
            A_ub[row, idx] = 1.0
        b_ub[row] = 2.0
        row += 1
        
    # Ensure dimensions match
    A_ub = A_ub.tocsr()
    
    # Solve
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        # milp expects c, constraints, integrality, bounds
        # We have A_ub @ x <= b_ub
        cons = LinearConstraint(A_ub, ub=b_ub)
        bounds_obj = Bounds(0, 1)
        
        res = milp(c, constraints=cons, integrality=integrality, bounds=bounds_obj)
        
        if res.status == 0:
            x_vals = res.x
            edges = []
            for i in range(num_edges):
                if x_vals[i] > 0.5:
                    edges.append(edge_list[i])
            return edges
        else:
            # Fallback if solver fails: return a random valid graph or empty
            # For safety, return a small valid graph
            return [[0, 1]]
    except Exception as e:
        # Fallback to a simple heuristic if ILP fails or is not available
        # Simple greedy build-up
        n = 20
        adj = [[0]*n for _ in range(n)]
        edges = []
        import random
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            # Check codegree
            common = sum(1 for z in range(n) if adj[u][z] and adj[v][z])
            if common <= 2:
                # Also need to check if adding (u,v) creates new codegree violations for neighbors
                # This greedy check is complex, so we do a full check
                valid = True
                # Check all pairs (a, b) where a or b is u or v
                # Actually, just check if adding (u,v) makes any pair have codegree > 2
                # Pairs affected: (u, x) for x in N(v), (v, y) for y in N(u), and (u, v)
                # We already checked (u,v).
                # For x in N(v): new common neighbor is u. So codeg(u, x) increases by 1.
                # We need current codeg(u, x) <= 1.
                for x in range(n):
                    if x == u or x == v: continue
                    if adj[v][x]:
                        cnt = sum(1 for z in range(n) if z != u and z != x and adj[u][z] and adj[x][z])
                        if cnt > 1:
                            valid = False
                            break
                if valid:
                    for y in range(n):
                        if y == u or y == v: continue
                        if adj[u][y]:
                            cnt = sum(1 for z in range(n) if z != v and z != y and adj[v][z] and adj[y][z])
                            if cnt > 1:
                                valid = False
                                break
                if valid:
                    adj[u][v] = 1
                    adj[v][u] = 1
                    edges.append([u, v])
        return edges
