import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    """
    Finds a K2,3-free graph on 20 vertices maximizing edges using Integer Linear Programming.
    
    Strategy:
    - Formulate as ILP: maximize sum of edge variables
    - Constraints: for each pair (u,v), the number of common neighbors <= 2
    - Use auxiliary variables x[w,u,v] = 1 iff w is a common neighbor of u and v
    - scipy.optimize.milp uses HiGHS solver which handles this size efficiently
    """
    n = 20
    start_time = time.time()
    time_limit = 120.0
    safety_margin = 15.0  # Return at least 15s before timeout
    
    # Edge variables: x[i] for edge (i//n, i%n) where i//n < i%n
    # Total possible edges: n*(n-1)//2 = 190
    num_edges = n * (n - 1) // 2
    
    # Build edge index mapping
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_to_idx[(u, v)] = idx
            idx_to_edge.append((u, v))
            idx += 1
    
    # For each pair (u, v), we need to constrain common neighbors <= 2
    # Common neighbors of u and v: all w != u, v such that (u,w) and (v,w) are edges
    # Introduce auxiliary variable y[w, u, v] for w < u < v or similar ordering
    # Actually, let's use: for each pair (u,v) with u<v, and each w not in {u,v},
    #   y_{w,u,v} <= x_{u,w}, y_{w,u,v} <= x_{v,w}, y_{w,u,v} >= x_{u,w} + x_{v,w} - 1
    #   sum_w y_{w,u,v} <= 2
    
    # This creates too many variables. Let's think differently.
    # Alternative: For each pair (u,v), the common neighbors are determined by the edges.
    # We can use a direct approach with the constraint that for any three vertices u,v,w,
    # if u~v, u~w, v~w then... no, that's triangles.
    
    # Actually, the cleanest ILP formulation:
    # Variables: x_e for each edge e (binary)
    # For each pair (u,v), u<v: sum over w != u,v of [x_{u,w} * x_{v,w}] <= 2
    # The product x_{u,w} * x_{v,w} needs linearization with auxiliary vars.
    
    # Number of (pair, common_neighbor) triples: C(20,2) * 18 = 190 * 18 = 3420
    # This is manageable.
    
    # Let's build the full ILP
    # Total binary variables: 190 (edges) + 3420 (auxiliary) = 3610
    
    # Auxiliary variable index: for pair (u,v) with u<v, and neighbor w (w != u, w != v)
    # Index: pair_idx * 18 + w_idx where w_idx is the position of w in the list of vertices excluding u,v
    
    # Actually, let me simplify. For each pair (u,v), there are n-2 = 18 possible common neighbors.
    # aux_idx = pair_idx * (n-2) + local_w_idx
    
    num_pairs = num_edges  # C(20,2) = 190
    num_aux_per_pair = n - 2  # 18
    num_aux = num_pairs * num_aux_per_pair  # 3420
    total_vars = num_edges + num_aux  # 3610
    
    # Objective: maximize sum of edge variables
    c = np.zeros(total_vars)
    c[:num_edges] = 1.0  # Maximize edges
    
    # Constraints:
    # 1. For each aux var y_{w,u,v}: y <= x_{u,w}, y <= x_{v,w}, y >= x_{u,w} + x_{v,w} - 1
    # 2. For each pair (u,v): sum_w y_{w,u,v} <= 2
    
    # Let's build constraint matrix A and bounds
    # Using scipy's milp which takes LinearConstraint(A, lb, ub)
    
    # Count total constraints:
    # For each aux var: 3 constraints (y - x_uw <= 0, y - x_vw <= 0, -y + x_uw + x_vw >= -1)
    # That's 3 * 3420 = 10260
    # Plus 190 constraints for sum of aux per pair <= 2
    # Total: ~10450 constraints
    
    num_constraints = 3 * num_aux + num_pairs
    A = np.zeros((num_constraints, total_vars))
    lb = np.zeros(num_constraints)
    ub = np.zeros(num_constraints)
    
    constraint_idx = 0
    
    for pair_idx in range(num_pairs):
        u, v = idx_to_edge[pair_idx]
        
        # Get all w that are not u or v
        ws = [w for w in range(n) if w != u and w != v]
        
        for local_w_idx, w in enumerate(ws):
            aux_var_idx = num_edges + pair_idx * num_aux_per_pair + local_w_idx
            
            # Find edge indices for (u,w) and (v,w)
            if u < w:
                uw_edge = edge_to_idx[(u, w)]
            else:
                uw_edge = edge_to_idx[(w, u)]
            
            if v < w:
                vw_edge = edge_to_idx[(v, w)]
            else:
                vw_edge = edge_to_idx[(w, v)]
            
            # Constraint 1: y_{w,u,v} - x_{u,w} <= 0
            A[constraint_idx, aux_var_idx] = 1.0
            A[constraint_idx, uw_edge] = -1.0
            lb[constraint_idx] = -np.inf
            ub[constraint_idx] = 0.0
            constraint_idx += 1
            
            # Constraint 2: y_{w,u,v} - x_{v,w} <= 0
            A[constraint_idx, aux_var_idx] = 1.0
            A[constraint_idx, vw_edge] = -1.0
            lb[constraint_idx] = -np.inf
            ub[constraint_idx] = 0.0
            constraint_idx += 1
            
            # Constraint 3: -y_{w,u,v} + x_{u,w} + x_{v,w} >= -1
            # i.e., x_{u,w} + x_{v,w} - y_{w,u,v} >= -1
            A[constraint_idx, aux_var_idx] = -1.0
            A[constraint_idx, uw_edge] = 1.0
            A[constraint_idx, vw_edge] = 1.0
            lb[constraint_idx] = -1.0
            ub[constraint_idx] = np.inf
            constraint_idx += 1
        
        # Constraint 4: sum of y_{w,u,v} over all w <= 2
        for local_w_idx, w in enumerate(ws):
            aux_var_idx = num_edges + pair_idx * num_aux_per_pair + local_w_idx
            A[constraint_idx, aux_var_idx] = 1.0
        lb[constraint_idx] = -np.inf
        ub[constraint_idx] = 2.0
        constraint_idx += 1
    
    # Ensure constraint_idx matches
    assert constraint_idx == num_constraints, f"Constraint count mismatch: {constraint_idx} vs {num_constraints}"
    
    # Variable bounds: all binary [0, 1]
    bounds = Bounds(lb=np.zeros(total_vars), ub=np.ones(total_vars))
    
    # Integrality: all variables are integer (binary)
    integrality = np.ones(total_vars)
    
    # Solve
    try:
        result = milp(
            c=c,
            constraints=LinearConstraint(A, lb, ub),
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": int(time_limit - safety_margin - (time.time() - start_time))}
        )
    except Exception as e:
        # Fallback to deterministic 5xK4 construction
        adj = [0] * n
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    u2 = base + a
                    v2 = base + b
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
        edges = []
        for u2 in range(n):
            temp = adj[u2]
            while temp:
                lsb = temp & -temp
                v2 = lsb.bit_length() - 1
                if u2 < v2:
                    edges.append([u2, v2])
                temp -= lsb
        return edges
    
    # Check time
    if time.time() - start_time > time_limit - safety_margin:
        # Fallback
        adj = [0] * n
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    u2 = base + a
                    v2 = base + b
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
        edges = []
        for u2 in range(n):
            temp = adj[u2]
            while temp:
                lsb = temp & -temp
                v2 = lsb.bit_length() - 1
                if u2 < v2:
                    edges.append([u2, v2])
                temp -= lsb
        return edges
    
    if result.status == 0 or result.status == 1:  # Optimal or feasible
        # Extract edge variables
        x = result.x
        edges = []
        for i in range(num_edges):
            if x[i] > 0.5:
                u, v = idx_to_edge[i]
                edges.append([u, v])
        
        # Verify K2,3-free
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        valid = True
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if bin(common).count('1') > 2:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            return edges
    
    # Fallback to deterministic 5xK4
    adj = [0] * n
    for k in range(5):
        base = 4 * k
        for a in range(4):
            for b in range(a + 1, 4):
                u2 = base + a
                v2 = base + b
                adj[u2] |= (1 << v2)
                adj[v2] |= (1 << u2)
    edges = []
    for u2 in range(n):
        temp = adj[u2]
        while temp:
            lsb = temp & -temp
            v2 = lsb.bit_length() - 1
            if u2 < v2:
                edges.append([u2, v2])
            temp -= lsb
    return edges
