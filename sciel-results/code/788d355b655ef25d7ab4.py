"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from collections import deque
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    base_set = set()
    for e in base_edges:
        base_set.add(tuple(sorted(e)))
    
    # Candidate edges: between even and odd, not in base
    candidates = []
    cand_index = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            if (u % 2) != (v % 2) and (u, v) not in base_set:
                candidates.append((u, v))
                cand_index[(u, v)] = idx
                idx += 1
                
    num_cands = len(candidates)
    
    # Identify pairs that need shortening (base distance > 4)
    pairs_to_fix = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) > 4:
                pairs_to_fix.append((i, j))
                
    # Generate all simple paths of length <= 4 between i and j
    # A path is a sequence of vertices v0, v1, ..., vk
    # Edges are (v0,v1), ..., (vk-1,vk)
    # We need to express the constraint: at least one path exists.
    # For each path, identify the set of candidate edges used.
    
    # Pre-compute adjacency for path finding (all possible edges: base + candidates)
    # For path generation, we consider the graph where ALL candidate edges are available.
    # If a path is found, we check which of its edges are candidates.
    
    adj_all = [[] for _ in range(n)]
    for u, v in base_edges:
        adj_all[u].append(v)
        adj_all[v].append(u)
    for u, v in candidates:
        adj_all[u].append(v)
        adj_all[v].append(u)
        
    # Function to find all simple paths of length L from src to dst
    def find_paths(src, dst, max_len):
        paths = []
        def dfs(curr, path, visited):
            if len(path) - 1 > max_len:
                return
            if curr == dst:
                if len(path) - 1 > 0: # Non-empty path
                    paths.append(list(path))
                return
            for nei in adj_all[curr]:
                if nei not in visited:
                    visited.add(nei)
                    path.append(nei)
                    dfs(nei, path, visited)
                    path.pop()
                    visited.remove(nei)
        
        dfs(src, [src], {src})
        return paths

    # Build ILP
    # Variables:
    # x[0..num_cands-1] : binary, 1 if candidate edge is added
    # y[k] : binary, 1 if path k is "selected" (i.e., all its candidate edges are present)
    # Actually, we don't need y to be a decision variable for the objective, but we need it to link constraints.
    # Constraint: For each pair (i,j), sum(y_p for p in Paths(i,j)) >= 1
    # Link: y_p <= x_e for all e in CandidateEdges(p)
    
    # Let's count the number of paths for each pair
    pair_paths = {}
    total_paths = 0
    for i, j in pairs_to_fix:
        paths = find_paths(i, j, 4)
        if not paths:
            # If no path of length <= 4 exists even with all edges, the problem is infeasible
            # But with all edges, it's a complete bipartite graph, diameter is 2. So paths exist.
            pass
        pair_paths[(i, j)] = paths
        total_paths += len(paths)
        
    # Total variables: num_cands + total_paths
    # x vars: 0 to num_cands - 1
    # y vars: num_cands to num_cands + total_paths - 1
    
    c = np.zeros(num_cands + total_paths)
    for i in range(num_cands):
        c[i] = 1.0 # Minimize sum of x
    
    # Constraints:
    # 1. y_p <= x_e for all e in C_p
    # 2. sum(y_p) >= 1 for each pair
    
    A_ub = [] # For y_p - x_e <= 0
    b_ub = []
    
    A_eq = [] # For sum(y_p) >= 1 -> -sum(y_p) <= -1
    b_eq = []
    
    # We will use LinearConstraint with bounds
    # Let's build the matrix for A_ub: y_p - x_e <= 0
    # And A_lb: sum(y_p) >= 1
    
    # To use scipy.optimize.milp, we define:
    # c: objective
    # constraints: list of LinearConstraint
    # integrality: 1 for binary vars
    # bounds: (0, 1) for all vars
    
    integrality = np.ones(num_cands + total_paths)
    bounds = Bounds(lb=0, ub=1)
    
    constraints = []
    
    # Helper to get index of y var for a path
    y_offset = num_cands
    y_index_map = {}
    current_y_idx = 0
    for i, j in pairs_to_fix:
        for p in pair_paths[(i, j)]:
            y_index_map[(i, j, tuple(p))] = y_offset + current_y_idx
            current_y_idx += 1
            
    # Constraint 1: y_p <= x_e
    # y_p - x_e <= 0
    # We can add these individually or in batches.
    # Number of such constraints is sum over paths of len(C_p).
    # Let's build a sparse matrix or just a list of linear constraints.
    # scipy.optimize.milp accepts a list of LinearConstraint objects.
    
    # Let's collect rows for A_ub (y - x <= 0)
    # And rows for A_lb (sum y >= 1)
    
    # Since N is small, we can build dense matrices if size is manageable.
    # num_vars ~ 121 + 276 * (avg paths).
    # Avg paths might be small. Let's estimate.
    # If it's too big, we use sparse. But let's try dense first for simplicity, or use linear_constraint with arrays.
    
    # Let's count total constraints to decide.
    # It's safer to use the list of LinearConstraint.
    
    # Constraint type 1: y_p - x_e <= 0
    # For each path p in pair (i,j):
    #   For each candidate edge e in p:
    #       y_p - x_e <= 0
    
    # Constraint type 2: sum_{p in pair(i,j)} y_p >= 1
    #       -sum y_p <= -1
    
    # Let's build the matrices for A_ub and A_lb if possible, or use the list format.
    # The list format is:
    # [LinearConstraint(A, lb, ub), ...]
    
    # Let's create the A matrix for all "y - x <= 0" constraints.
    # And a separate one for "sum y >= 1".
    
    # Actually, we can just append to lists of (coeffs, lb, ub) and then construct.
    
    # Let's estimate size.
    # Pairs: ~200.
    # Paths per pair: likely < 10.
    # Total y vars: ~2000.
    # Total x vars: 121.
    # Total vars: ~2121.
    # Constraints:
    #   y-x: Each path has length <=4, so <=4 candidate edges. 2000 * 4 = 8000 constraints.
    #   sum-y: 200 constraints.
    # Total ~8200 constraints.
    # This is small enough for dense matrices.
    
    A_ub_rows = []
    b_ub_vals = []
    
    # Map (i,j, path_tuple) to y_index
    # We already have y_index_map
    
    # We need to map candidate edge (u,v) to x_index
    # cand_index[(u,v)]
    
    # Iterate through all paths again to build constraints
    for i, j in pairs_to_fix:
        for p in pair_paths[(i, j)]:
            y_idx = y_index_map[(i, j, tuple(p))]
            # Find candidate edges in this path
            for k in range(len(p) - 1):
                u, v = p[k], p[k+1]
                edge = tuple(sorted((u, v)))
                if edge in cand_index:
                    x_idx = cand_index[edge]
                    # y_idx - x_idx <= 0
                    row = np.zeros(num_cands + total_paths)
                    row[y_idx] = 1.0
                    row[x_idx] = -1.0
                    A_ub_rows.append(row)
                    b_ub_vals.append(0.0)
                    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Constraints: sum y >= 1 for each pair
    A_lb_rows = []
    b_lb_vals = []
    
    for i, j in pairs_to_fix:
        row = np.zeros(num_cands + total_paths)
        for p in pair_paths[(i, j)]:
            y_idx = y_index_map[(i, j, tuple(p))]
            row[y_idx] = 1.0
        A_lb_rows.append(row)
        b_lb_vals.append(1.0)
        
    A_lb = np.array(A_lb_rows)
    b_lb = np.array(b_lb_vals)
    
    # Combine into LinearConstraint objects
    # Constraint 1: A_ub @ x <= b_ub
    # Constraint 2: A_lb @ x >= b_lb  =>  -A_lb @ x <= -b_lb
    # Or use LinearConstraint(A, lb, ub)
    
    cons1 = LinearConstraint(A_ub, ub=b_ub)
    cons2 = LinearConstraint(A_lb, lb=b_lb)
    
    res = milp(c=c, constraints=[cons1, cons2], integrality=integrality, bounds=bounds)
    
    if not res.success:
        # Fallback to greedy if ILP fails
        added = list(candidates)
        improved = True
        while improved:
            improved = False
            for i in range(len(added)):
                test_added = added[:i] + added[i+1:]
                if _check_diameter(n, base_edges, test_added) <= 4:
                    added = test_added
                    improved = True
                    break
        return [[u, v] for u, v in added]
        
    x_sol = res.x[:num_cands]
    added = []
    for i in range(num_cands):
        if x_sol[i] > 0.5:
            added.append(candidates[i])
            
    return [[u, v] for u, v in added]

def _build_adjacency(n, base_edges, added_edges):
    adj = [[] for _ in range(n)]
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added_edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _bfs_distances(n, adj, src):
    dist = [-1] * n
    dist[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _check_diameter(n, base_edges, added_edges):
    adj = _build_adjacency(n, base_edges, added_edges)
    max_dist = 0
    for i in range(n):
        dists = _bfs_distances(n, adj, i)
        for d in dists:
            if d > max_dist:
                max_dist = d
    return max_dist

# EVOLVE-BLOCK-END
