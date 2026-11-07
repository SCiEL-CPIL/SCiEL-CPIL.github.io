"""
Strategy: ILP (Integer Linear Programming) formulation using scipy.optimize.milp.
We model the selection of edges as binary variables x_e.
Constraints:
1. x_e = 0 for edges in the base graph (or simply not included in the variable set).
2. Triangle-free: For every potential triangle (u,v,w), x_uv + x_uw + x_vw <= 2.
   Also, if an edge is in the base graph, it counts as 1 for the triangle constraint.
   Let b_e be 1 if edge e is in base, 0 otherwise.
   Sum over e in triangle of (b_e + x_e) <= 2.
3. Connectivity: The graph must be connected. This is tricky for ILP.
   A common approach is to use a flow formulation or simply rely on the diameter constraint
   which implies connectivity if we ensure distances are finite.
   However, MILP solvers handle linear constraints. 
   To enforce diameter <= 4, we can use the "diameter" formulation with auxiliary variables.
   
   Let d_{ij} be the shortest path distance between i and j.
   We require d_{ij} <= 4 for all i,j.
   This is a non-convex constraint because d_{ij} = min over paths.
   Standard ILP for diameter uses the "all-pairs shortest path" linearization which is complex.
   
   Alternative: Since N=24 is small, we can iterate on the number of edges k.
   For a fixed k, we can use a different approach: 
   Actually, scipy's milp is a linear solver. It cannot directly handle "shortest path <= 4" easily without exponential variables or big-M tricks that might be loose.
   
   Let's refine the ILP approach for Diameter.
   We can use the following trick:
   Introduce variables y_{ij}^l representing whether there is a path of length <= l between i and j.
   This is still hard.
   
   Let's look at the structure. The base graph is a path.
   We are adding edges.
   Triangle-free condition is linear.
   Connectivity can be enforced by requiring that for every cut, there is at least one edge crossing it. (Exponential constraints, but N=24 is small enough to generate cuts? No, 2^23 cuts).
   
   Better approach for this specific task:
   The current best score is 10.0833 (which corresponds to 12 edges? 12/1.17? No, score = seed/value. Seed is likely 12? If score is 10.08, value is 12/10.08? No.
   Score = Seed / Value.
   If Seed is fixed, lower Value is better.
   Current program returns ~12 edges?
   Let's check the seed value. The prompt says "search score is seed/value".
   If the current program scores 10.0833, and it's a "seed program", maybe the seed value is the number of edges in the seed?
   Wait, the prompt says "score 10.0833" for the CURRENT PROGRAM.
   And "1.0 equals the seed program".
   So if Current Program Score = 10.0833, then Seed/Value_Current = 10.0833.
   If Value_Current is the number of edges, let's assume the current program finds 12 edges.
   Then Seed = 10.0833 * 12 = 121? That seems high.
   Or maybe the current program finds 1 edge? No.
   
   Let's look at the inspirations.
   Inspiration 1 score 11.0.
   Inspiration 2 score 10.0833.
   Inspiration 3 score 10.0833.
   
   If I find a solution with fewer edges, the score increases.
   
   ILP Formulation for Minimum Edge Augmentation with Diameter Constraint:
   Variables: x_e in {0,1} for each candidate edge e.
   Objective: Minimize sum x_e.
   
   Constraints:
   1. Triangle Free:
      For each triangle T = {e1, e2, e3} in the complete graph K24:
      sum_{e in T} (b_e + x_e) <= 2
      where b_e is 1 if e is in base, 0 otherwise.
      Note: If b_e=1, x_e must be 0 (since we only add edges not in base).
      
   2. Diameter <= 4:
      This is the hard part.
      We can use the "distance" variables d_{ij} for i < j.
      d_{ij} <= 4.
      d_{ij} >= 1 if i != j.
      d_{ij} <= d_{ik} + d_{kj} for all k.
      d_{ij} >= d_{ik} - d_{kj} ... (Triangle inequality for shortest paths is d_{ij} <= d_{ik} + d_{kj}).
      Also, d_{ij} must be at least 1 if there is an edge, but we need to link d_{ij} to the existence of paths.
      Standard ILP for shortest paths:
      d_{ij} >= d_{ik} + d_{kj} - M(1 - x_{path})? No.
      
      A robust way to enforce diameter <= D in ILP for small N:
      Use the "node potentials" method? No, that's for trees.
      
      Let's use the fact that N=24.
      We can iterate on k (number of edges) from 1 upwards.
      For a fixed k, we want to know if there exists a set of k edges such that:
      - Triangle free
      - Diameter <= 4
      
      This is a SAT-like problem.
      We can encode it as an ILP where we fix sum x_e = k.
      If the ILP is feasible, we have a solution with k edges.
      We increase k until feasible.
      
      How to encode Diameter <= 4 in ILP?
      Let d_{ij} be integer variables, 0 <= d_{ij} <= 4.
      d_{ii} = 0.
      d_{ij} >= 1 for i != j.
      d_{ij} <= 4 for all i,j.
      
      Constraints linking d_{ij} to edges:
      If there is an edge (u,v) in the final graph (base or added), then d_{uv} <= 1.
      Specifically: d_{uv} <= 1 if (b_{uv}=1 or x_{uv}=1).
      This can be written as: d_{uv} <= 1 + M * (1 - (b_{uv} + x_{uv}))?
      If b+x >= 1, then d <= 1.
      If b+x = 0, no constraint (d can be up to 4).
      So: d_{uv} <= 1 + 3 * (1 - (b_{uv} + x_{uv})) is wrong because if b+x=0, d <= 4.
      Actually:
      If edge exists, d <= 1.
      If edge does not exist, d can be anything (up to 4).
      So: d_{uv} <= 1 + 3 * (1 - (b_{uv} + x_{uv}))
      If b+x=1: d <= 1 + 0 = 1.
      If b+x=0: d <= 1 + 3 = 4.
      This works!
      
      Also, shortest path properties:
      d_{ij} <= d_{ik} + d_{kj} for all i,j,k.
      This ensures d_{ij} is a valid upper bound on the shortest path distance.
      Since we minimize the number of edges, and the constraints force d_{ij} to be small if edges are present, the solver will find the minimal edges that satisfy the distance bounds.
      Wait, does this force the *actual* shortest path to be <= 4?
      Yes, because d_{ij} is constrained to be <= 4.
      And d_{ij} is constrained by the triangle inequality d_{ij} <= d_{ik} + d_{kj}.
      And d_{uv} <= 1 if edge (u,v) exists.
      This system of linear inequalities defines the set of all possible shortest path distance matrices for graphs with the given edges.
      If a feasible solution exists, there is a graph with those edges that has diameter <= 4.
      
      So the ILP is:
      Minimize sum x_e
      Subject to:
      1. x_e in {0,1}
      2. Triangle constraints: sum_{e in T} (b_e + x_e) <= 2
      3. d_{ij} in {0,1,2,3,4} (or continuous 0..4, integrality not strictly needed for d if we use the edge constraints? No, d should be integer for correctness, but continuous might work with the <= constraints? Let's use integers to be safe, or just continuous since the bounds are tight. Actually, d_{ij} <= d_{ik} + d_{kj} with d_{uv}<=1 for edges ensures d is the shortest path distance. Shortest path distances are integers. So continuous variables with these constraints will yield integer solutions automatically? Not necessarily, but likely. Let's use integers for d to be safe.)
      4. d_{uv} <= 1 + 3 * (1 - (b_{uv} + x_{uv})) for all pairs u,v.
      5. d_{ij} <= d_{ik} + d_{kj} for all i,j,k.
      6. d_{ii} = 0.
      7. d_{ij} >= 1 for i != j. (Actually, if the graph is connected, d >= 1. If not connected, d can be large. But we require d <= 4, which implies connectivity for N=24? No, a disconnected graph can have d=inf. But our constraint d <= 4 forces finite distance, hence connectivity.)
      
      This is a valid ILP formulation!
      
      Complexity:
      Variables:
      x_e: ~276 candidates.
      d_{ij}: 24*24 = 576 variables.
      Total ~850 variables.
      
      Constraints:
      Triangle: Number of triangles in K24 is C(24,3) = 2024.
      Edge-Dist: 24*23/2 = 276 constraints.
      Triangle Ineq: 24*24*24 = 13824 constraints.
      Total ~16000 constraints.
      
      SciPy's milp (HiGHS) should handle this size very quickly.
      
      We iterate k from 1 to 10 (or until found).
      For each k, add constraint sum x_e = k.
      If feasible, extract x_e and return edges.
"""

import time
import sys
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

P = {
    'n': 24,
    'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9],
              [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
              [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]],
    'diameter': 4
}

N = P['n']
BASE_EDGES = set()
BASE_ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    BASE_EDGES.add((u, v))
    BASE_EDGES.add((v, u))
    BASE_ADJ[u].add(v)
    BASE_ADJ[v].add(u)

# Candidate edges: all pairs not in base
CANDIDATES = []
for u in range(N):
    for v in range(u + 1, N):
        if (u, v) not in BASE_EDGES:
            CANDIDATES.append((u, v))

NUM_CANDIDATES = len(CANDIDATES)

# Map candidate index to edge
CAND_ID_TO_EDGE = {i: cand for i, cand in enumerate(CANDIDATES)}
EDGE_TO_CAND_ID = {cand: i for i, cand in enumerate(CANDIDATES)}

# Precompute triangles
# A triangle is a set of 3 vertices {u,v,w} such that all 3 pairs are edges in K_N.
# We need to check if adding edges creates triangles.
# The constraint is: sum of (base_edge + added_edge) for the 3 pairs <= 2.
# Let's list all triangles in K_N.
TRIANGLES = []
for u, v, w in combinations(range(N), 3):
    e1 = (u, v) if u < v else (v, u)
    e2 = (v, w) if v < w else (w, v)
    e3 = (u, w) if u < w else (w, u)
    # Store the indices in CANDIDATES if they exist, or -1 if base
    # Actually, we just need to know the coefficient for x_e.
    # If edge is in base, coefficient is 0 for x, but contributes 1 to the sum.
    # If edge is in candidates, coefficient is 1 for x, contributes 0 to base sum.
    
    # We will build the A matrix for triangles later.
    TRIANGLES.append((u, v, w))

NUM_TRIANGLES = len(TRIANGLES)

def _solve_ilp(k, deadline):
    """
    Solve ILP for exactly k added edges.
    Returns list of edges if feasible, else None.
    """
    # Variables:
    # 0 .. NUM_CANDIDATES-1 : x_e (binary)
    # NUM_CANDIDATES .. NUM_CANDIDATES + N*N - 1 : d_{ij} (integer 0..4)
    
    n_x = NUM_CANDIDATES
    n_d = N * N
    n_vars = n_x + n_d
    
    # Objective: Minimize sum x_e
    c = np.zeros(n_vars)
    c[:n_x] = 1.0
    
    # Bounds
    # x_e in [0, 1]
    # d_{ij} in [0, 4]
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    # d_{ij} upper bound 4
    ub[n_x:] = 4.0
    
    # Integrality
    # x_e integer
    # d_{ij} integer
    integrality = np.ones(n_vars)
    
    # Constraints
    A_rows = []
    lb_rows = []
    ub_rows = []
    
    # 1. Sum x_e = k
    row = np.zeros(n_vars)
    row[:n_x] = 1.0
    A_rows.append(row)
    lb_rows.append(k)
    ub_rows.append(k)
    
    # 2. Triangle Constraints: sum (b_e + x_e) <= 2
    # For each triangle (u,v,w):
    #   Let e1,e2,e3 be the edges.
    #   base_count = number of edges in base.
    #   sum x_ei <= 2 - base_count
    #   If base_count == 2, sum x_ei <= 0 => x_ei = 0 for the non-base edge.
    #   If base_count == 1, sum x_ei <= 1.
    #   If base_count == 0, sum x_ei <= 2.
    
    for u, v, w in TRIANGLES:
        e1 = (u, v) if u < v else (v, u)
        e2 = (v, w) if v < w else (w, v)
        e3 = (u, w) if u < w else (w, u)
        
        base_count = 0
        cand_indices = []
        
        if e1 in BASE_EDGES: base_count += 1
        else: cand_indices.append(EDGE_TO_CAND_ID[e1])
        
        if e2 in BASE_EDGES: base_count += 1
        else: cand_indices.append(EDGE_TO_CAND_ID[e2])
        
        if e3 in BASE_EDGES: base_count += 1
        else: cand_indices.append(EDGE_TO_CAND_ID[e3])
        
        rhs = 2 - base_count
        if rhs < 0:
            # Infeasible triangle (impossible, base graph is a path, so max base_count in triangle is 2? 
            # Path 0-1-2. Triangle 0,1,2. Edges (0,1) base, (1,2) base, (0,2) not base.
            # base_count=2. rhs=0. x_{02} <= 0. So x_{02}=0. Correct.)
            # If base_count=3, impossible in simple graph.
            # If rhs < 0, return infeasible
            return None
        
        row = np.zeros(n_vars)
        for idx in cand_indices:
            row[idx] = 1.0
        A_rows.append(row)
        lb_rows.append(0)
        ub_rows.append(rhs)
        
    # 3. Edge-Distance Constraints: d_{uv} <= 1 + 3 * (1 - (b_{uv} + x_{uv}))
    # Rearranged: d_{uv} + 3 * (b_{uv} + x_{uv}) <= 4
    # If b+x=1: d + 3 <= 4 => d <= 1.
    # If b+x=0: d <= 4.
    
    for u in range(N):
        for v in range(u + 1, N):
            d_idx = n_x + u * N + v
            d_idx_rev = n_x + v * N + u
            
            e = (u, v)
            if e in BASE_EDGES:
                b_val = 1
                x_idx = -1
            else:
                b_val = 0
                x_idx = EDGE_TO_CAND_ID[e]
            
            # d_{uv} + 3*b + 3*x <= 4
            row = np.zeros(n_vars)
            row[d_idx] = 1.0
            row[d_idx_rev] = 1.0 # d is symmetric, we constrain both? 
            # Actually, we defined d_{ij} for all i,j. 
            # We should ensure d_{ij} = d_{ji}. 
            # Or just use one variable per pair? 
            # My variable indexing is u*N+v. 
            # Let's just constrain d_{uv} and d_{vu} separately with the same logic?
            # Or better, enforce d_{uv} = d_{vu} as an equality constraint?
            # To save constraints, let's just use d_{uv} for u<v and assume symmetry in the triangle inequality?
            # No, the triangle inequality d_{ij} <= d_{ik} + d_{kj} uses all pairs.
            # It's safer to have d_{ij} for all i,j and add symmetry constraints d_{ij} - d_{ji} = 0.
            # That adds N*(N-1)/2 constraints.
            
            # Let's stick to d_{ij} for all i,j.
            # Constraint for (u,v):
            # d_{uv} + 3*b + 3*x <= 4
            # d_{vu} + 3*b + 3*x <= 4
            
            # We can combine them or add two rows.
            # Let's add two rows.
            
            row1 = np.zeros(n_vars)
            row1[d_idx] = 1.0
            if x_idx != -1:
                row1[x_idx] = 3.0
            A_rows.append(row1)
            lb_rows.append(0)
            ub_rows.append(4 - 3*b_val)
            
            row2 = np.zeros(n_vars)
            row2[d_idx_rev] = 1.0
            if x_idx != -1:
                row2[x_idx] = 3.0
            A_rows.append(row2)
            lb_rows.append(0)
            ub_rows.append(4 - 3*b_val)
            
    # 4. Symmetry: d_{ij} - d_{ji} = 0
    for u in range(N):
        for v in range(u + 1, N):
            d_idx = n_x + u * N + v
            d_idx_rev = n_x + v * N + u
            row = np.zeros(n_vars)
            row[d_idx] = 1.0
            row[d_idx_rev] = -1.0
            A_rows.append(row)
            lb_rows.append(0)
            ub_rows.append(0)
            
    # 5. Diagonal: d_{ii} = 0
    for u in range(N):
        d_idx = n_x + u * N + u
        row = np.zeros(n_vars)
        row[d_idx] = 1.0
        A_rows.append(row)
        lb_rows.append(0)
        ub_rows.append(0)
        
    # 6. Triangle Inequality: d_{ij} <= d_{ik} + d_{kj}
    # d_{ij} - d_{ik} - d_{kj} <= 0
    for i in range(N):
        for j in range(N):
            if i == j: continue
            for k in range(N):
                if k == i or k == j: continue
                d_ij = n_x + i * N + j
                d_ik = n_x + i * N + k
                d_kj = n_x + k * N + j
                row = np.zeros(n_vars)
                row[d_ij] = 1.0
                row[d_ik] = -1.0
                row[d_kj] = -1.0
                A_rows.append(row)
                lb_rows.append(-np.inf)
                ub_rows.append(0)
                
    A = np.array(A_rows)
    lb = np.array(lb_rows)
    ub = np.array(ub_rows)
    
    # Solve
    try:
        res = milp(c=c, constraints=LinearConstraint(A, lb, ub), integrality=integrality, bounds=Bounds(lb=lb, ub=ub))
        if time.time() > deadline:
            return None
        if res.status == 0:
            x = res.x[:n_x]
            edges = []
            for i in range(n_x):
                if x[i] > 0.5:
                    edges.append(CAND_ID_TO_EDGE[i])
            return edges
        else:
            return None
    except Exception:
        return None

def _verify_solution(added_edges):
    """Verify the solution is valid."""
    adj = [set() for _ in range(N)]
    for u in range(N):
        adj[u] = set(BASE_ADJ[u])
    for u, v in added_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Triangle free
    for u in range(N):
        nbrs = adj[u]
        for v in nbrs:
            if v > u:
                if nbrs & adj[v]:
                    return False
                    
    # Diameter
    for i in range(N):
        dist = [-1] * N
        dist[i] = 0
        queue = [i]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > 4:
                        return False
                    queue.append(v)
        for j in range(N):
            if dist[j] == -1:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    # Try k from 1 to 10
    for k in range(1, 11):
        if time.time() > deadline:
            break
        edges = _solve_ilp(k, deadline)
        if edges is not None:
            # Convert to list of lists of ints
            res = [[int(u), int(v)] for u, v in edges]
            if _verify_solution(res):
                return res
            else:
                # ILP might have found a solution that violates some implicit assumption?
                # Unlikely, but check.
                pass
                
    # Fallback: Greedy
    # (Keep a simple fallback just in case ILP fails or times out)
    adj = [set() for _ in range(N)]
    for u in range(N):
        adj[u] = set(BASE_ADJ[u])
    added = []
    for _ in range(20):
        # Find farthest pair
        dist = [[-1]*N for _ in range(N)]
        for i in range(N):
            dist[i][i]=0
            q=[i]; qi=0
            while qi<len(q):
                u=q[qi]; qi+=1
                for v in adj[u]:
                    if dist[i][v]==-1:
                        dist[i][v]=dist[i][u]+1
                        q.append(v)
        max_d = -1; bu=-1; bv=-1
        for i in range(N):
            for j in range(i+1,N):
                if dist[i][j] > max_d:
                    max_d = dist[i][j]
                    bu, bv = i, j
        if max_d <= 4: break
        if (bu,bv) not in BASE_EDGES and (bv,bu) not in BASE_EDGES:
             # Check triangle
             ok = True
             for w in range(N):
                 if w!=bu and w!=bv:
                     if (bu,w) in BASE_EDGES or (bu,w) in [(u,v) for u,v in added] or (w,bu) in BASE_EDGES or (w,bu) in [(u,v) for u,v in added]:
                         if (bv,w) in BASE_EDGES or (bv,w) in [(u,v) for u,v in added] or (w,bv) in BASE_EDGES or (w,bv) in [(u,v) for u,v in added]:
                             ok = False
                             break
             if ok:
                 adj[bu].add(bv); adj[bv].add(bu)
                 added.append((bu,bv))
    if _verify_solution(added):
         return [[int(u), int(v)] for u, v in added]
         
    return []

# EVOLVE-BLOCK-END
