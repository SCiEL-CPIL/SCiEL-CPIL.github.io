import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 15
    # All possible edges in a simple undirected graph on n vertices
    edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(edge_list)
    
    # Map edge index to (u, v)
    edge_to_uv = {idx: edge_list[idx] for idx in range(m)}
    
    # Precompute for each edge e=(u,v), the set of edges incident to u (excluding e) and v (excluding e)
    # We need to ensure that for any path s-u-v-t, deg(s) != deg(t).
    # This constraint is non-linear because deg(s) and deg(t) depend on the solution.
    # However, we can linearize by observing that deg(s) is the sum of x_k for edges k incident to s.
    # The constraint "deg(s) != deg(t)" is equivalent to "deg(s) - deg(t) != 0".
    # This is a disjunction: (deg(s) - deg(t) >= 1) OR (deg(s) - deg(t) <= -1).
    # This requires integer variables or big-M constraints with binary variables for the direction.
    
    # Alternative approach:
    # Since n=15, degrees are integers between 0 and 14.
    # We can introduce binary variables y_{s,t,d} indicating that deg(s) - deg(t) = d? No, too many.
    
    # Let's try a different formulation.
    # The condition is: For all distinct s, u, v, t such that (s,u), (u,v), (v,t) are edges in the graph,
    # it is NOT the case that deg(s) == deg(t).
    
    # Let x_e be binary variable for edge e.
    # deg(i) = sum_{e incident to i} x_e.
    
    # For a specific path configuration (s, u, v, t), the path exists if x_{su}=1, x_{uv}=1, x_{vt}=1.
    # If the path exists, we must have deg(s) != deg(t).
    # This can be written as:
    # If x_{su} + x_{uv} + x_{vt} == 3, then |deg(s) - deg(t)| >= 1.
    
    # This is a logical implication: (x_{su} & x_{uv} & x_{vt}) => (deg(s) != deg(t)).
    # Equivalent to: (x_{su} & x_{uv} & x_{vt}) => (deg(s) - deg(t) >= 1 OR deg(s) - deg(t) <= -1).
    
    # We can introduce binary variables z_{s,t} for each pair (s,t) to indicate the direction of inequality?
    # Or simpler: Since degrees are small, we can iterate over possible degree differences?
    
    # Actually, there is a simpler structure.
    # If we fix the degrees of all vertices, the problem becomes:
    # Maximize edges such that for any path s-u-v-t, deg(s) != deg(t).
    # But degrees are determined by edges.
    
    # Let's use the big-M method with binary direction variables.
    # For each potential path (s, u, v, t) with distinct vertices:
    # Let P_{s,u,v,t} be the indicator that edges (s,u), (u,v), (v,t) are all present.
    # P_{s,u,v,t} <= x_{su}, x_{uv}, x_{vt}.
    # We want: P_{s,u,v,t} == 1 => (deg(s) - deg(t) >= 1 OR deg(s) - deg(t) <= -1).
    
    # This can be modeled with two binary variables b^+_{s,t} and b^-_{s,t} for each pair (s,t)?
    # No, the direction depends on the specific path.
    
    # Let's define for each ordered pair of distinct vertices (s, t):
    # d_{s,t} = deg(s) - deg(t).
    # We want: For all paths s-u-v-t, if the path exists, then d_{s,t} != 0.
    
    # This is equivalent to:
    # For all paths s-u-v-t:
    # x_{su} + x_{uv} + x_{vt} - 3 <= M * (1 - y_{s,u,v,t}^+) + M * (1 - y_{s,u,v,t}^-) ? No.
    
    # Let's introduce a binary variable z_{s,t} for each unordered pair {s,t} ?
    # No, the constraint is specific to the path.
    
    # Let's try a different angle.
    # The constraint is violated if there is a path s-u-v-t with deg(s) = deg(t).
    # This means x_{su}=1, x_{uv}=1, x_{vt}=1 and deg(s) = deg(t).
    # We can forbid this combination.
    
    # Since n=15, the number of potential paths is roughly n^4 = 50,625.
    # This is manageable.
    
    # Variables:
    # x_e for e in edges (m variables, binary)
    # We need to linearize the condition:
    # (x_{su} & x_{uv} & x_{vt}) & (deg(s) == deg(t)) is forbidden.
    
    # deg(s) == deg(t) is a linear equality: sum_{i~s} x_{is} - sum_{j~t} x_{jt} = 0.
    # The conjunction of binary variables and a linear equality is hard.
    
    # However, we can enumerate the possible degree values.
    # deg(i) is an integer in [0, 14].
    # We can introduce binary variables d_{i,k} indicating deg(i) = k.
    # Constraints:
    # 1. sum_k d_{i,k} = 1 for all i.
    # 2. sum_k k * d_{i,k} = deg(i) = sum_{e~i} x_e.
    
    # Now the condition "deg(s) = deg(t)" becomes:
    # sum_k d_{s,k} * d_{t,k} = 1 ? No, that's non-linear (product of binaries).
    # It is equivalent to: There exists k such that d_{s,k}=1 AND d_{t,k}=1.
    # This can be linearized with auxiliary variables q_{s,t,k} = d_{s,k} * d_{t,k}.
    # q_{s,t,k} <= d_{s,k}, q_{s,t,k} <= d_{t,k}, q_{s,t,k} >= d_{s,k} + d_{t,k} - 1.
    
    # The path exists if x_{su}=1, x_{uv}=1, x_{vt}=1.
    # Let p_{s,u,v,t} be the indicator that the path exists.
    # p_{s,u,v,t} <= x_{su}, p_{s,u,v,t} <= x_{uv}, p_{s,u,v,t} <= x_{vt}.
    # p_{s,u,v,t} >= x_{su} + x_{uv} + x_{vt} - 2.
    
    # The constraint is: If path exists AND deg(s)==deg(t), then invalid.
    # So: p_{s,u,v,t} * (sum_k q_{s,t,k}) <= 0.
    # Since p and q are binary (or 0/1), this means p_{s,u,v,t} <= 1 - q_{s,t,k} for all k?
    # No, if deg(s)==deg(t), then sum_k q_{s,t,k} = 1.
    # We want to forbid p=1 when sum_k q_{s,t,k} = 1.
    # So: p_{s,u,v,t} <= 1 - q_{s,t,k} is not sufficient because q_{s,t,k} is 1 for only one k.
    # We need: p_{s,u,v,t} + sum_k q_{s,t,k} <= 1.
    
    # This works!
    # If the path exists (p=1), then sum_k q_{s,t,k} must be 0, meaning deg(s) != deg(t).
    # If deg(s) == deg(t), then sum_k q_{s,t,k} = 1, so p must be 0, meaning the path cannot exist.
    
    # This formulation is valid.
    
    # Number of variables:
    # x_e: 105
    # d_{i,k}: 15 * 15 = 225
    # q_{s,t,k}: 15 * 14 * 15 = 3150 (s!=t, k=0..14)
    # p_{s,u,v,t}: Number of simple paths of length 3.
    # s, u, v, t distinct.
    # u, v can be any edge. s in N(u)\{v}, t in N(v)\{u}.
    # In the complete graph, for each edge (u,v), there are 13 choices for s and 13 for t.
    # 105 edges * 13 * 13 = 17,745 paths.
    # Total variables ~ 21,000.
    # Constraints:
    # Degree definitions: 15
    # Degree binary consistency: 15 * 15 = 225
    # q definitions: 3150 * 3 = 9450
    # p definitions: 17745 * 4 = 70980
    # Final constraint: 17745
    # Total constraints ~ 90,000.
    
    # This might be too large for scipy.milp to solve quickly within 50s.
    # scipy.milp uses HiGHS. HiGHS is fast, but 90k constraints might be tight.
    
    # Optimization:
    # We don't need q_{s,t,k} for all s,t. Only for pairs that can be endpoints of a path.
    # In a simple graph, any pair s,t can be endpoints if there is a path of length 3 between them.
    # In K15, any pair is connected by many paths.
    
    # Can we reduce the problem?
    # Maybe we can fix the degree sequence?
    # No, we want to maximize edges.
    
    # Let's try a simpler ILP first without the q variables, using big-M on the degree difference.
    # Constraint: p_{s,u,v,t} == 1 => |deg(s) - deg(t)| >= 1.
    # This is: p_{s,u,v,t} == 1 => (deg(s) - deg(t) >= 1 OR deg(s) - deg(t) <= -1).
    # Introduce binary b_{s,u,v,t} for each path?
    # If b=1, then deg(s) - deg(t) >= 1.
    # If b=0, then deg(s) - deg(t) <= -1.
    # Constraints:
    # deg(s) - deg(t) >= 1 - M * (1 - b)
    # deg(s) - deg(t) <= -1 + M * b
    # And we need to link b to p?
    # Actually, we don't need b to be linked to p directly if we just enforce the implication.
    # The implication is: p=1 => (A OR B).
    # This is equivalent to: p=1 => A OR p=1 => B.
    # This requires two binaries per path? Or one binary per path to choose the direction?
    # Yes, one binary b_{path} per path.
    # If p=1, b can be 0 or 1.
    # If b=1, enforce deg(s)-deg(t) >= 1.
    # If b=0, enforce deg(s)-deg(t) <= -1.
    # We also need to ensure that if p=0, the constraints are relaxed.
    # deg(s) - deg(t) >= 1 - M * (1 - b) - M * (1 - p)?
    # If p=0, we want no constraint.
    # If p=1 and b=1: deg(s)-deg(t) >= 1.
    # If p=1 and b=0: deg(s)-deg(t) <= -1.
    
    # Let's write the constraints:
    # 1. deg(s) - deg(t) - 1 >= -M * (1 - b) - M * (1 - p)
    #    If p=1, b=1: LHS >= 0 => deg(s)-deg(t) >= 1.
    #    If p=1, b=0: LHS >= -M => no constraint from this one?
    #    Wait, if b=0, we want the OTHER constraint to be active.
    
    # Constraint A: deg(s) - deg(t) >= 1 - M * (1 - b) - M * (1 - p)
    # Constraint B: -(deg(s) - deg(t)) >= 1 - M * b - M * (1 - p)
    
    # If p=1:
    #   If b=1: A is active (deg diff >= 1), B is relaxed (deg diff <= M).
    #   If b=0: A is relaxed, B is active (deg diff <= -1).
    # If p=0:
    #   A and B are relaxed.
    
    # This works!
    # Variables:
    # x_e (105)
    # b_path (17745)
    # Total ~ 18k variables.
    # Constraints:
    # 2 per path (35k)
    # Plus degree definitions (15)
    # Plus bounds.
    
    # This is much smaller than the previous formulation.
    
    # Let's implement this.
    
    # M value: Max degree difference is 14. So M=14 is sufficient.
    
    c = np.zeros(m) # Objective: maximize sum x_e, so minimize -sum x_e
    c[:] = -1
    
    # Variable indices
    # 0..m-1: x_e
    # m..m+num_paths-1: b_path
    
    path_indices = []
    # Generate all simple paths s-u-v-t
    # u, v are adjacent in potential graph (any pair).
    # s is neighbor of u (s!=v, s!=t).
    # t is neighbor of v (t!=u, t!=s).
    # In ILP, we consider all potential paths in K15.
    
    # We need to map path (s,u,v,t) to edge indices of (s,u), (u,v), (v,t).
    edge_to_idx = {edge_list[i]: i for i in range(m)}
    
    A_rows = []
    b_lower = []
    b_upper = []
    
    # 1. Degree definitions are not explicit constraints in this formulation?
    # We use deg(s) in the constraints. deg(s) is a linear function of x.
    # deg(s) = sum_{e~s} x_e.
    # So the constraints are linear in x and b.
    
    # Let's build the matrix A.
    # Rows for each path constraint.
    
    num_paths = 0
    path_data = []
    for u in range(n):
        for v in range(u + 1, n):
            # Edge (u,v) exists? No, we are checking if it DOES exist in the solution.
            # The path is s-u-v-t.
            # Edges involved: (s,u), (u,v), (v,t).
            # s can be any vertex != u, v, t.
            # t can be any vertex != u, v, s.
            # But s and t must be distinct from u and v.
            # Also s != t.
            
            # Iterate s in 0..n-1, s!=u, s!=v
            # Iterate t in 0..n-1, t!=u, t!=v, t!=s
            
            for s in range(n):
                if s == u or s == v:
                    continue
                for t in range(n):
                    if t == u or t == v or t == s:
                        continue
                    
                    # Edges: (s,u), (u,v), (v,t)
                    # Get indices
                    e_su = tuple(sorted((s, u)))
                    e_uv = tuple(sorted((u, v)))
                    e_vt = tuple(sorted((v, t)))
                    
                    if e_su in edge_to_idx and e_uv in edge_to_idx and e_vt in edge_to_idx:
                        idx_su = edge_to_idx[e_su]
                        idx_uv = edge_to_idx[e_uv]
                        idx_vt = edge_to_idx[e_vt]
                        
                        path_data.append((s, t, idx_su, idx_uv, idx_vt))
                        num_paths += 1
                    else:
                        # This shouldn't happen for K15
                        pass
    
    # Total variables: m + num_paths
    total_vars = m + num_paths
    
    # Objective
    c_obj = np.zeros(total_vars)
    c_obj[:m] = -1 # Minimize -sum(x) => Maximize sum(x)
    
    # Constraints
    # For each path (s, t, idx_su, idx_uv, idx_vt):
    # b_var index = m + path_id
    
    # Constraint A: deg(s) - deg(t) >= 1 - 14*(1-b) - 14*(1-p)
    # p is the indicator that the path exists.
    # p <= x_su, p <= x_uv, p <= x_vt
    # p >= x_su + x_uv + x_vt - 2
    # We don't have p as a variable. We can substitute p?
    # No, p is a logical AND of x_su, x_uv, x_vt.
    # The constraint is:
    # If x_su=1, x_uv=1, x_vt=1, then (deg(s)-deg(t) >= 1 OR deg(s)-deg(t) <= -1).
    
    # We can use the big-M formulation directly on x?
    # deg(s) - deg(t) >= 1 - 14*(1-b) - 14*(1 - x_su) - 14*(1 - x_uv) - 14*(1 - x_vt)
    # If any x is 0, the RHS becomes very negative, relaxing the constraint.
    # If all x are 1, RHS is 1 - 14*(1-b) = 1 - 14 + 14b = 14b - 13.
    # If b=1, RHS = 1. deg(s)-deg(t) >= 1.
    # If b=0, RHS = -13. deg(s)-deg(t) >= -13. (Always true since min diff is -14)
    
    # Wait, if b=0, we want the OTHER constraint to be active.
    # Constraint B: -(deg(s) - deg(t)) >= 1 - 14*b - 14*(1 - x_su) - 14*(1 - x_uv) - 14*(1 - x_vt)
    # If all x=1, b=0: RHS = 1. -(deg(s)-deg(t)) >= 1 => deg(s)-deg(t) <= -1.
    
    # So we just need these two constraints per path!
    # No p variable needed!
    
    # Let's verify.
    # Case 1: Path exists (x_su=x_uv=x_vt=1).
    #   Constraint A: deg(s)-deg(t) >= 1 - 14(1-b) = 14b - 13.
    #   Constraint B: deg(t)-deg(s) >= 1 - 14b = 1 - 14b.
    #   If b=1: A: diff >= 1. B: -diff >= -13 => diff <= 13. (Consistent)
    #   If b=0: A: diff >= -13. B: -diff >= 1 => diff <= -1. (Consistent)
    #   So if path exists, diff is either >=1 or <=-1. i.e., diff != 0. Correct.
    
    # Case 2: Path does not exist (at least one x is 0).
    #   Suppose x_su=0.
    #   Constraint A: diff >= 1 - 14(1-b) - 14(1-0) - ... = 1 - 14 + 14b - 14 = 14b - 27.
    #   Min diff is -14. 14b - 27 is -27 (if b=0) or -13 (if b=1).
    #   So diff >= -13 or -27. Always true.
    #   Constraint B: -diff >= 1 - 14b - 14 - ... = -13 - 14b.
    #   Max diff is 14. -14 >= -13 - 14b.
    #   If b=0: -14 >= -13 (False). Wait.
    #   Let's re-calculate.
    #   Constraint B: deg(t) - deg(s) >= 1 - 14b - 14(1-x_su) - 14(1-x_uv) - 14(1-x_vt).
    #   If x_su=0, x_uv=1, x_vt=1.
    #   RHS = 1 - 14b - 14 - 0 - 0 = -13 - 14b.
    #   We need deg(t) - deg(s) >= -13 - 14b.
    #   Min value of deg(t)-deg(s) is -14.
    #   If b=0: RHS = -13. -14 >= -13 is False.
    #   So if b=0, the constraint is violated if deg(t)-deg(s) = -14.
    #   But we can choose b=1 to relax it?
    #   If b=1: RHS = 1 - 14 - 14 = -27. -14 >= -27 is True.
    #   So the solver will choose b to satisfy the constraints.
    #   Since b is a free binary variable (except for its role in A and B),
    #   it will always be possible to set b such that the constraints are satisfied
    #   when the path does not exist.
    
    # So the formulation is correct.
    
    # Number of constraints: 2 * num_paths.
    # Variables: m + num_paths.
    
    # Let's build the sparse matrix.
    # Use scipy.sparse for efficiency.
    from scipy.sparse import lil_matrix, csr_matrix
    
    A = lil_matrix((2 * num_paths, total_vars))
    lb = np.zeros(2 * num_paths)
    ub = np.zeros(2 * num_paths)
    
    for pid, (s, t, idx_su, idx_uv, idx_vt) in enumerate(path_data):
        b_idx = m + pid
        
        # Constraint A: deg(s) - deg(t) >= 1 - 14*(1-b) - 14*(1-x_su) - 14*(1-x_uv) - 14*(1-x_vt)
        # deg(s) = sum_{e~s} x_e
        # deg(t) = sum_{e~t} x_e
        # LHS: deg(s) - deg(t)
        # We need to add coefficients for x_e.
        # For each edge e incident to s, coeff += 1.
        # For each edge e incident to t, coeff -= 1.
        
        # Let's precompute incidence.
        # Instead of looping all edges for each path, we can just add the specific edges?
        # No, deg(s) depends on ALL edges incident to s.
        
        # This is O(m) per constraint. 35k * 105 ~ 3.7M operations. Acceptable.
        
        # Initialize row with 0s
        row_A = A[2*pid].toarray()
        row_B = A[2*pid+1].toarray()
        
        # Add deg(s) to row_A
        for e_idx in range(m):
            u, v = edge_list[e_idx]
            if u == s or v == s:
                row_A[e_idx] += 1
            if u == t or v == t:
                row_A[e_idx] -= 1
        
        # Add -deg(s) + deg(t) to row_B (which is -(deg(s)-deg(t)))
        row_B[:] = -row_A[:]
        
        # Add b terms
        # A: ... + 14*b
        row_A[b_idx] += 14
        # B: ... - 14*b
        row_B[b_idx] -= 14
        
        # Add x_su, x_uv, x_vt terms
        # A: ... + 14*x_su + 14*x_uv + 14*x_vt
        row_A[idx_su] += 14
        row_A[idx_uv] += 14
        row_A[idx_vt] += 14
        
        # B: ... + 14*x_su + 14*x_uv + 14*x_vt
        row_B[idx_su] += 14
        row_B[idx_uv] += 14
        row_B[idx_vt] += 14
        
        # Set bounds
        # A: deg(s)-deg(t) - 14b - 14x_su - 14x_uv - 14x_vt >= -13
        # Wait, let's re-derive the constant.
        # Original: deg(s) - deg(t) >= 1 - 14(1-b) - 14(1-x_su) - 14(1-x_uv) - 14(1-x_vt)
        # deg(s) - deg(t) >= 1 - 14 + 14b - 14 + 14x_su - 14 + 14x_uv - 14 + 14x_vt
        # deg(s) - deg(t) >= 14b + 14x_su + 14x_uv + 14x_vt - 41
        # deg(s) - deg(t) - 14b - 14x_su - 14x_uv - 14x_vt >= -41
        
        # So LHS in matrix is: deg(s)-deg(t) - 14b - 14x_su...
        # My row_A has: deg(s)-deg(t) + 14b + 14x_su...
        # So I need to flip signs for b and x in the matrix?
        # No, the matrix A * x >= lb.
        # If I put +14 in the matrix for b, and lb is -41, it means:
        # ... + 14b >= -41.
        # But I want: ... - 14b >= -41.
        
        # Let's restart the row construction with correct signs.
        
        row_A = np.zeros(total_vars)
        row_B = np.zeros(total_vars)
        
        # deg(s) - deg(t)
        for e_idx in range(m):
            u, v = edge_list[e_idx]
            if u == s or v == s:
                row_A[e_idx] += 1
                row_B[e_idx] -= 1
            if u == t or v == t:
                row_A[e_idx] -= 1
                row_B[e_idx] += 1
        
        # -14b
        row_A[b_idx] -= 14
        row_B[b_idx] += 14
        
        # -14x_su
        row_A[idx_su] -= 14
        row_B[idx_su] += 14
        row_A[idx_uv] -= 14
        row_B[idx_uv] += 14
        row_A[idx_vt] -= 14
        row_B[idx_vt] += 14
        
        A[2*pid] = row_A
        A[2*pid+1] = row_B
        
        lb[2*pid] = -41
        ub[2*pid] = np.inf
        lb[2*pid+1] = -41
        ub[2*pid+1] = np.inf
    
    A = csr_matrix(A)
    
    # Bounds for variables
    # x_e: [0, 1]
    # b_path: [0, 1]
    bounds = Bounds(lb=np.zeros(total_vars), ub=np.ones(total_vars))
    
    # Integrality
    integrality = np.ones(total_vars)
    
    # Solve
    try:
        res = milp(c=c_obj, constraints=LinearConstraint(A, lb, ub), bounds=bounds, integrality=integrality, options={"time_limit": 50})
        
        if res.status == 0:
            x_sol = res.x[:m]
            edges = []
            for i in range(m):
                if x_sol[i] > 0.5:
                    edges.append(list(edge_list[i]))
            
            # Verify validity
            if len(edges) > 0:
                # Quick check
                deg = [0]*n
                for u,v in edges:
                    deg[u]+=1; deg[v]+=1
                adj = [set() for _ in range(n)]
                for u,v in edges:
                    adj[u].add(v); adj[v].add(u)
                
                valid = True
                for u in range(n):
                    for v in adj[u]:
                        if v < u: continue # avoid double check? No, path direction matters?
                        # Path s-u-v-t.
                        # Just check all.
                        deg_s_set = set()
                        for s in adj[u]:
                            if s != v:
                                deg_s_set.add(deg[s])
                        if not deg_s_set: continue
                        for t in adj[v]:
                            if t != u:
                                if deg[t] in deg_s_set:
                                    valid = False
                                    break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    return edges
                else:
                    # Fallback to star
                    return [[0, i] for i in range(1, n)]
            else:
                return [[0, i] for i in range(1, n)]
        else:
            # Fallback
            return [[0, i] for i in range(1, n)]
            
    except Exception as e:
        # Fallback
        return [[0, i] for i in range(1, n)]

