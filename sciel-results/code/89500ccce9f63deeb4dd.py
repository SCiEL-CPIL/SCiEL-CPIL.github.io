"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_ilp(edges, time_limit=10.0):
        """
        Exact solution using Integer Linear Programming (ILP).
        Variables: x[e, dir] = 1 if edge e is traversed in direction dir.
        Objective: Minimize number of path starts.
        Constraints: 
          1. Each edge is used exactly once (x[e,0] + x[e,1] = 1).
          2. Flow conservation: For each vertex v, 
             (Outgoing edges from v) - (Incoming edges to v) <= 1 (start)
             (Incoming edges to v) - (Outgoing edges from v) <= 1 (end)
             Actually, to minimize paths, we minimize starts.
             Let s_v = 1 if v is a start, 0 otherwise.
             Out_v - In_v = s_v - e_v, where e_v is end indicator.
             Since we want to minimize sum(s_v), and s_v, e_v >= 0,
             we can just enforce Out_v - In_v <= 1 and Out_v - In_v >= -1.
             Wait, if Out - In = 1, it's a start. If -1, it's an end. If 0, it's internal.
             The number of paths is the number of starts.
             So minimize sum(s_v) subject to Out_v - In_v = s_v - e_v.
             This is equivalent to minimizing sum(max(0, Out_v - In_v)).
             Since this is linear programming with integer variables, we can introduce
             auxiliary variables s_v and e_v.
             
             Let's refine the ILP formulation:
             For each undirected edge (u, v), we have two binary variables:
             x_{u,v} = 1 if u->v, 0 otherwise.
             x_{v,u} = 1 if v->u, 0 otherwise.
             Constraint: x_{u,v} + x_{v,u} = 1.
             
             For each vertex v:
             Out_v = sum_{u} x_{v,u}
             In_v = sum_{u} x_{u,v}
             
             Let s_v be a binary variable indicating if v is a start.
             Let e_v be a binary variable indicating if v is an end.
             
             Constraint: Out_v - In_v = s_v - e_v.
             Since Out_v - In_v can be -1, 0, or 1 (for a valid path decomposition where paths are simple and edge-disjoint, actually Out_v - In_v can be larger if multiple paths start/end at v? 
             No, in a path decomposition, each vertex can be a start of at most one path? 
             No, multiple paths can start at the same vertex. 
             However, Gallai's conjecture is about the minimum number of paths.
             If multiple paths start at v, s_v would need to be > 1.
             But we can model s_v as an integer >= 0.
             
             Objective: Minimize sum(s_v).
             
             Constraints:
             1. x_{u,v} + x_{v,u} = 1 for all edges (u,v).
             2. Out_v - In_v = s_v - e_v for all v.
             3. s_v, e_v >= 0 integers.
             4. x_{u,v} in {0, 1}.
             
             This is a valid ILP. The number of paths is sum(s_v).
        """
        if not edges:
            return []
        
        # Map vertices to indices
        vertices = sorted(set().union(*[set(e) for e in edges]))
        if not vertices:
            return []
        
        v_to_idx = {v: i for i, v in enumerate(vertices)}
        n_v = len(vertices)
        n_e = len(edges)
        
        # Variables:
        # 0 to 2*n_e - 1: x variables (2 per edge)
        # 2*n_e to 2*n_e + n_v - 1: s_v variables
        # 2*n_e + n_v to 2*n_e + 2*n_v - 1: e_v variables
        
        n_vars = 2 * n_e + 2 * n_v
        
        # Objective: Minimize sum(s_v)
        c = np.zeros(n_vars)
        for i in range(n_v):
            c[2 * n_e + i] = 1.0  # Coefficient for s_i
            
        # Constraints:
        # 1. x_{u,v} + x_{v,u} = 1 for each edge
        # 2. Out_v - In_v = s_v - e_v for each vertex
        
        A_rows = []
        b_lower = []
        b_upper = []
        
        # Edge constraints: x_{u,v} + x_{v,u} = 1
        for i, (u, v) in enumerate(edges):
            row = np.zeros(n_vars)
            row[2 * i] = 1.0      # x_{u,v}
            row[2 * i + 1] = 1.0  # x_{v,u}
            A_rows.append(row)
            b_lower.append(1.0)
            b_upper.append(1.0)
            
        # Vertex constraints: Out_v - In_v - s_v + e_v = 0
        for v in vertices:
            i_v = v_to_idx[v]
            row = np.zeros(n_vars)
            
            # Find all edges incident to v
            for i, (u, w) in enumerate(edges):
                if u == v:
                    # Edge (v, w). x_{v,w} is outgoing, x_{w,v} is incoming.
                    # In our variable mapping:
                    # x_{u,v} corresponds to direction u->v? 
                    # Let's define:
                    # var 2*i is 1 if edge i is directed from u to v (as defined in edges list (u,v))
                    # var 2*i+1 is 1 if edge i is directed from v to u
                    # So for edge (u,v) where u==v? No, u != v.
                    # If u == v (current vertex):
                    # Direction u->v means v->w? No, edge is (u,v). If u is the vertex, then u->v is OUT.
                    # So x_{u,v} (var 2*i) is OUT.
                    # x_{v,u} (var 2*i+1) is IN.
                    row[2 * i] = 1.0   # Out
                    row[2 * i + 1] = -1.0 # In
                elif w == v:
                    # Edge (u, v) where w==v. So edge is (u, v).
                    # Direction u->v is IN to v.
                    # Direction v->u is OUT from v.
                    row[2 * i] = -1.0  # In (u->v)
                    row[2 * i + 1] = 1.0 # Out (v->u)
                
            # s_v and e_v terms: -s_v + e_v = 0 => Out - In - s_v + e_v = 0
            row[2 * n_e + i_v] = -1.0  # -s_v
            row[2 * n_e + n_v + i_v] = 1.0 # +e_v
            
            A_rows.append(row)
            b_lower.append(0.0)
            b_upper.append(0.0)
            
        A = np.array(A_rows)
        
        # Bounds
        lb = np.zeros(n_vars)
        ub = np.ones(n_vars)
        
        # Integrality
        integrality = np.ones(n_vars)
        
        # Solve
        try:
            res = milp(
                c=c,
                constraints=LinearConstraint(A, b_lower, b_upper),
                integrality=integrality,
                bounds=Bounds(lb, ub)
            )
            
            if res.status != 0:
                # Fallback to heuristic if ILP fails
                return solve_heuristic(edges, time_limit=time_limit)
                
            # Extract solution
            # x variables are in res.x[0 : 2*n_e]
            used_directed = []
            for i, (u, v) in enumerate(edges):
                x_uv = res.x[2 * i]
                x_vu = res.x[2 * i + 1]
                
                # Due to floating point, check which is closer to 1
                if abs(x_uv - 1.0) < 1e-5:
                    used_directed.append((u, v))
                elif abs(x_vu - 1.0) < 1e-5:
                    used_directed.append((v, u))
                else:
                    # Should not happen
                    used_directed.append((u, v))
                    
            # Reconstruct paths
            next_vertex = {}
            for u, v in used_directed:
                next_vertex[u] = v
                
            dests = set(v for _, v in used_directed)
            starts = [u for u in vertices if u not in dests]
            
            paths = []
            for s in starts:
                if s in next_vertex:
                    path = [s]
                    cur = s
                    while cur in next_vertex:
                        cur = next_vertex[cur]
                        path.append(cur)
                    paths.append(path)
                    
            # Verify all edges are covered
            covered_edges = set()
            for p in paths:
                for j in range(len(p) - 1):
                    u, v = p[j], p[j+1]
                    covered_edges.add(tuple(sorted((u, v))))
                    
            if covered_edges != set(edges):
                # Something went wrong, fallback
                return solve_heuristic(edges, time_limit=time_limit)
                
            return paths
            
        except Exception:
            return solve_heuristic(edges, time_limit=time_limit)

    def solve_heuristic(edges, time_limit=10.0):
        """
        Fallback heuristic: Greedy Warnsdorff.
        """
        if not edges:
            return []
        
        edge_set = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        best_paths = None
        best_count = float('inf')
        start_time = time.time()
        
        # Try multiple restarts
        for restart in range(50):
            if time.time() > start_time + time_limit:
                break
                
            rng = random.Random(restart * 7 + 13)
            current_edges = set(edge_set)
            paths = []
            
            while current_edges:
                degs = defaultdict(int)
                for u, v in current_edges:
                    degs[u] += 1
                    degs[v] += 1
                    
                if not degs:
                    break
                    
                max_deg = max(degs.values())
                candidates = [v for v, d in degs.items() if d == max_deg]
                start_v = rng.choice(candidates)
                
                path = [start_v]
                visited = {start_v}
                last_v = start_v
                
                while True:
                    neighbors = []
                    for nb in adj[last_v]:
                        if nb not in visited:
                            e = tuple(sorted((last_v, nb)))
                            if e in current_edges:
                                onward = 0
                                for nn in adj[nb]:
                                    if nn not in visited and nn != last_v:
                                        e2 = tuple(sorted((nb, nn)))
                                        if e2 in current_edges:
                                            onward += 1
                                neighbors.append((onward, nb))
                    
                    if not neighbors:
                        break
                        
                    neighbors.sort(key=lambda x: x[0])
                    min_onward = neighbors[0][0]
                    best_nbs = [nb for on, nb in neighbors if on == min_onward]
                    next_v = rng.choice(best_nbs)
                    
                    path.append(next_v)
                    visited.add(next_v)
                    e = tuple(sorted((last_v, next_v)))
                    current_edges.discard(e)
                    last_v = next_v
                    
                if len(path) > 1:
                    paths.append(path)
                else:
                    for e in list(current_edges):
                        paths.append(list(e))
                    current_edges = set()
                    break
                    
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [p[:] for p in paths]
                
        if best_paths is None:
            return [[u, v] for u, v in edges]
            
        return best_paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for i, spec in enumerate(specs):
        g_edges = instance(*spec)
        time_left = deadline - time.time()
        if time_left < 10:
            # Emergency: use very fast heuristic
            paths = solve_heuristic(g_edges, time_limit=1.0)
        else:
            # Try ILP with a time limit
            # ILP can be slow, so we give it a chunk of the remaining time
            tl = min(30.0, time_left / 2.0)
            paths = solve_ilp(g_edges, time_limit=tl)
        results.append(paths)
        
    return results

# EVOLVE-BLOCK-END
