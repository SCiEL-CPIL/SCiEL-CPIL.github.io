"""Path decomposition using MILP via scipy.optimize.milp with greedy fallback."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback: each edge as its own path
            edges = _instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = _instance(*spec)
        result = _decompose_milp(edges, deadline)
        if result is None:
            # Fallback to greedy
            result = _decompose_greedy(edges, deadline)
        if result is None:
            # Ultimate fallback
            result = [list(e) for e in edges]
        results.append(result)
    
    return results


def _instance(n, p, seed):
    """Generate the instance graph exactly as in the verifier."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose_milp(edges, deadline):
    """
    Solve the minimum path decomposition using MILP.
    
    Formulation:
    - For each directed edge (u,v), binary variable x_{u,v} indicating if used in path.
    - Each undirected edge {u,v} must be covered: x_{u,v} + x_{v,u} = 1.
    - Flow conservation: for each vertex v, sum_in - sum_out = b_v where b_v in {-1, 0, 1}.
    - b_v = 1 means v is a path start, b_v = -1 means path end, b_v = 0 means internal.
    - Minimize sum of b_v over all v (which equals 2 * number of paths).
    - Actually, number of paths = sum of b_v / 2 where b_v = 1.
    - To minimize paths, minimize sum of b_v (since each path has exactly one start and one end).
    
    Simplified: minimize sum_v b_v where b_v in {-1, 0, 1} and sum_v b_v = 0.
    Number of paths = (number of vertices with b_v = 1) = (1/2) * sum_v |b_v|.
    So minimize sum_v |b_v|, or equivalently, minimize the number of vertices with b_v != 0.
    
    For MILP, we can use: minimize sum_v (b_v + 1) where we enforce b_v in {-1,0,1}
    and sum b_v = 0. But this is tricky. 
    
    Alternative: Let s_v = 1 if v is a start, e_v = 1 if v is an end.
    Then for each v: (sum_in) - (sum_out) = s_v - e_v.
    Minimize sum_v (s_v + e_v) / 2 = sum_v s_v (since total starts = total ends).
    
    Actually, let's use: for each vertex v, net_flow_v = inflow - outflow.
    net_flow_v in {-1, 0, 1}.
    Number of paths = sum_v max(0, net_flow_v) = sum_v min(0, -net_flow_v).
    Minimize sum_v max(0, net_flow_v).
    
    For MILP: introduce auxiliary variables t_v >= 0, t_v >= net_flow_v, minimize sum t_v.
    But net_flow_v can be negative, so t_v >= net_flow_v and t_v >= 0.
    This works since if net_flow_v > 0, t_v >= net_flow_v; if net_flow_v <= 0, t_v >= 0 is enough.
    """
    n = max(max(u, v) for u, v in edges) + 1 if edges else 0
    if n == 0:
        return []
    
    edge_list = list(edges)
    m = len(edge_list)
    if m == 0:
        return []
    
    # Build directed edge indices
    # For each undirected edge {u,v}, we have two directed arcs: (u,v) and (v,u)
    # Variable x[i] for directed arc i
    # 2*m variables for directed arcs
    # n variables for t_v (auxiliary)
    # Total: 2*m + n variables
    
    # Objective: minimize sum of t_v
    c = [0.0] * (2 * m) + [1.0] * n
    
    # Constraints:
    # 1. For each undirected edge {u,v}: x_{u,v} + x_{v,u} = 1
    # 2. Flow conservation with t_v: 
    #    For each vertex v: (inflow_v - outflow_v) - t_v <= 0
    #    And t_v >= 0 (handled by variable bounds)
    #    Also: inflow_v - outflow_v >= -1 (since net_flow >= -1)
    #    And: inflow_v - outflow_v <= 1 (since net_flow <= 1)
    #    
    #    Actually, let's think more carefully.
    #    net_flow_v = inflow_v - outflow_v
    #    We want net_flow_v in {-1, 0, 1}
    #    t_v = max(0, net_flow_v)
    #    So: t_v >= net_flow_v, t_v >= 0
    #    And: net_flow_v >= -1
    #    And: net_flow_v <= 1 (this follows from t_v >= net_flow_v and t_v being bounded)
    
    # Let's use a different approach. Since the graph is small, let's use a simpler formulation.
    # 
    # Alternative formulation using path variables:
    # This is complex. Let's stick with flow.
    
    # Variables: x[0..2m-1] for directed arcs, t[0..n-1] for auxiliary
    # 
    # Constraints:
    # A. Edge cover: for each undirected edge i, x[2i] + x[2i+1] = 1
    # B. For each vertex v: inflow_v - outflow_v - t_v <= 0  (t_v >= net_flow_v)
    # C. For each vertex v: -inflow_v + outflow_v - t_v' <= 0 where t_v' = max(0, -net_flow_v)
    #    But we only need to count starts or ends, not both.
    #    Number of paths = number of starts = sum of positive net_flow.
    #    So we just need t_v >= net_flow_v and t_v >= 0, minimize sum t_v.
    #    But we also need net_flow_v >= -1 to prevent negative values from being "free".
    #    Actually, if net_flow_v < 0, it means v is an end, which is fine.
    #    The constraint t_v >= net_flow_v with t_v >= 0 handles it: if net_flow_v = -1, t_v >= -1 is always true since t_v >= 0.
    #    But we need to ensure net_flow_v >= -1. Let's add that.
    # D. For each vertex v: inflow_v - outflow_v >= -1
    # E. For each vertex v: inflow_v - outflow_v <= 1
    
    # Let's build the constraint matrix.
    # Number of constraints: m (edge cover) + n (t_v >= net_flow) + n (net_flow >= -1) + n (net_flow <= 1)
    # = 4n + m
    
    num_constraints = m + 3 * n
    A = []
    b = []
    
    # Edge cover constraints: x[2i] + x[2i+1] = 1
    for i in range(m):
        row = [0.0] * (2 * m + n)
        row[2 * i] = 1.0
        row[2 * i + 1] = 1.0
        A.append(row)
        b.append(1.0)
    
    # For each vertex v: inflow_v - outflow_v - t_v <= 0
    # inflow_v = sum of x[arc] where arc ends at v
    # outflow_v = sum of x[arc] where arc starts at v
    for v in range(n):
        row = [0.0] * (2 * m + n)
        for i in range(m):
            u, w = edge_list[i]
            # Arc (u,w): index 2i, Arc (w,u): index 2i+1
            if w == v:  # arc (u,w) ends at v, so it's inflow to v
                row[2 * i] += 1.0
            if u == v:  # arc (w,u) starts at v... wait, arc (w,u) is from w to u
                # Arc (u,w) is from u to w. If u == v, then arc (u,w) starts at v, so outflow.
                # Arc (w,u) is from w to u. If w == v, then arc (w,u) starts at v, so outflow.
                pass
            # Let me redo this.
            # Undirected edge {u, w}. Directed arcs: (u,w) at index 2i, (w,u) at index 2i+1.
            # For vertex v:
            #   Inflow: arcs ending at v. (u,w) ends at w, (w,u) ends at u.
            #   So if w == v, arc (u,w) is inflow. If u == v, arc (w,u) is inflow.
            #   Outflow: arcs starting at v. (u,w) starts at u, (w,u) starts at w.
            #   So if u == v, arc (u,w) is outflow. If w == v, arc (w,u) is outflow.
            if w == v:
                row[2 * i] += 1.0   # inflow from (u,w)
                row[2 * i + 1] -= 1.0  # outflow from (w,u)
            if u == v:
                row[2 * i] -= 1.0   # outflow from (u,w)
                row[2 * i + 1] += 1.0  # inflow from (w,u)
        row[2 * m + v] = -1.0  # -t_v
        A.append(row)
        b.append(0.0)
    
    # For each vertex v: inflow_v - outflow_v >= -1  =>  -(inflow_v - outflow_v) <= 1
    for v in range(n):
        row = [0.0] * (2 * m + n)
        for i in range(m):
            u, w = edge_list[i]
            if w == v:
                row[2 * i] -= 1.0
                row[2 * i + 1] += 1.0
            if u == v:
                row[2 * i] += 1.0
                row[2 * i + 1] -= 1.0
        A.append(row)
        b.append(1.0)
    
    # For each vertex v: inflow_v - outflow_v <= 1
    for v in range(n):
        row = [0.0] * (2 * m + n)
        for i in range(m):
            u, w = edge_list[i]
            if w == v:
                row[2 * i] += 1.0
                row[2 * i + 1] -= 1.0
            if u == v:
                row[2 * i] -= 1.0
                row[2 * i + 1] += 1.0
        A.append(row)
        b.append(1.0)
    
    # Variable bounds
    # x variables: binary [0, 1]
    # t variables: [0, 1] (since net_flow in {-1,0,1}, t_v in {0,1})
    lb = [0.0] * (2 * m) + [0.0] * n
    ub = [1.0] * (2 * m) + [1.0] * n
    integrality = [1] * (2 * m) + [0] * n  # x are integer, t are continuous
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        constraint = LinearConstraint(A, lb=b, ub=b)
        bounds = Bounds(lb, ub)
        result = milp(c, constraints=constraint, integrality=integrality, bounds=bounds,
                      options={'time_limit': max(1, deadline - time.time())})
        if result.status == 0 and result.x is not None:
            # Extract paths from solution
            return _extract_paths(result.x[:2*m], edge_list, n)
    except Exception:
        pass
    
    return None


def _extract_paths(x, edge_list, n):
    """Extract paths from the MILP solution."""
    # x[2i] = 1 means directed arc (u,w) is used, x[2i+1] = 1 means (w,u) is used
    # Build directed graph
    adj = defaultdict(list)
    for i in range(len(edge_list)):
        u, w = edge_list[i]
        if x[2 * i] > 0.5:
            adj[u].append(w)
        if x[2 * i + 1] > 0.5:
            adj[w].append(u)
    
    # Find paths: start from vertices with out-degree 1 and in-degree 0 (or just out-degree > in-degree)
    in_deg = defaultdict(int)
    out_deg = defaultdict(int)
    for u in adj:
        for v in adj[u]:
            out_deg[u] += 1
            in_deg[v] += 1
    
    # Find starts: vertices with out_deg > in_deg
    starts = [v for v in range(n) if out_deg.get(v, 0) > in_deg.get(v, 0)]
    
    paths = []
    visited_edges = set()
    
    for start in starts:
        path = [start]
        current = start
        while True:
            # Find next vertex
            next_v = None
            for v in adj[current]:
                edge_key = (min(current, v), max(current, v))
                if edge_key not in visited_edges:
                    next_v = v
                    break
            if next_v is None:
                break
            path.append(next_v)
            visited_edges.add((min(current, next_v), max(current, next_v)))
            current = next_v
        if len(path) >= 2:
            paths.append(path)
    
    # Check if all edges are covered
    if len(visited_edges) != len(edge_list):
        return None
    
    return paths


def _decompose_greedy(edges, deadline):
    """Greedy fallback decomposition."""
    if not edges:
        return []
    
    remaining = set(edges)
    adj = defaultdict(set)
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
    
    degree = defaultdict(int)
    for u, v in remaining:
        degree[u] += 1
        degree[v] += 1
    
    paths = []
    rng = random.Random(42)
    
    while remaining:
        if time.time() > deadline:
            break
        
        # Find odd-degree vertices
        odd_vertices = [v for v in adj if degree.get(v, 0) % 2 == 1 and adj[v]]
        if odd_vertices:
            start = rng.choice(odd_vertices)
        else:
            candidates = [v for v in adj if adj[v]]
            if not candidates:
                break
            start = rng.choice(candidates)
        
        # Find a long path
        path = _find_path_greedy(start, adj, degree, rng, remaining)
        
        if path is None or len(path) < 2:
            if adj[start]:
                next_v = rng.choice(list(adj[start]))
                path = [start, next_v]
            else:
                e = next(iter(remaining))
                path = [e[0], e[1]]
        
        # Remove edges
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge = tuple(sorted((u, v)))
            if edge in remaining:
                remaining.discard(edge)
                adj[u].discard(v)
                adj[v].discard(u)
                degree[u] -= 1
                degree[v] -= 1
        
        for v in list(adj.keys()):
            if not adj[v]:
                del adj[v]
        
        paths.append(path)
    
    return paths


def _find_path_greedy(start, adj, degree, rng, remaining):
    """Find a long simple path greedily."""
    path = [start]
    visited = {start}
    
    while True:
        current = path[-1]
        neighbors = [nb for nb in adj[current] if nb not in visited]
        if not neighbors:
            break
        
        # Prefer neighbors with fewer unvisited neighbors (Warnsdorff)
        def score(nb):
            return sum(1 for x in adj[nb] if x not in visited)
        
        neighbors.sort(key=score)
        if rng.random() < 0.7:
            choice = rng.choice(neighbors[:min(3, len(neighbors))])
        else:
            choice = rng.choice(neighbors)
        
        path.append(choice)
        visited.add(choice)
    
    return path if len(path) >= 2 else None

# EVOLVE-BLOCK-END
