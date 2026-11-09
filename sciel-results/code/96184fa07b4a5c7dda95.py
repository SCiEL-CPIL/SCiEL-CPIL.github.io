"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def priority(instance):
    """
    Score each assembly task for the priority-based list scheduling.
    
    Strategy: Derive priority scores from the dual variables of an LP relaxation
    of the assembly line balancing problem. The LP minimizes the number of stations
    (or equivalently, the makespan), and the dual values indicate the marginal cost
    of each task's resource consumption, guiding the greedy assignment.
    
    If the LP solve fails or times out, fall back to a composite heuristic.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    if n == 0:
        return np.array([])
    
    # Build successor map and predecessor counts
    successors = [[] for _ in range(n)]
    pred_count = [0] * n
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        pred_count[i] = len(preds)
        for p in preds:
            successors[p].append(i)
    
    # Topological order using Kahn's algorithm
    in_degree = pred_count[:]
    q = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while q:
        u = q.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
    
    if len(topo_order) < n:
        # Cycle detected, fallback to times-based priority
        return times.copy()
    
    # Compute depth and height
    depth = [0.0] * n
    height = [0.0] * n
    
    for u in topo_order:
        for v in successors[u]:
            if depth[v] < depth[u] + times[u]:
                depth[v] = depth[u] + times[u]
    
    for u in reversed(topo_order):
        for v in successors[u]:
            if height[u] < times[u] + height[v]:
                height[u] = times[u] + height[v]
    
    # Try LP-based dual derivation
    try:
        lp_duals = _solve_lp_duals(n, times, predecessors, successors, cycle_time, deadline)
        if lp_duals is not None:
            # Normalize and return as priority (higher = better)
            max_dual = np.max(lp_duals)
            if max_dual > 1e-12:
                return lp_duals / max_dual
            return lp_duals
    except Exception:
        pass
    
    # Fallback: composite heuristic
    succ_count = [0] * n
    memo = [-1] * n
    
    def count_succs(u):
        if memo[u] >= 0:
            return memo[u]
        total = 0
        for v in successors[u]:
            total += 1 + count_succs(v)
        memo[u] = total
        return total
    
    for i in range(n):
        succ_count[i] = count_succs(i)
    
    max_time = np.max(times) if n > 0 else 1.0
    max_depth = np.max(depth) if n > 0 else 1.0
    max_height = np.max(height) if n > 0 else 1.0
    max_succ = np.max(succ_count) if n > 0 else 1.0
    
    if max_time == 0: max_time = 1.0
    if max_depth == 0: max_depth = 1.0
    if max_height == 0: max_height = 1.0
    if max_succ == 0: max_succ = 1.0
    
    score = (
        1.0 * (np.array(height) / max_height) +
        0.5 * (times / max_time) +
        0.3 * (np.array(succ_count) / max_succ)
    )
    
    if time.time() > deadline:
        return score
    
    return score


def _solve_lp_duals(n, times, predecessors, successors, cycle_time, deadline):
    """
    Solve an LP relaxation to obtain dual variables that serve as priority scores.
    
    Formulation:
    - Decision variables: x_i = 1 if task i is assigned to station 1 (binary, relaxed to [0,1])
    - We minimize the number of stations, but since that's hard, we use a surrogate:
      minimize sum_i (i's position weight) subject to precedence and capacity constraints.
    
    A simpler approach: solve a linear assignment-like problem where we assign each task
    to a "position" (station) and the objective is to minimize total weighted completion time.
    
    Actually, let's use a different formulation:
    - Let y_i be the start time of task i (continuous)
    - Minimize sum_i y_i (total flow time) subject to:
      * Precedence: y_v >= y_u + times[u] for all (u,v) in edges
      * Capacity: for each station k, sum of times of tasks in station k <= cycle_time
    
    This is complex. Instead, use a simpler LP:
    - Minimize sum_i t_i * z_i where z_i is a "lateness" variable
    - Subject to precedence constraints and z_i >= 0
    
    Even simpler: use the dual of a maximum flow / min cut formulation.
    
    For practical purposes, let's solve:
    minimize sum_i w_i * s_i
    where s_i is the station number (continuous relaxation)
    subject to:
      s_v >= s_u + 1  for each edge (u,v)  [precedence in station order]
      s_i >= 0
      sum of times of tasks in each "layer" <= cycle_time (this is hard)
    
    Let's use a different approach: solve for the critical path weights.
    
    Actually, the cleanest LP dual approach:
    Consider the problem of minimizing the makespan (number of stations * cycle_time).
    The LP relaxation with assignment variables x_{i,k} (task i in station k) is large.
    
    For n=30, let's use a compact formulation:
    - Variables: p_i = priority-like dual value for task i
    - We want p_i to reflect how "constrained" task i is
    
    Let's solve a standard LP:
    min sum_i p_i
    s.t.
      p_v - p_u >= times[u]  for all edges (u,v)  [p is a topological potential]
      p_i >= 0
      sum of times of tasks with p_i in [k*cycle_time, (k+1)*cycle_time) <= cycle_time (nonlinear, skip)
    
    This is getting complicated. Let's use a simpler surrogate:
    
    Solve: max sum_i alpha_i
    s.t.
      alpha_i + sum_{j in preds(i)} beta_j <= 1  for all i
      alpha_i >= 0, beta_j >= 0
    
    The dual of this gives us the priority scores.
    """
    # Simpler approach: use the longest path (height) as a proxy for dual values
    # but refine with an LP that captures resource contention.
    
    # For each task, compute how much it "blocks" other tasks.
    # Use a flow-based approach:
    
    # Build a network: source -> tasks -> sink
    # Task i has capacity = cycle_time - times[i] (remaining capacity after assigning i)
    # This doesn't quite work either.
    
    # Practical approach: solve a small LP to get shadow prices.
    # Variables: s_i = start station index (continuous)
    # Minimize: sum_i s_i * times[i]  (weighted completion time)
    # Subject to:
    #   s_v >= s_u + times[u]/cycle_time  for all edges (u,v)  (precedence with time scaling)
    #   s_i >= 0
    
    # The dual variables of the precedence constraints give us the priority scores.
    
    m = len(predecessors)
    # Count edges
    edge_count = sum(len(predecessors[i]) for i in range(n) if i < len(predecessors))
    
    if edge_count == 0:
        # No precedence, all tasks independent
        # Priority should be based on times (larger first)
        return times.copy()
    
    # LP: min c^T x  s.t. A_ub x <= b_ub, A_eq x = b_eq, bounds
    # x = [s_0, s_1, ..., s_{n-1}]
    # Objective: minimize sum_i times[i] * s_i
    c = times.copy()
    
    # Inequality constraints: s_v - s_u >= times[u]/cycle_time  =>  -s_v + s_u <= -times[u]/cycle_time
    # Collect all edges
    edges = []
    for i in range(n):
        if i < len(predecessors):
            for p in predecessors[i]:
                edges.append((p, i))
    
    if len(edges) == 0:
        return times.copy()
    
    A_ub = np.zeros((len(edges), n))
    b_ub = np.zeros(len(edges))
    
    for idx, (u, v) in enumerate(edges):
        A_ub[idx, u] = 1.0
        A_ub[idx, v] = -1.0
        b_ub[idx] = -times[u] / cycle_time
    
    # Bounds: s_i >= 0
    bounds = [(0, None)] * n
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if time.time() > deadline:
            return None
        
        if result.status != 0:
            return None
        
        # Get dual variables (shadow prices) for the inequality constraints
        # In scipy linprog, result.ineqlin.marginals gives the dual values
        # for A_ub x <= b_ub constraints
        
        duals = result.ineqlin.marginals
        
        # The dual variable for constraint (u,v): s_v - s_u >= times[u]/cycle_time
        # corresponds to how much the objective would increase if this constraint
        # were tightened. Higher dual = more critical edge.
        
        # Convert edge duals to node duals:
        # Node i's priority = sum of duals of edges leaving i + sum of duals of edges entering i
        # Actually, for a min problem with >= constraints, the dual values indicate
        # the marginal value of relaxing the constraint.
        
        # Let's compute node priority as the sum of absolute dual values of incident edges
        node_duals = np.zeros(n)
        for idx, (u, v) in enumerate(edges):
            d = abs(duals[idx])
            node_duals[u] += d
            node_duals[v] += d
        
        # Combine with height for a better score
        height = np.zeros(n)
        # Recompute height
        topo_order = []
        in_deg = [0] * n
        succ_map = [[] for _ in range(n)]
        for i in range(n):
            if i < len(predecessors):
                for p in predecessors[i]:
                    succ_map[p].append(i)
                    in_deg[i] += 1
        
        q2 = [i for i in range(n) if in_deg[i] == 0]
        while q2:
            u = q2.pop(0)
            topo_order.append(u)
            for v in succ_map[u]:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    q2.append(v)
        
        for u in reversed(topo_order):
            for v in succ_map[u]:
                if height[u] < times[u] + height[v]:
                    height[u] = times[u] + height[v]
        
        max_h = np.max(height) if n > 0 else 1.0
        if max_h == 0:
            max_h = 1.0
        
        max_nd = np.max(node_duals) if n > 0 else 1.0
        if max_nd == 0:
            max_nd = 1.0
        
        # Composite: dual-based + height-based
        score = 0.7 * (node_duals / max_nd) + 0.3 * (height / max_h)
        
        return score
    
    except Exception:
        return None
# EVOLVE-BLOCK-END
