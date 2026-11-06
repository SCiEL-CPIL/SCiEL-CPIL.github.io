"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from random import Random
from scipy.optimize import milp, LinearConstraint, Bounds

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _decompose(edges, n):
    """
    Decompose a connected graph into the minimum number of edge-disjoint simple paths
    covering all edges. Uses a reduction to min-cost flow / matching on a bipartite
    graph of edge endpoints.
    
    The key insight: A path decomposition of a graph G into k paths is equivalent to
    finding a set of directed paths in an orientation of G such that:
    - Each edge is used exactly once (in one direction)
    - At each vertex, the number of path-starts minus path-ends is in {-1, 0, 1}
    - The total number of paths is the number of vertices where starts > ends (or equivalently, sum of positive imbalances)
    
    More precisely, we can model this as:
    1. Orient each edge (choose a direction)
    2. For each vertex v, let out(v) - in(v) = b(v) where b(v) in {-1, 0, 1}
    3. The number of paths is sum_{v: b(v)=1} 1
    
    We want to minimize the number of paths. This is equivalent to minimizing the number of vertices with b(v) = 1.
    
    A simpler approach: We can use the fact that a graph has a path decomposition into k paths iff 
    we can pair up "excess" endpoints. 
    
    Actually, let's use a cleaner formulation. Consider the line graph or use the following:
    
    A path decomposition into k paths means we partition the edges into k sets, each forming a simple path.
    
    Key theorem: A connected graph G can be decomposed into k edge-disjoint paths covering all edges
    if and only if we can assign a direction to each edge such that for each vertex v,
    |outdeg(v) - indeg(v)| <= 1, and the number of vertices with outdeg > indeg equals k.
    
    Wait, that's not quite right either. Let me think more carefully.
    
    Actually, here's a better approach using matching:
    
    Step 1: For each edge e = (u, v), we decide which endpoint is the "start" and which is the "end" of the path segment.
    This is equivalent to orienting the edge.
    
    Step 2: After orienting all edges, at each vertex v, we have some edges coming in and some going out.
    In a path decomposition, at each vertex, the incoming path segments must be paired with outgoing path segments,
    except for at most one unpaired incoming (end of a path) and at most one unpaired outgoing (start of a path).
    
    So for vertex v with d(v) incident edges, after orientation we have in(v) incoming and out(v) outgoing edges.
    We need |in(v) - out(v)| <= 1, and the number of paths is the number of vertices where out(v) > in(v)
    (which equals the number where in(v) > out(v), and their sum is the total number of path endpoints, i.e., 2 * number of paths).
    
    To minimize the number of paths, we want to minimize the number of vertices with imbalance.
    
    Let's use ILP:
    - For each edge e = (u, v), binary variable x_e = 1 if oriented u->v, 0 if v->u
    - For each vertex v, define b_v = outdeg(v) - indeg(v)
    - b_v = sum_{e incident to v} (1 if e is oriented away from v, -1 if toward v)
    - Constraint: -1 <= b_v <= 1
    - Objective: minimize (1/2) * sum_v |b_v| = minimize sum_v max(0, b_v)
    
    Since b_v is an integer in {-1, 0, 1}, |b_v| = b_v if b_v > 0, -b_v if b_v < 0, 0 if b_v = 0.
    And sum_v b_v = 0 always (each edge contributes +1 to one endpoint and -1 to the other).
    
    So minimizing sum_v |b_v| is equivalent to minimizing 2 * (number of vertices with b_v = 1) = 2 * k.
    
    Let's set up the ILP.
    """
    m = len(edges)
    if m == 0:
        return []
    
    # Build adjacency and edge-to-vertex mapping
    edge_list = edges
    # Map each edge to an index
    edge_idx = {e: i for i, e in enumerate(edge_list)}
    
    # For ILP: binary variable for each edge indicating orientation
    # x_i = 1 means edge i is oriented from edge_list[i][0] to edge_list[i][1]
    # x_i = 0 means edge i is oriented from edge_list[i][1] to edge_list[i][0]
    
    # For each vertex v, compute b_v as a linear function of x variables
    # b_v = sum over edges e incident to v of:
    #   if e = (a, b) with a < b:
    #     if v == a: contribution is +1 if x_e = 1 (a->b), -1 if x_e = 0 (b->a) => 2*x_e - 1
    #     if v == b: contribution is -1 if x_e = 1 (a->b), +1 if x_e = 0 (b->a) => 1 - 2*x_e = -(2*x_e - 1)
    
    # So for edge (a, b) with a < b:
    #   contribution to b_a is (2*x_e - 1)
    #   contribution to b_b is -(2*x_e - 1) = (1 - 2*x_e)
    
    # Let y_e = 2*x_e - 1, so y_e in {-1, 1}
    # b_v = sum_{e incident to v, v is smaller endpoint} y_e + sum_{e incident to v, v is larger endpoint} (-y_e)
    
    # We want to minimize (1/2) * sum_v |b_v| subject to -1 <= b_v <= 1 for all v.
    
    # Using y_e = 2*x_e - 1, y_e in {-1, 1}
    
    # Let's build the constraint matrix for b_v = c_v . y where c_v is a row vector
    # c_v[e] = 1 if e is incident to v and v is the smaller endpoint
    # c_v[e] = -1 if e is incident to v and v is the larger endpoint
    # c_v[e] = 0 otherwise
    
    # Objective: minimize sum_v t_v where t_v >= |b_v|
    # Since b_v in {-1, 0, 1}, |b_v| = b_v if b_v >= 0 else -b_v
    # t_v >= b_v, t_v >= -b_v, t_v >= 0
    
    # Variables: y_0, ..., y_{m-1} (each in {-1, 1}), t_0, ..., t_{n-1} (each >= 0)
    # Total variables: m + n
    
    # Constraints:
    # 1. -1 <= b_v <= 1 for each v: i.e., -1 <= c_v . y <= 1
    # 2. t_v >= b_v = c_v . y for each v
    # 3. t_v >= -b_v = -c_v . y for each v
    # 4. t_v >= 0 for each v
    # 5. y_e in {-1, 1} for each e
    
    # Objective: minimize sum_v t_v
    
    # Let's build this as an MILP with scipy.optimize.milp
    
    # Variables: [y_0, ..., y_{m-1}, t_0, ..., t_{n-1}]
    # y_e are binary-like but in {-1, 1}. We can substitute y_e = 2*z_e - 1 where z_e in {0, 1}
    # Then b_v = c_v . (2*z - 1) = 2*c_v . z - c_v . ones
    # Let d_v = c_v . ones (constant), and let a_v = 2*c_v
    # b_v = a_v . z - d_v
    
    # Constraints become:
    # -1 <= a_v . z - d_v <= 1  =>  d_v - 1 <= a_v . z <= d_v + 1
    # t_v >= a_v . z - d_v
    # t_v >= -(a_v . z - d_v) = -a_v . z + d_v
    # t_v >= 0
    # z_e in {0, 1}
    
    # Objective: minimize sum_v t_v
    
    # Let's build the matrices
    # Number of z variables: m
    # Number of t variables: n
    # Total: m + n
    
    # A matrix for constraints, b for bounds
    # We'll use LinearConstraint for the inequality constraints
    
    # Let's collect all constraints:
    # For each vertex v:
    #   d_v - 1 <= a_v . z <= d_v + 1  (2 constraints per vertex, but can be combined)
    #   t_v - a_v . z >= -d_v  (rewriting t_v >= a_v . z - d_v)
    #   t_v + a_v . z >= d_v   (rewriting t_v >= -a_v . z + d_v)
    #   t_v >= 0
    
    # Total constraints: 2n (for the b_v bounds) + n (t_v >= b_v) + n (t_v >= -b_v) + n (t_v >= 0)
    # But t_v >= 0 is redundant if we have the other constraints since t_v >= |b_v| >= 0
    
    # Actually, let's just use:
    # For each v: 
    #   a_v . z - t_v <= d_v
    #   -a_v . z - t_v <= d_v
    #   t_v >= 0
    # And z_e in [0, 1]
    
    # Wait, t_v >= b_v means t_v >= a_v . z - d_v, i.e., t_v - a_v . z >= -d_v, i.e., a_v . z - t_v <= d_v
    # t_v >= -b_v means t_v >= -a_v . z + d_v, i.e., t_v + a_v . z >= d_v, i.e., -a_v . z - t_v <= -d_v
    
    # So constraints:
    # a_v . z - t_v <= d_v  for all v
    # -a_v . z - t_v <= -d_v  for all v
    # t_v >= 0 for all v
    # z_e in [0, 1] for all e
    
    # Let's build the A matrix for the form A @ vars <= b_ub
    # vars = [z_0, ..., z_{m-1}, t_0, ..., t_{n-1}]
    
    # First set of constraints: for each v, a_v . z - t_v <= d_v
    # Second set: for each v, -a_v . z - t_v <= -d_v
    
    A_rows = []
    b_ub = []
    
    for v in range(n):
        # Compute a_v and d_v
        a_v = np.zeros(m)
        d_v = 0
        for i, (u, w) in enumerate(edge_list):
            if v == u:
                # v is smaller endpoint (since u < w in our edge representation)
                a_v[i] = 2
                d_v += 1
            elif v == w:
                # v is larger endpoint
                a_v[i] = -2
                d_v += 1
        
        # Constraint: a_v . z - t_v <= d_v
        row1 = np.zeros(m + n)
        row1[:m] = a_v
        row1[m + v] = -1
        A_rows.append(row1)
        b_ub.append(d_v)
        
        # Constraint: -a_v . z - t_v <= -d_v
        row2 = np.zeros(m + n)
        row2[:m] = -a_v
        row2[m + v] = -1
        A_rows.append(row2)
        b_ub.append(-d_v)
    
    A = np.array(A_rows)
    b_ub = np.array(b_ub)
    
    # t_v >= 0 is handled by bounds on t variables
    # z_e in [0, 1]
    
    # Objective: minimize sum_v t_v
    c = np.zeros(m + n)
    c[m:] = 1  # t_v coefficients
    
    # Bounds
    # z_e in [0, 1]
    # t_v in [0, inf]
    lb = np.zeros(m + n)
    ub = np.full(m + n, np.inf)
    ub[:m] = 1  # z_e in [0, 1]
    
    # Integrality: z_e are integer (0 or 1), t_v can be continuous
    integrality = np.zeros(m + n)
    integrality[:m] = 1  # z_e are binary
    
    # Solve
    res = milp(c=c, constraints=LinearConstraint(A, -np.inf, b_ub), 
               integrality=integrality, bounds=Bounds(lb, ub))
    
    if not res.success:
        # Fallback: return each edge as a separate path
        return [[list(e)] for e in edges]
    
    # Extract solution
    z = res.x[:m]
    
    # Reconstruct paths
    # For each edge, determine its direction
    # edge_list[i] = (u, w) with u < w
    # If z[i] = 1, edge is oriented u -> w
    # If z[i] = 0, edge is oriented w -> u
    
    # Build directed graph
    from collections import defaultdict
    out_edges = defaultdict(list)
    in_edges = defaultdict(list)
    
    for i, (u, w) in enumerate(edge_list):
        if z[i] > 0.5:
            # u -> w
            out_edges[u].append((w, i))
            in_edges[w].append((u, i))
        else:
            # w -> u
            out_edges[w].append((u, i))
            in_edges[u].append((w, i))
    
    # Now trace paths
    # Find all vertices with outdeg > indeg (these are path starts)
    # And all vertices with indeg > outdeg (these are path ends)
    
    paths = []
    visited_edges = set()
    
    # Find path starts: vertices where outdeg > indeg
    for v in range(n):
        outdeg = len(out_edges[v])
        indeg = len(in_edges[v])
        if outdeg > indeg:
            # Start a path from v
            if not out_edges[v]:
                continue
            # Follow the path
            path = [v]
            current = v
            while True:
                # Find an unvisited outgoing edge from current
                next_edge = None
                for (nxt, eidx) in out_edges[current]:
                    if eidx not in visited_edges:
                        next_edge = (nxt, eidx)
                        break
                if next_edge is None:
                    break
                nxt, eidx = next_edge
                visited_edges.add(eidx)
                path.append(nxt)
                current = nxt
            if len(path) >= 2:
                paths.append(path)
    
    # Check if all edges are covered
    if len(visited_edges) != m:
        # Fallback: add remaining edges as individual paths
        for i, (u, w) in enumerate(edge_list):
            if i not in visited_edges:
                paths.append([u, w])
    
    return paths

def construct():
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    for (n, p, seed) in specs:
        edges = _instance(n, p, seed)
        paths = _decompose(edges, n)
        results.append(paths)
    return results
# EVOLVE-BLOCK-END
