"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find edge-disjoint path covers for three fixed random connected graphs.
    Uses minimum-cost flow formulation:
    - Split each undirected edge into two directed arcs (capacity 1, cost 1)
    - Node balances: +1 for odd-degree vertices (path starts), -1 for odd-degree (path ends), 0 for even-degree
    - Solve LP to minimize total flow (which equals number of paths)
    """
    from random import Random
    import numpy as np
    from scipy.optimize import linprog

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_path_cover(n, edges):
        """
        Solve minimum edge-disjoint path cover using min-cost flow.
        
        For each undirected edge (u,v), create two arcs: u->v and v->u, each with capacity 1.
        The flow conservation constraints ensure that at each node, the net flow in/out
        matches the required balance. For path cover:
        - Nodes with odd degree must be endpoints of paths (balance ±1)
        - Nodes with even degree have balance 0
        
        The total number of paths = (number of odd-degree nodes) / 2.
        But we need to find the actual paths.
        
        Actually, the min-cost flow will give us a set of arcs that form edge-disjoint paths.
        The cost is the number of arcs used, which should equal the number of edges.
        The number of paths is determined by the structure.
        
        Let me think more carefully. In a path cover:
        - Each edge is used exactly once
        - The subgraph formed by the used arcs consists of disjoint paths
        
        For min-cost flow:
        - We have a directed graph with arcs for both directions of each edge
        - Capacity 1 on each arc
        - Cost 1 on each arc (we want to use exactly the right number of arcs)
        
        The key insight: we need to select a subset of arcs (one direction per edge) such that:
        - Each edge is covered exactly once (exactly one of its two arcs is selected)
        - The selected arcs form a collection of edge-disjoint paths
        
        This is equivalent to: find a flow where each edge has exactly one unit of flow
        (in one direction or the other), and the flow decomposes into paths.
        
        For the flow formulation:
        - Let x_{uv} be 1 if arc u->v is used, 0 otherwise
        - For each edge {u,v}: x_{uv} + x_{vu} = 1
        - Flow conservation: for each node v, sum_{u} x_{uv} - sum_{u} x_{vu} = b_v
          where b_v = 1 if v is a path start, -1 if path end, 0 otherwise
        
        The number of paths = number of nodes with b_v = 1 = number of nodes with b_v = -1
        
        To minimize the number of paths, we want to maximize the number of nodes with b_v = 0.
        But b_v is determined by the parity of the degree... wait, no.
        
        Actually, in any path cover of an undirected graph:
        - The number of paths is at least (number of odd-degree vertices) / 2
        - This lower bound is achievable if and only if the graph has a certain structure
        
        Gallai's conjecture says that for any connected graph with n vertices and maximum degree Δ,
        the edges can be covered by at most (n-1)/2 paths... no, that's not right either.
        
        Let me just solve the LP directly. We'll set up:
        - Variables: x_{uv} for each directed arc (u,v) where {u,v} is an edge
        - Constraints:
          1. For each edge {u,v}: x_{uv} + x_{vu} = 1
          2. For each node v: sum_{u adjacent to v} x_{uv} - sum_{u adjacent to v} x_{vu} = b_v
             where b_v is chosen to minimize the number of paths
        
        The number of paths = (1/2) * sum_v max(0, b_v) = (1/2) * sum_v |b_v| / 2... 
        
        Actually, sum_v b_v = 0 (flow conservation globally), and the number of paths
        = number of nodes with positive b_v = number of nodes with negative b_v.
        
        To minimize the number of paths, we want to minimize the number of non-zero b_v's,
        or equivalently, maximize the number of zero b_v's.
        
        But b_v must satisfy: b_v ≡ deg(v) (mod 2) because:
        sum_{u} x_{uv} + sum_{u} x_{vu} = deg(v) (each edge incident to v contributes 1)
        sum_{u} x_{uv} - sum_{u} x_{vu} = b_v
        Adding: 2 * sum_{u} x_{uv} = deg(v) + b_v
        So deg(v) + b_v must be even, i.e., b_v ≡ deg(v) (mod 2).
        
        Also, |b_v| ≤ deg(v) and b_v has the same parity as deg(v).
        
        The minimum number of paths is at least (number of odd-degree vertices) / 2.
        
        For the LP, we can fix b_v = 1 for some odd-degree vertices, b_v = -1 for others,
        and b_v = 0 for even-degree vertices. The question is which odd-degree vertices
        get +1 and which get -1.
        
        Actually, for any assignment of b_v with b_v ≡ deg(v) mod 2, sum b_v = 0,
        and |b_v| ≤ deg(v), there exists a feasible flow if and only if certain conditions hold.
        
        Let's just try the simplest approach: set b_v = 1 for half the odd-degree vertices
        and b_v = -1 for the other half. But which half? We need to choose it to make
        the flow feasible.
        
        Alternative approach: use the fact that the minimum number of paths in an edge-disjoint
        path cover of a connected graph is at least max(1, (odd_degree_count)/2) and at most
        something related to the structure.
        
        Let me try a different approach: use Eulerian trail decomposition.
        
        If all degrees are even, the graph is Eulerian, so we can cover all edges with 1 path
        (if connected) or with a number of paths equal to the number of connected components.
        
        If there are odd-degree vertices, we can pair them up and add "virtual" edges to make
        all degrees even, find an Eulerian circuit, then remove the virtual edges to get paths.
        
        The number of paths will be the number of pairs of odd-degree vertices.
        
        To minimize the number of paths, we want to pair up the odd-degree vertices in a way
        that minimizes... well, the number of pairs is fixed at (number of odd vertices)/2.
        So the number of paths is always (number of odd vertices)/2 for a connected graph?
        
        No, that's not right. Consider a star graph with 4 vertices: center has degree 3,
        leaves have degree 1. Total 4 odd vertices, so 2 paths. Indeed, we can cover with 2 paths.
        
        Consider a path graph with 4 vertices: degrees are 1,2,2,1. Two odd vertices, so 1 path.
        Indeed, the whole thing is one path.
        
        So the minimum number of paths is exactly (number of odd-degree vertices) / 2 for any
        connected graph? Let me verify with a cycle: all degrees 2, zero odd vertices, so
        0 paths? But we need at least 1 path to cover the cycle.
        
        Ah, I see. If there are no odd-degree vertices, the minimum is 1 (for a connected graph).
        If there are 2k odd-degree vertices, the minimum is k.
        
        So the minimum number of paths is max(1, (number of odd-degree vertices) / 2).
        
        Wait, is that always achievable? I believe so, by the following argument:
        - If all degrees are even, the graph has an Eulerian circuit, which is one path (cycle).
          But a cycle is not a simple path in the usual sense... actually, a cycle can be viewed
          as a path that starts and ends at the same vertex. But the problem says "simple paths",
          which typically means no repeated vertices. A cycle with n vertices has n vertices,
          and as a path it would have n vertices with the first and last being the same, which
          is not simple.
        
        Hmm, let me re-read the problem. "Paths must be simple, use graph edges, be edge-disjoint
        and cover all edges."
        
        So paths must be simple (no repeated vertices). A cycle cannot be represented as a single
        simple path. We need to break it into at least 2 paths? No, a cycle of length n can be
        covered by 1 path if we allow the path to visit n distinct vertices... but a path from v1
        to vn in a cycle v1-v2-...-vn-v1 would need to go v1-v2-...-vn, which uses n-1 edges,
        leaving the edge vn-v1 uncovered. So we need at least 2 paths for a cycle.
        
        Wait, no. A path is a sequence of distinct vertices where consecutive vertices are adjacent.
        A cycle of length n: v1-v2-...-vn-v1. To cover all n edges with simple paths:
        - Path 1: v1-v2-...-vn (uses n-1 edges)
        - Path 2: just the edge vn-v1? But that's a path of length 1, which is fine.
        
        So a cycle needs 2 paths? No wait, we can do:
        - Path 1: v1-v2-...-vn (n-1 edges)
        - Path 2: vn-v1 (1 edge)
        
        But these two paths share no edges, and together cover all n edges. So 2 paths.
        
        But actually, can we do it with 1 path? A simple path cannot repeat vertices, so a path
        in a cycle can use at most n-1 edges (visiting all n vertices). So no, 1 path is not enough.
        
        So for a cycle, we need 2 paths. The number of odd-degree vertices is 0, so my formula
        max(1, 0/2) = 1 is wrong.
        
        Let me reconsider. The correct formula for the minimum number of edge-disjoint simple paths
        covering all edges of a connected graph is more complex.
        
        Actually, I think the right approach is to use the min-cost flow formulation directly.
        Let me set up the LP properly.
        
        Variables: x_{uv} for each directed arc (u,v) where {u,v} is an edge. x_{uv} ∈ {0,1}.
        
        Constraints:
        1. For each edge {u,v}: x_{uv} + x_{vu} = 1
        2. For each node v: sum_{u} x_{uv} - sum_{u} x_{vu} = b_v, where b_v is a variable.
        
        Objective: minimize sum_v max(0, b_v) = (1/2) sum_v |b_v|
        
        Subject to: b_v ≡ deg(v) (mod 2), sum_v b_v = 0, |b_v| ≤ deg(v).
        
        This is an integer program, but we can solve the LP relaxation and hope it's integral.
        
        Alternatively, since the number of odd-degree vertices is fixed, we know that
        sum_v |b_v| ≥ number of odd-degree vertices, and the minimum is achieved when
        b_v ∈ {-1, 0, 1} for all v, with b_v = 0 for even-degree v, and b_v ∈ {1, -1}
        for odd-degree v.
        
        So the minimum number of paths is (number of odd-degree vertices) / 2, UNLESS the
        graph is a cycle (or more generally, unless the flow cannot be decomposed into
        simple paths).
        
        I think for the purpose of this task, let me just implement the flow-based approach
        and extract the paths from the solution.
        """
        if not edges:
            return []
        
        # Build adjacency and degree info
        adj = {v: set() for v in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        degrees = {v: len(adj[v]) for v in range(n)}
        
        # Create list of directed arcs
        arcs = []
        for u, v in edges:
            arcs.append((u, v))
            arcs.append((v, u))
        
        m = len(arcs)  # number of directed arcs
        num_nodes = n
        
        # LP formulation:
        # Variables: x_0, ..., x_{m-1} for each arc, plus b_0, ..., b_{n-1} for node balances
        # Total variables: m + n
        # 
        # Constraints:
        # 1. For each edge {u,v} with arcs (u,v) at index i and (v,u) at index j:
        #    x_i + x_j = 1
        # 2. For each node v: sum_{arcs out of v} x_i - sum_{arcs into v} x_i - b_v = 0
        # 3. b_v >= -deg(v), b_v <= deg(v)
        # 4. b_v has same parity as deg(v) - handled by integrality
        # 
        # Objective: minimize sum_v |b_v| = minimize t where t >= b_v and t >= -b_v for each v
        # Actually, minimize sum_v max(0, b_v) + sum_v max(0, -b_v) = sum_v |b_v|
        # Let's use: minimize sum_v t_v where t_v >= b_v, t_v >= -b_v, t_v >= 0
        
        # Simpler: since we know the optimal b_v values are in {-1, 0, 1} for this problem,
        # let's just try all possible assignments... no, that's exponential.
        
        # Let's use the LP with absolute values handled by auxiliary variables.
        
        # Variables: x (m vars), b (n vars), t (n vars)
        # Total: m + 2*n variables
        
        # Objective: minimize sum(t)
        c = np.zeros(m + 2 * n)
        c[m + n:m + 2 * n] = 1.0  # coefficients for t variables
        
        # Equality constraints:
        # 1. For each edge: x_i + x_j = 1
        # 2. For each node: sum_out x - sum_in x - b_v = 0
        
        num_eq_constraints = len(edges) + n
        A_eq = np.zeros((num_eq_constraints, m + 2 * n))
        b_eq = np.zeros(num_eq_constraints)
        
        # Edge constraints
        arc_index = {}
        for idx, (u, v) in enumerate(arcs):
            arc_index[(u, v)] = idx
        
        for k, (u, v) in enumerate(edges):
            i = arc_index[(u, v)]
            j = arc_index[(v, u)]
            A_eq[k, i] = 1.0
            A_eq[k, j] = 1.0
            b_eq[k] = 1.0
        
        # Node balance constraints
        for v in range(n):
            row = len(edges) + v
            for idx, (a, b) in enumerate(arcs):
                if a == v:
                    A_eq[row, idx] = 1.0
                elif b == v:
                    A_eq[row, idx] = -1.0
            A_eq[row, m + v] = -1.0  # -b_v
            b_eq[row] = 0.0
        
        # Inequality constraints:
        # t_v >= b_v  =>  b_v - t_v <= 0
        # t_v >= -b_v  =>  -b_v - t_v <= 0
        # t_v >= 0  =>  -t_v <= 0
        # b_v >= -deg(v)  =>  -b_v <= deg(v)
        # b_v <= deg(v)  =>  b_v <= deg(v)
        
        num_ineq = 3 * n + 2 * n
        A_ub = np.zeros((num_ineq, m + 2 * n))
        b_ub = np.zeros(num_ineq)
        
        for v in range(n):
            # b_v - t_v <= 0
            row = v
            A_ub[row, m + v] = 1.0
            A_ub[row, m + n + v] = -1.0
            b_ub[row] = 0.0
            
            # -b_v - t_v <= 0
            row = n + v
            A_ub[row, m + v] = -1.0
            A_ub[row, m + n + v] = -1.0
            b_ub[row] = 0.0
            
            # -t_v <= 0
            row = 2 * n + v
            A_ub[row, m + n + v] = -1.0
            b_ub[row] = 0.0
            
            # -b_v <= deg(v)
            row = 3 * n + v
            A_ub[row, m + v] = -1.0
            b_ub[row] = degrees[v]
            
            # b_v <= deg(v)
            row = 3 * n + n + v
            A_ub[row, m + v] = 1.0
            b_ub[row] = degrees[v]
        
        # Bounds: 0 <= x_i <= 1, -inf <= b_v <= inf, 0 <= t_v <= inf
        bounds = [(0, 1)] * m + [(-np.inf, np.inf)] * n + [(0, np.inf)] * n
        
        # Solve LP
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs')
        
        if not result.success:
            # Fallback: return each edge as its own path
            return [[list(e)] for e in edges]
        
        # Extract solution
        x_vals = result.x[:m]
        b_vals = result.x[m:m + n]
        
        # Determine which arcs are selected (x > 0.5)
        selected_arcs = []
        for idx, (u, v) in enumerate(arcs):
            if x_vals[idx] > 0.5:
                selected_arcs.append((u, v))
        
        # Build path decomposition from selected arcs
        # Each node has out-degree and in-degree determined by selected arcs
        out_edges = {v: [] for v in range(n)}
        in_edges = {v: [] for v in range(n)}
        for u, v in selected_arcs:
            out_edges[u].append(v)
            in_edges[v].append(u)
        
        # Find paths: start from nodes with out-degree > in-degree (path starts)
        paths = []
        visited_arcs = set()
        
        for v in range(n):
            out_deg = len(out_edges[v])
            in_deg = len(in_edges[v])
            if out_deg > in_deg:
                # This is a path start
                current = v
                path = [v]
                while out_edges[current]:
                    # Pick an unvisited outgoing edge
                    next_v = None
                    for w in out_edges[current]:
                        if (current, w) not in visited_arcs:
                            next_v = w
                            break
                    if next_v is None:
                        break
                    visited_arcs.add((current, next_v))
                    path.append(next_v)
                    current = next_v
                paths.append(path)
        
        # Check if all arcs are covered
        if len(visited_arcs) != len(selected_arcs):
            # Some arcs not covered, try to find remaining paths
            for u, v in selected_arcs:
                if (u, v) not in visited_arcs:
                    # Find path starting from u
                    current = u
                    path = [u]
                    while True:
                        next_v = None
                        for w in out_edges[current]:
                            if (current, w) not in visited_arcs:
                                next_v = w
                                break
                        if next_v is None:
                            break
                        visited_arcs.add((current, next_v))
                        path.append(next_v)
                        current = next_v
                    paths.append(path)
        
        return paths
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = solve_path_cover(n, edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
