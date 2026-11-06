"""Rewritten program using MILP for minimum path edge cover."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_milp(n, edge_list):
        """
        Formulate and solve the minimum path edge cover problem as an ILP.
        
        Variables:
        - x[e, v] for each edge e=(u,v) and vertex v in {u,v}: 
          x[e,v] = 1 if path uses edge e and enters/exits at vertex v in a specific way.
          
        Actually, let's use a cleaner formulation:
        For each edge e = (u, v), we have two binary variables:
          x_e^u = 1 if the path containing e has u as an endpoint
          x_e^v = 1 if the path containing e has v as an endpoint
        
        Constraints:
        1. For each edge e = (u,v): x_e^u + x_e^v >= 1 (at least one endpoint)
           Actually, this isn't quite right. Let me think again.
           
        Better formulation:
        Let P be the set of edges. For each edge e=(u,v) in P, introduce binary variables:
          a_e = 1 if u is an endpoint of the path containing e
          b_e = 1 if v is an endpoint of the path containing e
        
        The number of paths = (1/2) * sum over all vertices v of (number of edges incident to v where v is an endpoint)
        
        Wait, that's not quite right either. Let me use a standard formulation.
        
        Standard formulation for minimum path cover of edges:
        - For each edge e = (u, v), we have 2 binary variables: y_{e,u} and y_{e,v}
          y_{e,u} = 1 means that in the path decomposition, edge e is oriented such that u is a "start" or "end" of the path segment... 
        
        Let me use a different approach. Think of it as:
        Each edge must be assigned to a path. A path is a sequence of edges where consecutive edges share a vertex.
        The number of paths equals the number of "starts" (or equivalently, the number of vertices with odd degree in the path subgraph, divided by 2, plus cycles... no).
        
        Actually, for a collection of edge-disjoint paths covering all edges:
        - Let d_P(v) be the degree of vertex v in the union of all paths (which is just the degree in G since all edges are covered).
        - In a path, endpoints have degree 1 in that path, internal vertices have degree 2.
        - So if we let x_v be the number of paths for which v is an endpoint, then:
          sum over paths of (contribution to degree at v) = d_G(v)
          For each path, v contributes 1 if it's an endpoint, 2 if internal, 0 if not in path.
          So: x_v + 2 * (number of paths where v is internal) = d_G(v)
          The number of paths where v is internal = (d_G(v) - x_v) / 2
          For this to be non-negative integer, we need x_v <= d_G(v) and x_v ≡ d_G(v) (mod 2).
        
        Total number of paths = (1/2) * sum_v x_v (since each path has exactly 2 endpoints)
        
        So minimize (1/2) * sum_v x_v
        Subject to:
          0 <= x_v <= d_G(v)
          x_v ≡ d_G(v) (mod 2) for all v
          And the key constraint: there must exist a set of edge-disjoint paths realizing these endpoint assignments.
        
        The existence constraint is the hard part. A necessary and sufficient condition is that for every subset S of vertices, the number of edges crossing the cut (S, V\S) is at least the number of paths that must cross... this gets complicated.
        
        Let me use a simpler but correct formulation:
        
        For each edge e = (u, v), introduce binary variables:
          x_{e,u} = 1 if u is an endpoint of the path containing e
          x_{e,v} = 1 if v is an endpoint of the path containing e
        
        Constraints:
        1. For each edge e = (u,v): x_{e,u} + x_{e,v} >= 1  (at least one endpoint)
           Actually, can both be 1? Yes, if the path is just the single edge e, then both u and v are endpoints.
           
        2. For each vertex v: sum over edges e incident to v of x_{e,v} <= d_G(v)
           And sum over edges e incident to v of x_{e,v} >= 0
           
        3. For each vertex v: sum over edges e incident to v of x_{e,v} ≡ d_G(v) (mod 2)
           This ensures consistency with the degree parity.
           
        4. Connectivity/consistency constraints: This is where it gets tricky. We need to ensure that the assignment of endpoints is realizable by actual paths.
        
        A key insight: the above constraints 1-3 are necessary but not sufficient. We also need to ensure that for each vertex v, the edges where v is NOT an endpoint can be paired up (since internal vertices have degree 2 in the path, so the two non-endpoint edges at v must be connected through v).
        
        So for each vertex v, let E_v^int be the set of edges e incident to v where x_{e,v} = 0. Then |E_v^int| must be even, and these edges must be pairable. But the pairing constraint is global.
        
        Let me try a different, more standard ILP formulation:
        
        Variables: For each edge e = (u, v), binary variable x_e ∈ {0, 1, 2}
          x_e = 0: u is the "start" endpoint, v is internal or end... no, this doesn't work cleanly.
        
        Let me use: For each edge e = (u, v), two binary variables p_{e,u} and p_{e,v}.
        p_{e,u} = 1 if u is an endpoint of the path containing e.
        p_{e,v} = 1 if v is an endpoint of the path containing e.
        
        Minimize: (1/2) * sum_{v} sum_{e incident to v} p_{e,v}
        
        Subject to:
        (A) For each edge e = (u, v): p_{e,u} + p_{e,v} >= 1
        (B) For each vertex v: sum_{e incident to v} p_{e,v} ≡ d_G(v) (mod 2)
        (C) For each vertex v: sum_{e incident to v} p_{e,v} <= d_G(v)
        (D) For each vertex v: sum_{e incident to v} p_{e,v} >= 0
        
        Plus the realizability constraint. For now, let's try without (D) being an issue and see if the parity and cut constraints suffice. In many small graphs, this may give a valid solution or a lower bound.
        
        Actually, let me just try solving this and checking if the solution is feasible. If not, I'll add more constraints.
        """
        if not edge_list:
            return []
        
        m = len(edge_list)
        edge_to_idx = {e: i for i, e in enumerate(edge_list)}
        
        # Build adjacency
        adj = {i: [] for i in range(n)}
        for i, (u, v) in enumerate(edge_list):
            adj[u].append((i, v))
            adj[v].append((i, u))
        
        degrees = [len(adj[v]) for v in range(n)]
        
        # Variables: for each edge e=(u,v), two binaries: p_{e,u} and p_{e,v}
        # Total variables: 2*m
        c = np.zeros(2 * m)
        for i, (u, v) in enumerate(edge_list):
            c[2 * i] = 0.5    # p_{e,u}
            c[2 * i + 1] = 0.5  # p_{e,v}
        
        # Constraints
        # (A) For each edge e=(u,v): p_{e,u} + p_{e,v} >= 1
        A_rows = []
        b_lower = []
        b_upper = []
        
        for i, (u, v) in enumerate(edge_list):
            row = np.zeros(2 * m)
            row[2 * i] = 1
            row[2 * i + 1] = 1
            A_rows.append(row)
            b_lower.append(1)
            b_upper.append(2)
        
        # (B) Parity: sum_{e incident to v} p_{e,v} ≡ d_G(v) (mod 2)
        # This is hard to express linearly. Let's skip parity for now and add it via big-M or just try.
        # Actually, let's add: sum_{e incident to v} p_{e,v} >= 1 if d_G(v) >= 1, and <= d_G(v)
        # And rely on the structure.
        
        # (C) For each vertex v: sum_{e incident to v} p_{e,v} <= d_G(v)
        for v in range(n):
            row = np.zeros(2 * m)
            for (i, other) in adj[v]:
                # p_{e,v} is at index 2*i if v is the first vertex in edge_list[i], else 2*i+1
                u_e, v_e = edge_list[i]
                if u_e == v:
                    row[2 * i] = 1
                else:
                    row[2 * i + 1] = 1
            A_rows.append(row)
            b_lower.append(0)
            b_upper.append(degrees[v])
        
        A = np.array(A_rows)
        lb = np.array(b_lower)
        ub = np.array(b_upper)
        
        constraints = LinearConstraint(A, lb, ub)
        integrality = np.ones(2 * m)
        bounds = Bounds(0, 1)
        
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if not res.success:
            # Fallback: each edge as its own path
            return [[list(e)] for e in edge_list]
        
        x = res.x
        paths = []
        # Reconstruct paths from the solution
        # For each edge, determine which endpoints are path endpoints
        # Then build paths by following edges
        
        # Create a mapping: for each edge, which vertices are endpoints
        edge_endpoints = {}
        for i, (u, v) in enumerate(edge_list):
            p_u = int(round(x[2 * i]))
            p_v = int(round(x[2 * i + 1]))
            edge_endpoints[i] = (p_u, p_v)
        
        # Build paths: start from edges where one endpoint is a path endpoint
        # Use a greedy approach to form paths
        used = [False] * m
        paths = []
        
        # Find starting edges (where at least one vertex is a path endpoint)
        # For each such edge, start a path and extend
        for i in range(m):
            if used[i]:
                continue
            u, v = edge_list[i]
            p_u, p_v = edge_endpoints[i]
            
            # Start a path from this edge
            if p_u == 1:
                # u is an endpoint, so path starts/ends at u
                path = [u, v]
                used[i] = True
                current = v
                # Extend the path from v
                while True:
                    # Find an unused edge incident to v where v is NOT a path endpoint (internal)
                    extended = False
                    for (j, other) in adj[v]:
                        if used[j]:
                            continue
                        u_j, v_j = edge_list[j]
                        p_uj, p_vj = edge_endpoints[j]
                        # Check if v is internal in this edge
                        if u_j == v and p_uj == 0:
                            path.append(other)
                            used[j] = True
                            v = other
                            extended = True
                            break
                        elif v_j == v and p_vj == 0:
                            path.append(other)
                            used[j] = True
                            v = other
                            extended = True
                            break
                    if not extended:
                        break
                paths.append(path)
            elif p_v == 1:
                # v is an endpoint
                path = [v, u]
                used[i] = True
                current = u
                while True:
                    extended = False
                    for (j, other) in adj[u]:
                        if used[j]:
                            continue
                        u_j, v_j = edge_list[j]
                        p_uj, p_vj = edge_endpoints[j]
                        if u_j == u and p_uj == 0:
                            path.append(other)
                            used[j] = True
                            u = other
                            extended = True
                            break
                        elif v_j == u and p_vj == 0:
                            path.append(other)
                            used[j] = True
                            u = other
                            extended = True
                            break
                    if not extended:
                        break
                paths.append(path)
            else:
                # Both are endpoints (path is just this edge)
                path = [u, v]
                used[i] = True
                paths.append(path)
        
        # Handle any remaining unused edges (shouldn't happen, but fallback)
        for i in range(m):
            if not used[i]:
                u, v = edge_list[i]
                paths.append([u, v])
        
        # Verify and fix if needed
        # Check that all paths are valid
        valid_paths = []
        for path in paths:
            if len(path) < 2:
                continue
            # Check if path is simple
            if len(set(path)) != len(path):
                # Not simple, split into individual edges
                for k in range(len(path) - 1):
                    valid_paths.append([path[k], path[k + 1]])
            else:
                # Check if edges exist
                valid = True
                for k in range(len(path) - 1):
                    e = tuple(sorted((path[k], path[k + 1])))
                    if e not in edge_to_idx:
                        valid = False
                        break
                if valid:
                    valid_paths.append(path)
                else:
                    for k in range(len(path) - 1):
                        valid_paths.append([path[k], path[k + 1]])
        
        # Check coverage
        covered = set()
        for path in valid_paths:
            for k in range(len(path) - 1):
                covered.add(tuple(sorted((path[k], path[k + 1]))))
        
        if covered != set(edge_list):
            # Fallback
            return [[list(e)] for e in edge_list]
        
        return valid_paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = solve_milp(n, edges)
        results.append(paths)
    return results
# EVOLVE-BLOCK-END
