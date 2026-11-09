"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def path_cover_lp(edges, time_limit):
        """
        Model edge-disjoint path cover as a min-cost flow / LP problem.
        Each undirected edge (u,v) becomes two directed arcs (u->v) and (v->u) with cap 1, cost 1.
        Node balance: out - in = b(v) where b(v) = 1 for one endpoint, -1 for other, 0 otherwise.
        To minimize number of paths, we want to maximize the number of "internal" vertices
        (those with in=1, out=1) and minimize endpoints.
        
        Actually, a cleaner formulation:
        - For each undirected edge {u,v}, we choose at most one direction.
        - Let x_uv = 1 if we direct edge u->v, x_vu = 1 if v->u. x_uv + x_vu <= 1.
        - For each vertex v: outdeg(v) - indeg(v) = b(v) where b(v) is in {-1, 0, 1}.
        - We want to minimize sum of max(outdeg(v), 1) over all v... no that's not right.
        
        Better: The number of paths = (number of vertices with outdeg > indeg) ... no.
        Actually: number of paths = sum_v max(0, outdeg(v) - indeg(v)) = sum_v max(0, indeg(v) - outdeg(v)).
        Wait, that's not quite right either. Let me think.
        
        In a directed path decomposition:
        - Each path has one start (outdeg > indeg by 1) and one end (indeg > outdeg by 1), 
          or it's a cycle (all balanced).
        - Number of paths = (number of starts) = (number of ends) = (1/2) * sum_v |outdeg(v) - indeg(v)|.
        
        So we want to minimize (1/2) * sum_v |outdeg(v) - indeg(v)|.
        
        This is a min-cost flow problem. Let's use scipy.optimize.linprog.
        """
        import numpy as np
        from scipy.optimize import linprog
        
        n = max(max(u, v) for u, v in edges) + 1 if edges else 0
        if not edges:
            return []
        
        m = len(edges)
        # Variables: x_{u,v} for each directed arc. 2*m variables.
        # x_i >= 0, x_i <= 1 for each arc i.
        # For each undirected edge {u,v}: x_{u,v} + x_{v,u} <= 1
        # For each vertex v: sum_out x - sum_in x = b_v, where we choose b_v in {-1, 0, 1}
        # But we can't fix b_v; we want to choose it to minimize total paths.
        
        # Alternative: minimize sum_v (outdeg(v) + indeg(v)) ... no.
        # Minimize sum_v max(outdeg(v)-indeg(v), 0) + max(indeg(v)-outdeg(v), 0)
        # = sum_v |outdeg(v) - indeg(v)|
        # = 2 * number of paths
        
        # Let d_v = outdeg(v) - indeg(v). We want to minimize sum_v |d_v|.
        # Introduce auxiliary variables t_v >= |d_v|, minimize sum t_v.
        # Constraints:
        #   t_v >= d_v, t_v >= -d_v for each v
        #   sum_out x - sum_in x = d_v for each v
        #   x_{u,v} + x_{v,u} <= 1 for each undirected edge
        #   0 <= x <= 1
        
        # Number of variables: 2*m (arcs) + n (d_v) + n (t_v) = 2m + 2n
        # Actually, we can eliminate d_v: d_v = sum_out x - sum_in x
        # So variables: 2*m + n (for t_v)
        
        num_vars = 2 * m + n
        
        # Objective: minimize sum t_v
        c = np.zeros(num_vars)
        for v in range(n):
            c[2 * m + v] = 1.0  # t_v coefficient
        
        # Constraints:
        # 1) For each undirected edge i = (u, v): x_{u,v} + x_{v,u} <= 1
        # 2) For each vertex v: t_v >= outdeg(v) - indeg(v)  =>  t_v - outdeg(v) + indeg(v) >= 0
        #    => -t_v + outdeg(v) - indeg(v) <= 0
        # 3) For each vertex v: t_v >= -(outdeg(v) - indeg(v))  =>  t_v + outdeg(v) - indeg(v) >= 0
        #    => -t_v - outdeg(v) + indeg(v) <= 0
        
        # Build A_ub, b_ub for linprog (minimize c^T x s.t. A_ub x <= b_ub)
        
        # Let's index arcs: arc 2*i is (u_i -> v_i), arc 2*i+1 is (v_i -> u_i)
        # where edges[i] = (u_i, v_i)
        
        A_ub_rows = []
        b_ub_vals = []
        
        # Constraint 1: for each edge, x_{u,v} + x_{v,u} <= 1
        for i in range(m):
            row = np.zeros(num_vars)
            row[2 * i] = 1.0
            row[2 * i + 1] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(1.0)
        
        # For each vertex v, we need to compute outdeg(v) - indeg(v) in terms of x variables.
        # outdeg(v) = sum of x_{v,w} for all arcs v->w
        # indeg(v) = sum of x_{w,v} for all arcs w->v
        
        # Let's precompute for each vertex v and each arc j, the coefficient of x_j in (outdeg(v) - indeg(v))
        # arc 2*i: u_i -> v_i, so contributes +1 to outdeg(u_i), +1 to indeg(v_i)
        # arc 2*i+1: v_i -> u_i, so contributes +1 to outdeg(v_i), +1 to indeg(u_i)
        
        # For vertex v, coeff in (outdeg - indeg):
        #   +1 if arc is v->w (v is tail)
        #   -1 if arc is w->v (v is head)
        
        for v in range(n):
            # Constraint: -t_v + (outdeg(v) - indeg(v)) <= 0
            row = np.zeros(num_vars)
            row[2 * m + v] = -1.0
            for i in range(m):
                u, w = edges[i]
                # arc 2*i: u->w
                if u == v:
                    row[2 * i] += 1.0
                if w == v:
                    row[2 * i] -= 1.0
                # arc 2*i+1: w->u
                if w == v:
                    row[2 * i + 1] += 1.0
                if u == v:
                    row[2 * i + 1] -= 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
            
            # Constraint: -t_v - (outdeg(v) - indeg(v)) <= 0
            row2 = np.zeros(num_vars)
            row2[2 * m + v] = -1.0
            for i in range(m):
                u, w = edges[i]
                # arc 2*i: u->w
                if u == v:
                    row2[2 * i] -= 1.0
                if w == v:
                    row2[2 * i] += 1.0
                # arc 2*i+1: w->u
                if w == v:
                    row2[2 * i + 1] -= 1.0
                if u == v:
                    row2[2 * i + 1] += 1.0
            A_ub_rows.append(row2)
            b_ub_vals.append(0.0)
        
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)
        
        # Bounds: 0 <= x <= 1 for arcs, t_v >= 0
        bounds = []
        for i in range(2 * m):
            bounds.append((0, 1))
        for v in range(n):
            bounds.append((0, None))
        
        # Solve LP
        try:
            if time.time() > time_limit:
                return [[u, v] for u, v in edges]
            
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if not res.success:
                return [[u, v] for u, v in edges]
            
            # Extract solution
            x = res.x
            # x[2*i] = flow on arc u_i->v_i, x[2*i+1] = flow on arc v_i->u_i
            # Due to LP relaxation, values may be fractional. But for this problem,
            # the constraint matrix is totally unimodular, so we should get integer solutions.
            
            # Build directed graph
            directed = []
            for i in range(m):
                u, v = edges[i]
                xv = x[2 * i]
                xu = x[2 * i + 1]
                if xv > 0.5:
                    directed.append((u, v))
                elif xu > 0.5:
                    directed.append((v, u))
                else:
                    # Fractional or zero; pick one arbitrarily
                    directed.append((u, v))
            
            # Now decompose directed graph into paths
            # Build adjacency
            adj = defaultdict(list)
            indeg = defaultdict(int)
            outdeg = defaultdict(int)
            for u, v in directed:
                adj[u].append(v)
                outdeg[u] += 1
                indeg[v] += 1
            
            # Find all paths
            # Start nodes: outdeg > indeg
            # End nodes: indeg > outdeg
            # Internal: outdeg == indeg
            paths = []
            visited_edges = set()
            
            # Find start nodes
            starts = []
            for v in range(n):
                if outdeg[v] > indeg[v]:
                    starts.append(v)
            
            # Also, isolated vertices with no edges don't matter.
            # If no starts, we might have cycles.
            
            # Trace paths from starts
            for start in starts:
                path = [start]
                cur = start
                while True:
                    found = False
                    for nbr in adj[cur]:
                        e = (cur, nbr)
                        if e not in visited_edges:
                            visited_edges.add(e)
                            path.append(nbr)
                            cur = nbr
                            found = True
                            break
                    if not found:
                        break
                paths.append(path)
            
            # Handle cycles (if any edges remain)
            remaining_edges = set(directed) - visited_edges
            while remaining_edges:
                # Pick an edge, follow the cycle
                start_edge = next(iter(remaining_edges))
                path = [start_edge[0], start_edge[1]]
                visited_edges.add(start_edge)
                remaining_edges.discard(start_edge)
                cur = start_edge[1]
                while cur != path[0]:
                    found = False
                    for nbr in adj[cur]:
                        e = (cur, nbr)
                        if e in remaining_edges:
                            remaining_edges.discard(e)
                            visited_edges.add(e)
                            path.append(nbr)
                            cur = nbr
                            found = True
                            break
                    if not found:
                        break
                paths.append(path)
            
            return paths
        
        except Exception:
            return [[u, v] for u, v in edges]

    def greedy_path_cover(edges, time_limit):
        """Fallback: repeatedly extract longest simple path from remaining edges."""
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        remaining = set(edges)
        paths = []
        
        while remaining:
            if time.time() > time_limit:
                for u, v in remaining:
                    paths.append([u, v])
                break
            
            best_path = None
            verts = list(adj.keys())
            verts.sort(key=lambda v: len(adj[v]), reverse=True)
            
            for start in verts:
                if not any((start, v) in remaining or (v, start) in remaining for v in adj[start]):
                    continue
                best = _dfs_longest(start, remaining, adj, time_limit)
                if best and (best_path is None or len(best) > len(best_path)):
                    best_path = best
                if best_path and len(best_path) > 2 * len(remaining) / max(1, len(verts)):
                    break
            
            if best_path is None or len(best_path) < 2:
                for u, v in remaining:
                    paths.append([u, v])
                break
            
            paths.append(best_path)
            for i in range(len(best_path) - 1):
                u, v = best_path[i], best_path[i + 1]
                e = tuple(sorted((u, v)))
                remaining.discard(e)
                adj[u].discard(v)
                adj[v].discard(u)
        
        return paths

    def _dfs_longest(start, remaining, adj, time_limit):
        best = [start]
        
        def dfs(node, path):
            nonlocal best
            if time.time() > time_limit:
                return
            if len(path) > len(best):
                best = path[:]
            for nbr in list(adj[node]):
                e = tuple(sorted((node, nbr)))
                if e in remaining and nbr not in path:
                    path.append(nbr)
                    dfs(nbr, path)
                    path.pop()
                    if time.time() > time_limit:
                        return
        
        dfs(start, [start])
        return best

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        elapsed = time.time() - t0
        remaining_time = DEADLINE - elapsed
        if remaining_time < 5:
            # Not enough time; use trivial
            results.append([[u, v] for u, v in edges])
            continue
        
        # Try LP approach first
        per_graph_limit = min(remaining_time - 2, 45.0)
        paths = path_cover_lp(edges, time.time() + per_graph_limit)
        
        # Validate: check that paths are valid (simple, use edges, edge-disjoint, cover all)
        if not _validate_paths(paths, edges):
            # Fallback to greedy
            paths = greedy_path_cover(edges, time.time() + min(remaining_time - 2, 45.0))
        
        results.append(paths)
    
    # Fill in any missing graphs
    while len(results) < 3:
        spec = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])
    
    return results[:3]

def _validate_paths(paths, edges):
    """Validate that paths form a valid edge-disjoint path cover."""
    if not paths:
        return len(edges) == 0
    
    edge_set = set(tuple(sorted(e)) for e in edges)
    used_edges = set()
    
    for path in paths:
        if len(path) < 2:
            return False
        # Check simple (no repeated vertices)
        if len(set(path)) != len(path):
            return False
        # Check edges exist
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            e = tuple(sorted((u, v)))
            if e not in edge_set:
                return False
            if e in used_edges:
                return False
            used_edges.add(e)
    
    # Check all edges covered
    if used_edges != edge_set:
        return False
    
    return True
# EVOLVE-BLOCK-END
