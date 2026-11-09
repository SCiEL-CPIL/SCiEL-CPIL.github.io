import time
import random
from collections import defaultdict, deque
from scipy.optimize import linprog
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def solve_with_flow(edges, seed):
        """
        Solve edge-disjoint path cover via min-cost flow.
        
        Key insight: An edge-disjoint path cover of a graph G can be found by:
        1. For each undirected edge, create two directed arcs (both directions)
        2. Assign flow of exactly 1 to each arc (covering each edge exactly once)
        3. At each vertex, conservation allows "through" flow (path continues)
           or "endpoint" flow (path starts/ends)
        4. Minimize the number of endpoints (each path has 2 endpoints)
        
        Formulation:
        - Variables: x_{u,v} in {0,1} for each directed arc (u->v)
        - Constraint: For each undirected edge {u,v}: x_{u,v} + x_{v,u} = 1
        - At each vertex v: sum_in(v) + s_v = sum_out(v) + t_v
          where s_v, t_v >= 0 are "source" and "sink" indicators
        - Minimize: sum_v (s_v + t_v)  [each path contributes 2 to this]
        
        Since the graph is small, we can solve this as an LP (relaxation is integral
        due to total unimodularity of network flow constraints).
        """
        if not edges:
            return []
        
        n = max(max(u, v) for u, v in edges) + 1
        
        # Build directed arcs
        arcs = []  # list of (u, v, edge_idx)
        arc_index = {}  # (u,v) -> index in arcs
        edge_to_arcs = {}  # edge_idx -> [arc_idx1, arc_idx2]
        
        for i, (u, v) in enumerate(edges):
            a1 = len(arcs)
            arcs.append((u, v, i))
            arc_index[(u, v)] = a1
            
            a2 = len(arcs)
            arcs.append((v, u, i))
            arc_index[(v, u)] = a2
            
            edge_to_arcs[i] = [a1, a2]
        
        m = len(arcs)  # number of directed arcs
        E = len(edges)
        
        # Variables: x_0..x_{m-1} for arc flows, then s_0..s_{n-1}, t_0..t_{n-1}
        # Total variables: m + 2*n
        # Objective: minimize sum(s_v) + sum(t_v) = number of endpoints / 2 paths
        # Actually each path has 2 endpoints, so num_paths = (sum s + sum t) / 2
        
        # Build constraint matrix
        # Constraint type 1: For each edge i, x_{u,v} + x_{v,u} = 1
        # Constraint type 2: For each vertex v, flow conservation with s,t
        
        num_constraints = E + n  # edge constraints + vertex conservation
        
        A_eq = np.zeros((num_constraints, m + 2 * n))
        b_eq = np.zeros(num_constraints)
        
        # Edge constraints: x_{u,v} + x_{v,u} = 1
        for i, (u, v) in enumerate(edges):
            a1, a2 = edge_to_arcs[i]
            A_eq[i, a1] = 1.0
            A_eq[i, a2] = 1.0
            b_eq[i] = 1.0
        
        # Vertex conservation: sum_in(v) - sum_out(v) + s_v - t_v = 0
        # i.e., sum_in(v) + s_v = sum_out(v) + t_v
        for v in range(n):
            row = E + v
            for arc_idx, (a, b, _) in enumerate(arcs):
                if a == v:  # outgoing from v
                    A_eq[row, arc_idx] = -1.0
                if b == v:  # incoming to v
                    A_eq[row, arc_idx] = 1.0
            # s_v coefficient is +1, t_v coefficient is -1
            A_eq[row, m + v] = 1.0       # s_v
            A_eq[row, m + n + v] = -1.0  # t_v
            b_eq[row] = 0.0
        
        # Objective: minimize sum(s_v) + sum(t_v)
        c = np.zeros(m + 2 * n)
        for v in range(n):
            c[m + v] = 1.0       # s_v
            c[m + n + v] = 1.0   # t_v
        
        # Bounds: all variables >= 0, x arcs <= 1
        bounds = [(0, 1)] * m + [(0, None)] * (2 * n)
        
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if not result.success:
            # Fallback: greedy
            return solve_greedy_fallback(edges, seed)
        
        # Extract path decomposition from flow
        # x_{u,v} = 1 means we use arc u->v
        # Build directed graph from flow
        flow_arcs = []
        for i, (u, v, edge_idx) in enumerate(arcs):
            if result.x[i] > 0.5:  # arc is used
                flow_arcs.append((u, v))
        
        # Now decompose into paths
        # Build adjacency
        adj = defaultdict(list)
        in_deg = defaultdict(int)
        out_deg = defaultdict(int)
        for u, v in flow_arcs:
            adj[u].append(v)
            out_deg[u] += 1
            in_deg[v] += 1
        
        # Find path starts (vertices with out_deg > in_deg) and ends (in_deg > out_deg)
        # Actually, from the flow: s_v > 0 means v is a path start, t_v > 0 means v is a path end
        paths = []
        visited_arcs = set()
        
        # Find starts: vertices where s_v > 0.5
        starts = []
        for v in range(n):
            if result.x[m + v] > 0.5:
                starts.append(v)
        
        # If no clear starts (shouldn't happen), find vertices with out_deg > in_deg
        if not starts:
            for v in range(n):
                if out_deg[v] > in_deg[v]:
                    starts.append(v)
        
        # Follow paths from each start
        for start in starts:
            if out_deg[start] == 0:
                continue
            path = [start]
            current = start
            while out_deg[current] > 0:
                # Find an unvisited outgoing arc
                found = False
                for neighbor in adj[current]:
                    arc_key = (current, neighbor)
                    if arc_key not in visited_arcs:
                        visited_arcs.add(arc_key)
                        path.append(neighbor)
                        current = neighbor
                        found = True
                        break
                if not found:
                    break
            if len(path) >= 2:
                paths.append(path)
        
        # Check if we covered all edges; if not, add remaining as single edges
        covered_edges = set()
        for path in paths:
            for i in range(len(path) - 1):
                covered_edges.add(tuple(sorted((path[i], path[i+1]))))
        
        all_edge_set = set(tuple(sorted(e)) for e in edges)
        remaining = all_edge_set - covered_edges
        
        for u, v in remaining:
            paths.append([u, v])
        
        return paths
    
    def solve_greedy_fallback(edges, seed):
        """Fallback greedy path decomposition."""
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        rng = random.Random(seed + 100)
        paths = []
        available = set(edge_set)
        
        while available:
            if time.time() > deadline:
                break
            
            # Pick random edge to start
            u, v = next(iter(available))
            path = [u, v]
            available.remove(tuple(sorted((u, v))))
            current = v
            visited = {u, v}
            
            while True:
                neighbors = [nb for nb in adj[current] 
                            if nb not in visited and tuple(sorted((current, nb))) in available]
                if not neighbors:
                    break
                next_node = rng.choice(neighbors)
                available.remove(tuple(sorted((current, next_node))))
                path.append(next_node)
                visited.add(next_node)
                current = next_node
            
            paths.append(path)
        
        if available:
            for u, v in available:
                paths.append([u, v])
        
        return paths
    
    results = []
    for i, spec in enumerate([(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]):
        if time.time() > deadline - 5:
            # Fallback
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(*spec)
        paths = solve_with_flow(edges, spec[2])
        results.append(paths)
    
    return results
