import time
import sys
from collections import defaultdict
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph_milp(edges):
        if not edges:
            return []
        
        # Build graph structure
        n = max(max(u, v) for u, v in edges) + 1
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        m = len(edges)
        
        # MILP Formulation:
        # Variables:
        # 1. x[e] for each edge e: 1 if used in a path, 0 otherwise.
        #    Constraint: sum(x[e]) = m (all edges must be used)
        # 2. y[v, u] for each directed arc (v->u) where (min(v,u), max(v,u)) is an edge:
        #    1 if the path traverses from v to u.
        #    Constraint: y[v,u] + y[u,v] = x[edge(v,u)]
        # 3. s[v] for each vertex v: 1 if v is the start of a path, 0 otherwise.
        #    Objective: Minimize sum(s[v])
        #    Constraint: For each vertex v, flow conservation:
        #    sum(y[u,v] for u in neighbors) + s[v] = sum(y[v,w] for w in neighbors) + t[v]
        #    Where t[v] is 1 if v is the end of a path.
        #    Actually, simpler:
        #    inflow(v) + s[v] = outflow(v) + t[v]
        #    Also, s[v] + t[v] <= 1? No, a path of length 0 is not allowed? 
        #    Paths must have at least one edge? "cover all edges". 
        #    If a vertex is isolated in the path sense (degree 0 in the path subgraph), it doesn't contribute.
        #    But every edge is used.
        #    Let's define:
        #    outflow(v) = sum(y[v,w] for w in adj[v])
        #    inflow(v) = sum(y[u,v] for u in adj[v])
        #    For each v:
        #    outflow(v) - inflow(v) = s[v] - t[v]
        #    s[v], t[v] in {0,1}
        #    Sum(s[v]) is the number of paths.
        #    Also, we need to ensure paths are simple. 
        #    The flow formulation on a graph with unit capacities on vertex arcs ensures simple paths?
        #    Standard trick: split each vertex v into v_in and v_out.
        #    Edge (u,v) becomes arc u_out -> v_in with capacity 1, cost 0.
        #    Arc v_in -> v_out with capacity 1, cost 0.
        #    Source S connects to v_in (or v_out?) with capacity 1?
        #    Actually, we are minimizing the number of paths.
        #    This is equivalent to finding a minimum number of paths that cover all edges.
        #    This can be solved by min-cost flow where we send flow from a super source to a super sink?
        #    No, we want to cover edges.
        
        # Let's use the standard ILP for path cover.
        # Variables:
        # y[u, v] for each directed edge (u,v) in the graph (if undirected edge exists).
        # Binary.
        # Constraints:
        # 1. For each undirected edge (u,v): y[u,v] + y[v,u] = 1.
        # 2. For each vertex v:
        #    sum(y[v,u] for u in adj[v]) - sum(y[u,v] for u in adj[v]) = s[v] - t[v]
        #    where s[v] is 1 if v is a start, t[v] is 1 if v is an end.
        # 3. s[v], t[v] are binary.
        # 4. s[v] * t[v] = 0 (a vertex cannot be both start and end of the same path? 
        #    Actually, if a path is a single vertex, it has no edges. But we must cover edges.
        #    So no path is a single vertex. 
        #    However, a vertex can be the end of one path and the start of another?
        #    No, in a path decomposition, paths are edge-disjoint. 
        #    Vertex disjointness is NOT required. Only edge disjointness.
        #    "Paths must be simple, use graph edges, be edge-disjoint and cover all edges."
        #    Simple path = no repeated vertices.
        #    So within a single path, vertices are unique.
        #    Across different paths, vertices can repeat.
        
        # The flow formulation with vertex splitting enforces that each vertex is used at most once *per path*.
        # But since we are decomposing the whole graph, and edges are disjoint, 
        # can a vertex be in two different paths?
        # Yes.
        # So the "vertex capacity 1" constraint in a single flow network assumes all paths are in one flow.
        # If we send multiple units of flow, and each unit represents a path, 
        # and the vertex arc has capacity 1, then a vertex can only be part of ONE path.
        # This is incorrect for general path decomposition. 
        # Wait. If a vertex has degree 4, it must be in 2 paths. 
        # So vertex capacity must be degree/2? No.
        
        # Correct Model:
        # We want to partition the edges into paths.
        # This is the "Minimum Path Cover" of the edges.
        # For a graph, the minimum number of edge-disjoint paths covering all edges is:
        # max(1, ceil(odd_vertices / 2)) if the graph is connected?
        # No, that's for trails (Eulerian).
        # For simple paths, it's harder.
        
        # However, for small graphs, we can use ILP directly on the edges.
        # Let x_e be 1 if edge e is in the path cover. (All are 1).
        # Let y_{u,v} be 1 if edge (u,v) is traversed u->v.
        # Constraint: y_{u,v} + y_{v,u} = 1 for each edge.
        # Let s_v be 1 if v is a start of a path.
        # Let t_v be 1 if v is an end of a path.
        # Flow conservation at v:
        # sum_{u} y_{u,v} + s_v = sum_{w} y_{v,w} + t_v
        # Also, s_v + t_v <= 1? 
        # If a path is v1-v2-v3, v1 is start, v3 is end. 
        # If a path is just an edge u-v, u is start, v is end.
        # Can a vertex be both start and end? Only if the path has length 0, which doesn't cover edges.
        # So s_v + t_v <= 1 is valid for any vertex in a path cover of edges?
        # Wait. Consider a cycle. No starts or ends. s_v=0, t_v=0.
        # Consider a path. One start, one end.
        # So sum(s_v) = number of paths.
        
        # Does this ensure simplicity?
        # If we just solve this ILP, does it allow cycles?
        # Yes, if s_v=0 and t_v=0 for all v in a component, it forms a cycle.
        # But we want paths. A cycle can be broken into a path by removing one edge?
        # No, we must cover all edges.
        # A cycle is not a simple path (start != end).
        # But we can treat a cycle as a path by arbitrarily breaking it?
        # No, the path must be a sequence of vertices.
        # If the solution yields a cycle, we can break it into a path by picking a start and end?
        # But then the flow conservation breaks.
        
        # Actually, Gallai's conjecture is about decomposing into *paths*.
        # A cycle is not a path.
        # However, any cycle can be decomposed into one path (by removing one edge? No, that leaves an edge uncovered).
        # You can't cover a cycle with a single simple path.
        # You need to break it.
        # Wait, a simple path cannot repeat vertices. A cycle repeats the start vertex at the end.
        # So a cycle must be broken into a path by... ?
        # You can't. A cycle requires at least 2 paths? No.
        # If you have a triangle A-B-C-A.
        # Path 1: A-B. Path 2: C-A. (2 paths).
        # Or Path 1: A-B-C. (Covers A-B, B-C). Edge C-A remains.
        # Path 2: A-C. (Covers A-C).
        # Total 2 paths.
        # Lower bound: odd vertices. In a cycle, all degrees are 2 (even).
        # But we need simple paths.
        
        # The ILP formulation above allows s_v=0, t_v=0.
        # If a component is a cycle, the ILP will find a solution with s=t=0.
        # This corresponds to a cycle.
        # We need to convert cycles to paths.
        # A cycle of length L can be covered by 1 path? No.
        # It requires 1 path if we allow "closed paths"? No, "simple path".
        # So a cycle must be split.
        # Actually, if the ILP returns a set of cycles and paths, we can break each cycle into a path by removing one edge?
        # No, we must cover all edges.
        # If we have a cycle, we can pick any edge to be the "last" edge of a path?
        # No.
        # Let's check the definition. "Path" usually means open path.
        # If the ILP finds a cycle, we can break it into a path by arbitrarily designating a start and end?
        # But then the flow conservation changes.
        # If we force s_v + t_v >= 1 for all v? No.
        
        # Alternative: Add a constraint that prevents cycles.
        # Or, simply, if the solution contains a cycle, we can break it into a path by "cutting" it.
        # But cutting a cycle doesn't reduce the number of paths, it just makes it a path.
        # Wait. If I have a cycle A-B-C-A.
        # ILP says: 1 "component".
        # But it's not a path.
        # I need 2 paths to cover it?
        # No, I can have path A-B and path C-A. That's 2 paths.
        # Can I do it in 1? No.
        # So the number of paths for a cycle is 1?
        # If "path" allows start != end, a cycle is not a path.
        # So a cycle must be decomposed into at least 2 paths?
        # No, if I have a cycle, I can traverse it.
        # But I can't repeat vertices.
        # So I must stop before returning to start.
        # So I leave one edge.
        # So 1 path covers L-1 edges. 1 path covers 1 edge. Total 2 paths.
        
        # So, if the ILP finds a cycle, we need to add 1 to the path count for that cycle?
        # Or rather, the ILP objective minimizes sum(s_v).
        # For a cycle, sum(s_v) = 0.
        # But we need 2 paths.
        # So the ILP underestimates.
        
        # To fix this, we can add a constraint that forces at least one start and one end for every connected component of the selected edges?
        # Or simply, if the result has cycles, we break them.
        # How to break?
        # If we have a cycle, we can pick any edge (u,v) and "reverse" the flow? No.
        # We can just take the cycle and make it a path by removing one edge from the cycle structure?
        # No, we must cover all edges.
        # So we split the cycle into two paths?
        # Actually, if we have a cycle, we can just pick a start node s and end node t (adjacent in the cycle) and define the path as the rest of the cycle.
        # But then the edge (s,t) is left over.
        # So we have Path 1: s ... t. Path 2: t ... s (just the edge? No, t-s is the edge).
        # So yes, 2 paths.
        
        # So, strategy:
        # 1. Solve ILP minimizing sum(s_v).
        # 2. Extract paths.
        # 3. If any "path" is a cycle (start == end in the flow sense, i.e., s=t=0), break it into 2 paths.
        #    Wait, if s=t=0, it's a cycle.
        #    We can break it by picking an edge (u,v) in the cycle.
        #    Path 1: u ... v (following the cycle).
        #    Path 2: v ... u (the other way? No, just the edge v-u? No, the edge is already used).
        #    Ah, the edge (u,v) is part of the cycle.
        #    If we make Path 1: u ... v, it uses all edges except (v,u)? No, the cycle is u-v-...-u.
        #    If we stop at v, we haven't used the edge returning to u.
        #    So Path 1 uses L-1 edges.
        #    Path 2 uses the last edge.
        #    So yes, 2 paths.
        
        # So, total paths = sum(s_v) + number_of_cycles.
        # But wait. If a component is a tree, it has no cycles.
        # If a component has cycles, each independent cycle adds 1 to the count?
        # Actually, a connected component with E edges and V vertices.
        # Cyclomatic number = E - V + 1.
        # Number of paths needed?
        # For a tree, it's (odd_vertices + 2) / 2? No.
        # For a tree, it's ceil(odd/2).
        # For a graph with cycles, it's more complex.
        
        # Let's just solve the ILP and then post-process.
        
        # Variables:
        # y_{u,v} for each directed arc.
        # s_v, t_v for each vertex.
        
        # Indices:
        # Arcs: list of (u, v) for all undirected edges.
        # There are 2*m arcs.
        
        arc_list = []
        arc_to_edge = {}
        for u, v in edges:
            arc_list.append((u, v))
            arc_list.append((v, u))
            arc_to_edge[(u, v)] = (u, v)
            arc_to_edge[(v, u)] = (u, v)
        
        n_arcs = len(arc_list)
        n_vars = n_arcs + 2 * n # y's + s's + t's
        
        c = np.zeros(n_vars)
        for v in range(n):
            c[n_arcs + v] = 1.0 # Minimize sum(s_v)
            # t_v cost is 0
        
        # Constraints:
        # 1. y_{u,v} + y_{v,u} = 1 for each edge.
        # 2. Flow conservation: sum_in + s_v - sum_out - t_v = 0.
        
        A_rows = []
        b_lower = []
        b_upper = []
        
        # Edge constraints
        for i, (u, v) in enumerate(edges):
            row = np.zeros(n_vars)
            row[arc_list.index((u, v))] = 1.0
            row[arc_list.index((v, u))] = 1.0
            A_rows.append(row)
            b_lower.append(1.0)
            b_upper.append(1.0)
        
        # Flow conservation
        for v in range(n):
            row = np.zeros(n_vars)
            # Inflow
            for u in adj[v]:
                if (u, v) in arc_to_edge:
                    row[arc_list.index((u, v))] = 1.0
            # Outflow
            for w in adj[v]:
                if (v, w) in arc_to_edge:
                    row[arc_list.index((v, w))] = -1.0
            # s_v
            row[n_arcs + v] = 1.0
            # t_v
            row[n_arcs + n + v] = -1.0
            
            A_rows.append(row)
            b_lower.append(0.0)
            b_upper.append(0.0)
        
        A = np.array(A_rows)
        lb = np.array(b_lower)
        ub = np.array(b_upper)
        
        integrality = np.zeros(n_vars)
        integrality[:] = 1 # All binary
        
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        
        try:
            res = milp(c=c, constraints=LinearConstraint(A, lb, ub), integrality=integrality, bounds=bounds)
        except Exception:
            return None
            
        if res.status != 0:
            return None
            
        x = res.x
        y_vals = x[:n_arcs]
        s_vals = x[n_arcs:n_arcs+n]
        t_vals = x[n_arcs+n:]
        
        # Reconstruct paths
        # Build directed graph
        dir_adj = defaultdict(list)
        for i, (u, v) in enumerate(arc_list):
            if y_vals[i] > 0.5:
                dir_adj[u].append(v)
                
        paths = []
        visited_arcs = set()
        
        # Find starts
        starts = [v for v in range(n) if s_vals[v] > 0.5]
        
        # Also, if a component has no start, it's a cycle.
        # We need to handle cycles.
        
        # First, follow starts
        for start in starts:
            path = [start]
            current = start
            while True:
                # Find next
                nexts = dir_adj[current]
                if not nexts:
                    break
                # Should be exactly one
                next_v = nexts[0]
                path.append(next_v)
                current = next_v
            paths.append(path)
            
        # Now, find cycles (components with no start)
        # Vertices not in any path so far?
        # Or just check for remaining arcs.
        remaining_arcs = [(u, v) for i, (u, v) in enumerate(arc_list) if y_vals[i] > 0.5 and (u, v) not in visited_arcs]
        # Wait, I didn't mark visited arcs in the first loop.
        
        # Let's just identify all edges covered by the paths found.
        covered_edges = set()
        for p in paths:
            for i in range(len(p)-1):
                u, v = p[i], p[i+1]
                covered_edges.add(tuple(sorted((u, v))))
                
        # Find edges not covered (these form cycles)
        cycle_edges = [e for e in edges if e not in covered_edges]
        
        # Decompose cycle_edges into cycles
        # Since they are disjoint cycles (from the flow solution), we can just find them.
        # Build graph for cycle_edges
        cycle_adj = defaultdict(list)
        for u, v in cycle_edges:
            cycle_adj[u].append(v)
            cycle_adj[v].append(u)
            
        visited_vertices = set()
        for u, v in cycle_edges:
            if u in visited_vertices:
                continue
            # Find cycle
            path = [u]
            visited_vertices.add(u)
            current = u
            prev = -1
            while True:
                nexts = [x for x in cycle_adj[current] if x != prev]
                if not nexts:
                    break
                next_v = nexts[0]
                if next_v == u:
                    # Closed cycle
                    break
                path.append(next_v)
                visited_vertices.add(next_v)
                prev = current
                current = next_v
            # The path is a cycle. Break it into 2 paths.
            # Path 1: path[0] ... path[-1]
            # Path 2: path[-1] ... path[0] (just the edge? No, the edge closing the cycle)
            # Actually, the cycle is path[0]-...-path[-1]-path[0].
            # We can make Path 1: path[0] ... path[-1]
            # And Path 2: path[-1] ... path[0] ? No, that's the same edge.
            # We need to cover the edge (path[-1], path[0]).
            # So Path 2 is just [path[-1], path[0]].
            # But is that a valid path? Yes.
            # So we add path and [path[-1], path[0]].
            if len(path) > 1:
                paths.append(path)
                paths.append([path[-1], path[0]])
            else:
                # Self loop? Unlikely in simple graph.
                pass
                
        return paths

    def decompose_graph_greedy(edges):
        # Fallback greedy
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(tuple(sorted(e)) for e in edges)
        paths = []
        
        while remaining_edges:
            rem_deg = defaultdict(int)
            for u, v in remaining_edges:
                rem_deg[u] += 1
                rem_deg[v] += 1
                
            if not rem_deg:
                break
                
            start = max(rem_deg.keys(), key=lambda x: rem_deg[x])
            
            path = [start]
            used_in_path = {start}
            current = start
            path_edges = []
            
            while True:
                candidates = []
                for nb in adj[current]:
                    edge = tuple(sorted((current, nb)))
                    if edge in remaining_edges and nb not in used_in_path:
                        candidates.append((rem_deg[nb], nb))
                
                if not candidates:
                    break
                
                candidates.sort(reverse=True)
                _, next_v = candidates[0]
                path.append(next_v)
                used_in_path.add(next_v)
                edge = tuple(sorted((current, next_v)))
                path_edges.append(edge)
                current = next_v
            
            for e in path_edges:
                remaining_edges.discard(e)
            for u, v in path_edges:
                rem_deg[u] -= 1
                rem_deg[v] -= 1
            
            paths.append(path)
            
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            break
            
        edges = instance(*spec)
        
        # Try MILP first
        try:
            paths = decompose_graph_milp(edges)
            if paths is None:
                paths = decompose_graph_greedy(edges)
        except Exception:
            paths = decompose_graph_greedy(edges)
            
        results.append(paths)
        
    while len(results) < 3:
        spec = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])
        
    return results

