import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin for 90s limit
    
    n = 21
    edges = P['edges']
    edge_set = set()
    adj = defaultdict(set)
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute edge index for fast lookup
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        key = (min(u, v), max(u, v))
        edge_index[key] = i
    
    def find_hamiltonian_cycle_global(vertices, max_cycles, adj, deadline):
        """
        Use a global ILP-like approach: assign each edge to at most one cycle,
        and each vertex to exactly 2 edges per cycle.
        We use a backtracking search that builds all cycles simultaneously.
        """
        vertices = frozenset(vertices)
        k = len(vertices)
        if k < 3:
            return []
        
        # Get all edges in the induced subgraph
        induced_edges = []
        for u in vertices:
            for v in adj[u]:
                if v in vertices and u < v:
                    induced_edges.append((u, v))
        
        m = len(induced_edges)
        if m < max_cycles * k:
            return []
        
        # For each vertex, list edges incident to it
        vert_edges = defaultdict(list)
        for i, (u, v) in enumerate(induced_edges):
            vert_edges[u].append(i)
            vert_edges[v].append(i)
        
        # Check degree constraint: each vertex needs 2*max_cycles incident edges
        for v in vertices:
            if len(vert_edges[v]) < 2 * max_cycles:
                return []
        
        # State: edge_assignment[e] = cycle_id or -1
        # vertex_degree[v][c] = number of edges assigned to cycle c at vertex v
        edge_assign = [-1] * m
        vert_deg = [[0] * max_cycles for _ in range(n)]
        
        # Precompute for each vertex and cycle, the set of edges that can still be used
        # We'll use a priority queue of (vertex, cycle) pairs that need more edges
        
        def can_assign_edge(e, c):
            """Check if edge e can be assigned to cycle c."""
            u, v = induced_edges[e]
            return vert_deg[u][c] < 2 and vert_deg[v][c] < 2
        
        def assign_edge(e, c):
            u, v = induced_edges[e]
            edge_assign[e] = c
            vert_deg[u][c] += 1
            vert_deg[v][c] += 1
        
        def unassign_edge(e, c):
            u, v = induced_edges[e]
            edge_assign[e] = -1
            vert_deg[u][c] -= 1
            vert_deg[v][c] -= 1
        
        def extract_cycles():
            """Extract cycles from the edge assignment."""
            cycles = []
            for c in range(max_cycles):
                # Build adjacency for this cycle
                cycle_adj = defaultdict(list)
                for e in range(m):
                    if edge_assign[e] == c:
                        u, v = induced_edges[e]
                        cycle_adj[u].append(v)
                        cycle_adj[v].append(u)
                
                # Find the cycle
                if not cycle_adj:
                    return None
                start = next(iter(cycle_adj))
                cycle = [start]
                visited_edges = set()
                current = start
                prev = None
                for _ in range(k):
                    next_v = None
                    for nb in cycle_adj[current]:
                        if nb != prev:
                            next_v = nb
                            break
                    if next_v is None or next_v == start and len(cycle) < k:
                        return None
                    cycle.append(next_v)
                    prev, current = current, next_v
                
                if cycle[0] != cycle[-1]:
                    return None
                cycle = cycle[:-1]  # Remove duplicate start/end
                if len(cycle) != k or len(set(cycle)) != k:
                    return None
                cycles.append(cycle)
            
            return cycles if len(cycles) == max_cycles else None
        
        # Backtracking search
        # Strategy: process edges in a smart order, try to assign to cycles
        # Use a heuristic: prioritize edges that are "critical" (fewer alternatives)
        
        def backtrack(edge_pos, cycles_completed):
            if time.time() > deadline:
                return None
            
            if cycles_completed == max_cycles:
                result = extract_cycles()
                return result
            
            if edge_pos >= m:
                return None
            
            e = edge_pos
            u, v = induced_edges[e]
            
            # Try assigning this edge to each cycle
            for c in range(max_cycles):
                if edge_assign[e] == c:
                    continue
                if can_assign_edge(e, c):
                    assign_edge(e, c)
                    result = backtrack(edge_pos + 1, cycles_completed)
                    if result is not None:
                        return result
                    unassign_edge(e, c)
            
            # Skip this edge
            result = backtrack(edge_pos + 1, cycles_completed)
            if result is not None:
                return result
            
            return None
        
        # The naive backtracking is too slow. Let's use a smarter approach:
        # Build cycles one at a time, but with lookahead.
        
        used_edges = set()
        cycles = []
        
        for c in range(max_cycles):
            if time.time() > deadline:
                return cycles
            
            # Find a Hamiltonian cycle avoiding used_edges
            cycle = _find_ham_cycle(vertices, used_edges, adj, deadline)
            if cycle is None:
                # Try with randomization
                for attempt in range(20):
                    cycle = _find_ham_cycle_random(vertices, used_edges, adj, deadline, seed=c*100+attempt)
                    if cycle is not None:
                        break
            if cycle is None:
                return cycles
            
            # Add edges to used
            for i in range(k):
                u = cycle[i]
                v = cycle[(i + 1) % k]
                used_edges.add((min(u, v), max(u, v)))
            cycles.append(cycle)
        
        return cycles
    
    def _find_ham_cycle(vertices, used_edges, adj, deadline):
        """Find a Hamiltonian cycle on vertices avoiding used_edges."""
        vertices = list(vertices)
        k = len(vertices)
        if k < 3:
            return None
        
        # Precompute available neighbors
        avail = {}
        for v in vertices:
            avail[v] = [u for u in adj[v] if u in vertices and (min(u,v), max(u,v)) not in used_edges]
            if len(avail[v]) < 2:
                return None
        
        # Choose start vertex with minimum degree
        start = min(vertices, key=lambda v: len(avail[v]))
        remaining = set(vertices) - {start}
        path = [start]
        
        def backtrack(path, remaining):
            if time.time() > deadline:
                return None
            if len(path) == k:
                last = path[-1]
                edge = (min(last, start), max(last, start))
                if edge not in used_edges:
                    return path[:]
                return None
            
            last = path[-1]
            candidates = []
            for v in remaining:
                if v in adj[last]:
                    edge = (min(last, v), max(last, v))
                    if edge not in used_edges:
                        future_options = len([w for w in avail[v] if w != last and (min(v,w),max(v,w)) not in used_edges])
                        candidates.append((future_options, v))
            
            candidates.sort()
            
            for _, v in candidates:
                if time.time() > deadline:
                    return None
                edge = (min(last, v), max(last, v))
                if edge not in used_edges:
                    path.append(v)
                    remaining.remove(v)
                    result = backtrack(path, remaining)
                    if result is not None:
                        return result
                    path.pop()
                    remaining.add(v)
            return None
        
        return backtrack(path, remaining)
    
    def _find_ham_cycle_random(vertices, used_edges, adj, deadline, seed=0):
        """Find a Hamiltonian cycle with random restarts."""
        vertices = list(vertices)
        k = len(vertices)
        if k < 3:
            return None
        
        rng = random.Random(seed)
        
        for attempt in range(30):
            if time.time() > deadline:
                return None
            
            avail = {}
            valid = True
            for v in vertices:
                avail[v] = [u for u in adj[v] if u in vertices and (min(u,v), max(u,v)) not in used_edges]
                if len(avail[v]) < 2:
                    valid = False
                    break
            if not valid:
                return None
            
            start = rng.choice(vertices)
            remaining = set(vertices) - {start}
            path = [start]
            
            def backtrack(path, remaining):
                if time.time() > deadline:
                    return None
                if len(path) == k:
                    last = path[-1]
                    edge = (min(last, start), max(last, start))
                    if edge not in used_edges:
                        return path[:]
                    return None
                
                last = path[-1]
                candidates = []
                for v in remaining:
                    if v in adj[last]:
                        edge = (min(last, v), max(last, v))
                        if edge not in used_edges:
                            future_options = len([w for w in avail[v] if w != last and (min(v,w),max(v,w)) not in used_edges])
                            candidates.append((future_options, v))
                
                if candidates:
                    min_opts = min(c[0] for c in candidates)
                    constrained = [c for c in candidates if c[0] <= min_opts + 1]
                    rng.shuffle(constrained)
                    candidates = constrained
                
                for _, v in candidates:
                    if time.time() > deadline:
                        return None
                    edge = (min(last, v), max(last, v))
                    if edge not in used_edges:
                        path.append(v)
                        remaining.remove(v)
                        result = backtrack(path, remaining)
                        if result is not None:
                            return result
                        path.pop()
                        remaining.add(v)
                return None
            
            result = backtrack(path, remaining)
            if result is not None:
                return result
        
        return None
    
    all_vertices = frozenset(range(n))
    
    # Try to find 3 edge-disjoint Hamiltonian cycles on the full vertex set
    best_cycles = []
    
    # First, try with the full vertex set
    for max_c in [3, 2, 1]:
        if time.time() > deadline:
            break
        cycles = find_hamiltonian_cycle_global(all_vertices, max_c, adj, deadline)
        if cycles and len(cycles) >= max_c:
            best_cycles = cycles[:max_c]
            break
    
    # If that didn't work, try with vertex removals
    if not best_cycles or len(best_cycles) < 3:
        for removed in range(n):
            if time.time() > deadline:
                break
            subset = all_vertices - {removed}
            for max_c in [3, 2]:
                if time.time() > deadline:
                    break
                cycles = find_hamiltonian_cycle_global(subset, max_c, adj, deadline)
                if cycles and len(cycles) >= max_c and len(cycles) > len(best_cycles):
                    best_cycles = cycles[:max_c]
    
    # Return best result
    if best_cycles:
        result = []
        for cycle in best_cycles:
            result.append([int(v) for v in cycle])
        return result
    else:
        # Fallback
        cycle = _find_ham_cycle(all_vertices, set(), adj, deadline)
        if cycle:
            return [[int(v) for v in cycle]]
        return [list(range(n))]

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}
