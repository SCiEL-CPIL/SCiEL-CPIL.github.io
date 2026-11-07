import time
import random
from collections import defaultdict, deque

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
    
    best_cycles = []
    best_count = 0
    
    def find_2regular_subgraph(vertices, used_edges, deadline):
        """
        Find a 2-regular spanning subgraph on vertices using max-flow.
        Returns list of cycles or None.
        """
        if time.time() > deadline:
            return None
        
        vertices = list(vertices)
        k = len(vertices)
        if k < 3:
            return None
        
        # Check each vertex has at least 2 available neighbors
        for v in vertices:
            avail_count = 0
            for u in adj[v]:
                if u in vertices:
                    e = (min(u, v), max(u, v))
                    if e not in used_edges:
                        avail_count += 1
            if avail_count < 2:
                return None
        
        # Build flow network for 2-factor:
        # Source -> v_out (cap 2)
        # v_out -> v_in for each available edge (cap 1)
        # v_in -> Sink (cap 2)
        # Find max flow; if flow == 2k, we have a 2-regular subgraph
        
        # Nodes: 0=source, 1=sink, then v_out and v_in for each vertex
        # Use simple BFS-based max flow (Edmonds-Karp) since graph is small
        
        # Map vertices to indices
        vlist = sorted(vertices)
        vidx = {v: i for i, v in enumerate(vlist)}
        num_v = len(vlist)
        
        # Node numbering: 0=source, 1=sink, 2..2+num_v-1 = v_out, 2+num_v..4+num_v-1 = v_in
        SRC = 0
        SNK = 1
        def out_node(i): return 2 + i
        def in_node(i): return 2 + num_v + i
        total_nodes = 2 + 2 * num_v
        
        # Build adjacency for flow network
        # Each edge in flow network: (from, to, cap)
        # We'll use a dict-based residual network
        # residual[u][v] = remaining capacity
        residual = defaultdict(lambda: defaultdict(int))
        # Also track reverse edges for residual graph
        rev_edge = {}  # (u,v) -> (v,u)
        
        def add_edge(u, v, cap):
            residual[u][v] += cap
            if v not in residual[u]:
                pass
            residual[v][u] += 0  # ensure reverse exists
            rev_edge[(u, v)] = (v, u)
        
        # Source to each v_out: cap 2
        for i in range(num_v):
            add_edge(SRC, out_node(i), 2)
        
        # Each v_in to sink: cap 2
        for i in range(num_v):
            add_edge(in_node(i), SNK, 2)
        
        # For each available host edge (u,v) where u,v in vertices:
        # Add flow edge out_node(u) -> in_node(v) with cap 1
        # And out_node(v) -> in_node(u) with cap 1
        # But we need to be careful: each undirected edge can be used at most once
        # Actually, in a 2-factor, each edge is used at most once
        # So for each undirected edge (u,v), we add:
        #   out_node(u) -> in_node(v) cap 1
        #   out_node(v) -> in_node(u) cap 1
        # But this allows both directions, which would mean using the edge twice
        # We need to ensure each undirected edge is used at most once
        
        # Alternative: model as matching-like problem
        # Actually, let me think differently.
        # A 2-regular graph is a union of cycles.
        # Each vertex has degree exactly 2.
        # This is equivalent to finding a perfect matching in a related graph,
        # or we can use the fact that in a 2-regular graph on k vertices,
        # there are exactly k edges.
        
        # Let me use a different approach: since k is at most 21,
        # and we just need a 2-factor, let's use a simpler method.
        # 
        # Actually, for small graphs, let's try a direct approach:
        # Find a 2-regular spanning subgraph by trying to assign 2 edges per vertex
        # such that no edge is used twice.
        
        # Let me use a flow formulation properly:
        # This is a b-matching problem. Each vertex needs degree 2.
        # Each edge can be used at most once.
        # 
        # Standard reduction: split each vertex v into v_in and v_out
        # Source -> v_in: cap 2
        # v_out -> Sink: cap 2
        # For each edge (u,v): v_out -> u_in cap 1, u_out -> v_in cap 1
        # But this double-counts. 
        # 
        # Better: for each undirected edge (u,v), create a single directed arc
        # in one direction with cap 1, but we need to choose direction.
        # 
        # Actually, the standard way: model as a flow where each edge (u,v)
        # becomes two directed arcs u->v and v->u each with cap 1, but we add
        # a constraint that at most one is used. This is tricky.
        
        # Simpler: since k <= 21, let's just use backtracking but with a
        # much better heuristic, or use a different decomposition.
        
        # Let me try: find ANY cycle first, remove its edges, repeat.
        # Use a faster cycle-finding heuristic.
        
        return None  # Placeholder - will use cycle-finding approach below
    
    def find_cycle_fast(vertices, used_edges, deadline):
        """
        Find a Hamiltonian cycle on vertices avoiding used_edges.
        Uses a greedy + local search approach for speed.
        """
        if time.time() > deadline:
            return None
        
        vertices = list(vertices)
        k = len(vertices)
        if k < 3:
            return None
        
        # Check feasibility
        for v in vertices:
            avail_count = 0
            for u in adj[v]:
                if u in vertices:
                    e = (min(u, v), max(u, v))
                    if e not in used_edges:
                        avail_count += 1
            if avail_count < 2:
                return None
        
        # Try multiple random starts with greedy + backtracking
        rng = random.Random(int(time.time() * 1000))
        
        for attempt in range(20):
            if time.time() > deadline:
                return None
            
            # Choose start vertex: prefer most constrained
            avail_deg = {}
            for v in vertices:
                avail_deg[v] = sum(1 for u in adj[v] if u in vertices and (min(u,v),max(u,v)) not in used_edges)
            
            # Start from most constrained vertex
            start = min(vertices, key=lambda v: (avail_deg[v], v))
            
            # Greedy construction with backtracking
            path = [start]
            remaining = set(vertices) - {start}
            
            def backtrack(path, remaining, depth=0):
                if time.time() > deadline:
                    return None
                if len(path) == k:
                    last = path[-1]
                    e = (min(last, start), max(last, start))
                    if e not in used_edges:
                        return path[:]
                    return None
                
                last = path[-1]
                # Get available neighbors
                candidates = []
                for v in remaining:
                    if v in adj[last]:
                        e = (min(last, v), max(last, v))
                        if e not in used_edges:
                            # Score by how constrained v is
                            score = 0
                            for w in adj[v]:
                                if w in vertices and w != last:
                                    e2 = (min(v, w), max(v, w))
                                    if e2 not in used_edges:
                                        score += 1
                            candidates.append((score, v))
                
                if not candidates:
                    return None
                
                # Sort by constraint (most constrained first)
                candidates.sort()
                
                for _, v in candidates:
                    e = (min(last, v), max(last, v))
                    if e not in used_edges:
                        path.append(v)
                        remaining.remove(v)
                        result = backtrack(path, remaining, depth + 1)
                        if result is not None:
                            return result
                        path.pop()
                        remaining.add(v)
                return None
            
            result = backtrack(path, remaining)
            if result is not None:
                return result
        
        return None
    
    def try_vertex_subset(vertices, deadline):
        """Try to find max edge-disjoint Hamiltonian cycles on this vertex set."""
        nonlocal best_cycles, best_count
        if time.time() > deadline:
            return
        
        vertices = frozenset(vertices)
        k = len(vertices)
        if k < 3:
            return
        
        # Count edges in induced subgraph
        induced_edges = set()
        for u in vertices:
            for v in adj[u]:
                if v in vertices:
                    edge = (min(u, v), max(u, v))
                    induced_edges.add(edge)
        
        max_possible = len(induced_edges) // k
        if max_possible <= best_count:
            return
        
        used_edges = set()
        cycles = []
        
        # Try to find cycles greedily
        for attempt in range(max_possible):
            if time.time() > deadline:
                break
            
            cycle = find_cycle_fast(vertices, used_edges, deadline)
            if cycle is None:
                break
            
            # Add edges to used
            for i in range(k):
                u = cycle[i]
                v = cycle[(i + 1) % k]
                edge = (min(u, v), max(u, v))
                used_edges.add(edge)
            cycles.append(cycle)
        
        if len(cycles) > best_count:
            best_count = len(cycles)
            best_cycles = [list(c) for c in cycles]
    
    all_vertices = set(range(n))
    
    # Try full vertex set
    try_vertex_subset(all_vertices, deadline)
    
    # Try all single-vertex removals
    for removed in range(n):
        if time.time() > deadline:
            break
        subset = all_vertices - {removed}
        try_vertex_subset(subset, deadline)
    
    # Try random subsets of various sizes
    random.seed(42)
    for k in range(18, 20):
        for _ in range(30):
            if time.time() > deadline:
                break
            subset = set(random.sample(range(n), k))
            try_vertex_subset(subset, deadline)
    
    # Try subsets of size 20 with different random seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        random.seed(seed)
        for _ in range(10):
            if time.time() > deadline:
                break
            subset = set(random.sample(range(n), 20))
            try_vertex_subset(subset, deadline)
    
    # Return best result
    if best_cycles:
        result = []
        for cycle in best_cycles:
            result.append([int(v) for v in cycle])
        return result
    else:
        # Fallback: try to find any cycle on full vertex set
        cycle = find_cycle_fast(all_vertices, set(), deadline)
        if cycle:
            return [[int(v) for v in cycle]]
        return [list(range(n))]

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}
