"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
    
    def time_up():
        return time.time() - start_time > DEADLINE
    
    def is_clique(vertices):
        for u, v in combinations(vertices, 2):
            if v not in adj[u]:
                return False
        return True
    
    def find_max_clique():
        """Bron-Kerbosch with pivot to find maximum clique."""
        best_clique = []
        
        def bronk(R, P_set, X):
            nonlocal best_clique
            if time_up():
                return
            if not P_set and not X:
                if len(R) > len(best_clique):
                    best_clique = list(R)
                return
            # Choose pivot
            union = P_set | X
            if not union:
                return
            # Pick pivot with max degree in P
            pivot = max(union, key=lambda v: len(adj[v] & P_set))
            candidates = P_set - adj[pivot]
            for v in list(candidates):
                bronk(R | {v}, P_set & adj[v], X & adj[v])
                P_set.remove(v)
                X.add(v)
        
        bronk(set(), set(range(n)), set())
        return best_clique
    
    def max_flow(s, t, adj_list, blocked):
        """Max flow using Edmonds-Karp. Returns max flow value."""
        # Build flow network
        # Use node-splitting for vertex capacities
        # Actually, let's just do standard max flow on the graph with unit capacities
        # on edges, and we'll handle vertex-disjointness separately
        # For vertex-disjoint paths, we need to split vertices
        # But for simplicity in checking if paths exist, let's use a different approach
        
        # Standard max flow with edge capacities = 1
        from collections import defaultdict
        flow_graph = defaultdict(lambda: defaultdict(int))
        for u in range(n):
            if u in blocked:
                continue
            for v in adj_list[u]:
                if v in blocked:
                    continue
                flow_graph[u][v] += 1
                flow_graph[v][u] += 1
        
        def bfs():
            parent = {s: None}
            queue = deque([s])
            while queue:
                u = queue.popleft()
                for v in flow_graph[u]:
                    if flow_graph[u][v] > 0 and v not in parent:
                        parent[v] = u
                        if v == t:
                            return parent
                        queue.append(v)
            return None
        
        total_flow = 0
        while not time_up():
            parent = bfs()
            if parent is None:
                break
            # Find bottleneck
            path_flow = float('inf')
            v = t
            while v != s:
                u = parent[v]
                path_flow = min(path_flow, flow_graph[u][v])
                v = u
            # Update residual
            v = t
            while v != s:
                u = parent[v]
                flow_graph[u][v] -= path_flow
                flow_graph[v][u] += path_flow
                v = u
            total_flow += path_flow
        return total_flow
    
    def find_path(s, t, blocked):
        """Find a simple path from s to t avoiding blocked vertices."""
        if s == t:
            return [s]
        if s in blocked or t in blocked:
            return None
        visited = {s}
        queue = deque([[s]])
        while queue:
            path = queue.popleft()
            u = path[-1]
            for v in adj[u]:
                if v == t:
                    return path + [v]
                if v not in visited and v not in blocked:
                    visited.add(v)
                    queue.append(path + [v])
        return None
    
    def verify_clique_subdivision(branches):
        """Try to find vertex-disjoint paths for all pairs of branches.
        Returns (paths_dict, success) where paths_dict maps (u,v) -> path."""
        branches = sorted(branches)
        k = len(branches)
        if k < 2:
            return None, False
        
        branch_set = set(branches)
        
        # For each pair, we need a path. Interior vertices must be disjoint across all paths.
        # Greedy approach: process pairs in some order, find shortest path avoiding used interior vertices.
        
        # Try multiple orderings of the pairs
        pairs = list(combinations(branches, 2))
        
        # Order pairs by difficulty (fewest options first)
        # For each pair, count available paths
        def pair_difficulty(pair):
            u, v = pair
            # Count common neighbors not in branch_set
            common = adj[u] & adj[v] - branch_set
            return len(common)
        
        # Try several orderings
        orderings = [
            sorted(pairs, key=pair_difficulty),
            sorted(pairs, key=lambda p: -pair_difficulty(p)),
            pairs,
            list(reversed(pairs)),
        ]
        
        for ordering in orderings:
            if time_up():
                break
            used_interior = set()
            paths_found = {}
            success = True
            for pair in ordering:
                u, v = pair
                # Find path avoiding used_interior and all branches except endpoints
                blocked = branch_set - {u, v} | used_interior
                path = find_path(u, v, blocked)
                if path is None:
                    success = False
                    break
                # Add interior vertices to used
                for vert in path[1:-1]:
                    used_interior.add(vert)
                paths_found[pair] = path
            if success:
                return paths_found, True
        return None, False
    
    def greedy_clique_subdivision():
        """Greedy: start with a clique, try to add vertices."""
        # First find a large clique
        max_clique = find_max_clique()
        if not max_clique:
            return None
        
        best = None
        best_size = 0
        
        # Try the max clique
        paths, success = verify_clique_subdivision(max_clique)
        if success:
            best = (list(max_clique), paths)
            best_size = len(max_clique)
        
        # Try to extend: add vertices one by one
        # Sort vertices by degree descending
        vertices_by_degree = sorted(range(n), key=lambda v: -len(adj[v]))
        
        # Try starting from different base cliques
        # Also try non-clique branch sets where paths can go through intermediate vertices
        
        # Greedy addition
        current = list(max_clique)
        for v in vertices_by_degree:
            if time_up():
                break
            if v in current:
                continue
            new_set = current + [v]
            paths, success = verify_clique_subdivision(new_set)
            if success:
                current = new_set
                if len(current) > best_size:
                    best = (list(current), paths)
                    best_size = len(current)
        
        return best
    
    def search_clique_subdivisions():
        """Search over branch sets of various sizes."""
        best = None
        best_size = 0
        
        # First, get a good initial solution
        initial = greedy_clique_subdivision()
        if initial:
            best = initial
            best_size = len(initial[0])
        
        # Try to improve by searching for larger branch sets
        # Use the greedy result as a starting point
        if best:
            current_branches = set(best[0])
            
            # Try adding vertices
            for v in range(n):
                if time_up():
                    break
                if v in current_branches:
                    continue
                new_set = list(current_branches) + [v]
                paths, success = verify_clique_subdivision(new_set)
                if success:
                    current_branches.add(v)
                    if len(current_branches) > best_size:
                        best = (list(current_branches), paths)
                        best_size = len(current_branches)
        
        # Also try different starting points
        # Try all 4-cliques
        if not time_up():
            for quad in combinations(range(n), 4):
                if time_up():
                    break
                if is_clique(quad):
                    paths, success = verify_clique_subdivision(quad)
                    if success and len(quad) > best_size:
                        best = (list(quad), paths)
                        best_size = len(quad)
        
        # Try 5-cliques
        if not time_up():
            for quint in combinations(range(n), 5):
                if time_up():
                    break
                if is_clique(quint):
                    paths, success = verify_clique_subdivision(quint)
                    if success and len(quint) > best_size:
                        best = (list(quint), paths)
                        best_size = len(quint)
        
        return best
    
    # Main search
    result = search_clique_subdivisions()
    
    if result is None:
        # Fallback: just return an edge
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    branches, paths = result
    # Convert to proper types
    branches_list = [int(b) for b in branches]
    paths_list = {}
    for pair, path in paths.items():
        paths_list[(int(pair[0]), int(pair[1]))] = [int(v) for v in path]
    
    # Format: paths should be a dict mapping pair to path
    # But the task says 'paths': one simple path for each unordered pair of branches
    # Let's return as a list of (pair, path) or as a dict
    # Looking at the seed: 'paths':[list(e) for e in combinations(row,2)]
    # That's just a list of edges, not paths!
    # Wait, re-reading: "one simple path for each unordered pair of branches"
    # The seed returns paths as a list of 2-element lists (just the edges)
    # For a clique, the path between u and v is just [u, v]
    
    # Let me re-read the format more carefully
    # "Return a dict with exactly the keys 'branches' and 'paths'"
    # "Each path is a list starting and ending at its two branch vertices"
    # The seed returns 'paths' as a list, not a dict
    # So paths should be a list where each element corresponds to a pair?
    # Or a dict?
    
    # Looking at seed: 'paths':[list(e) for e in combinations(row,2)]
    # This is a list of lists. For 3 branches, combinations gives 3 pairs.
    # So it's a list of 3 paths, one for each pair.
    
    # I'll return paths as a list, ordered by combinations of branches
    paths_ordered = []
    for pair in combinations(branches_list, 2):
        # Find the path for this pair
        key = (min(pair), max(pair))
        if key in paths_list:
            paths_ordered.append(paths_list[key])
        else:
            # Shouldn't happen
            paths_ordered.append(list(pair))
    
    return {'branches': branches_list, 'paths': paths_ordered}

# EVOLVE-BLOCK-END
