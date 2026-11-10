"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    
    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def edge_key(u, v):
        return (min(u, v), max(u, v))
        
    def find_longest_cycle(adj, start, max_depth, time_limit):
        """
        Find the longest cycle starting and ending at 'start' using DFS with pruning.
        Returns the path (list of vertices) or None.
        """
        best_path = None
        best_len = 0
        
        # We use a stack-based approach to avoid recursion depth issues and allow better control
        # State: (current_node, path_list, visited_set)
        # To optimize, we only explore neighbors that can potentially lead to a longer path
        # or are unvisited.
        
        # Simple recursive DFS with backtracking is often fastest for small graphs (n<=30)
        # if we prune aggressively.
        
        visited = {start}
        path = [start]
        
        def dfs(current, depth):
            nonlocal best_path, best_len
            
            if time.time() > time_limit:
                return False
                
            # If we've exceeded max depth, stop
            if depth > max_depth:
                return False
                
            # Check if we can close the cycle
            # A cycle is formed if 'current' is connected to 'start' and path length >= 3
            if start in adj[current] and len(path) >= 3:
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path[:]
                    
            # Explore neighbors
            # Heuristic: prioritize neighbors with higher degree to find longer cycles?
            # Or randomize to escape local minima?
            neighbors = list(adj[current])
            random.shuffle(neighbors)
            
            for nb in neighbors:
                if nb == start:
                    continue # Already handled above
                    
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    dfs(nb, depth + 1)
                    path.pop()
                    visited.remove(nb)
                    
            return True
            
        dfs(start, 0)
        
        return best_path

    def decompose_graph(edges, time_limit, seed):
        """
        Decompose graph into cycles and edges.
        Strategy:
        1. Repeatedly find the longest cycle in the remaining graph.
        2. Remove it.
        3. If no cycle exists, remove single edges.
        4. Post-processing: Try to merge adjacent single edges into cycles if possible.
        """
        random.seed(seed)
        remaining = set()
        for e in edges:
            remaining.add(edge_key(e[0], e[1]))
        
        if not remaining:
            return []
        
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        pieces = []
        
        def commit_cycle(cycle_vertices):
            for i in range(len(cycle_vertices)):
                u = cycle_vertices[i]
                v = cycle_vertices[(i + 1) % len(cycle_vertices)]
                ek = edge_key(u, v)
                remaining.discard(ek)
                adj[u].discard(v)
                adj[v].discard(u)
            pieces.append(cycle_vertices[:])
            
        def commit_single_edge(u, v):
            ek = edge_key(u, v)
            remaining.discard(ek)
            adj[u].discard(v)
            adj[v].discard(u)
            pieces.append([u, v])
            
        # Phase 1: Greedy Longest Cycle Extraction
        # To avoid getting stuck in a bad local minimum, we use a limited number of restarts
        # with different random seeds for the neighbor ordering, but we prioritize length.
        
        max_iterations = 100
        iterations = 0
        
        while remaining and iterations < max_iterations:
            if time.time() > time_limit:
                break
            iterations += 1
            
            # Identify candidate vertices with degree >= 2
            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                # No more cycles possible
                for u, v in list(remaining):
                    commit_single_edge(u, v)
                break
                
            # To find the *longest* cycle, we should try starting from different vertices.
            # However, searching from every vertex is expensive.
            # Heuristic: Pick a random candidate, but also try the one with highest degree.
            
            # Sort candidates by degree descending, but take a random subset to ensure diversity
            candidates.sort(key=lambda x: len(adj[x]), reverse=True)
            
            # Try top 5 candidates for a cycle
            best_cycle = None
            best_len = 0
            
            # Time budget per iteration
            iter_deadline = time.time() + (time_limit - time.time()) / (max_iterations - iterations + 1) * 0.8
            
            for start in candidates[:5]:
                if time.time() > iter_deadline:
                    break
                
                # Find longest cycle starting from this vertex
                # Limit depth to avoid exponential blowup on dense graphs
                # n is small (<=30), so depth 30 is safe
                cycle = find_longest_cycle(adj, start, 30, iter_deadline)
                
                if cycle and len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle
                    
            if best_cycle:
                commit_cycle(best_cycle)
            else:
                # If no cycle found (shouldn't happen if degree >= 2 exists, unless graph is bipartite and we missed it? No, DFS finds all simple paths)
                # Actually, if degree >= 2, a cycle MUST exist. 
                # If find_longest_cycle returns None, it might be due to time limit or bug.
                # Fallback: Just take a random edge to ensure progress, or try a simpler BFS cycle finder.
                
                # Simple BFS cycle finder as fallback
                cycle_found = False
                for start in candidates:
                    # BFS from start to find any cycle
                    visited = {start: -1}
                    parent = {start: None}
                    queue = deque([start])
                    
                    while queue:
                        if time.time() > iter_deadline:
                            break
                        u = queue.popleft()
                        for v in adj[u]:
                            if v not in visited:
                                visited[v] = u
                                parent[v] = u
                                queue.append(v)
                            elif v != parent[u] and v == start:
                                # Found a cycle
                                # Reconstruct path
                                path = []
                                node = u
                                while node != start:
                                    path.append(node)
                                    node = parent[node]
                                path.append(start)
                                path.reverse()
                                if len(path) >= 3:
                                    best_cycle = path
                                    cycle_found = True
                                    break
                        if cycle_found:
                            break
                    if cycle_found:
                        break
                        
                if best_cycle:
                    commit_cycle(best_cycle)
                else:
                    # Emergency: remove an edge
                    u, v = list(remaining)[0]
                    commit_single_edge(u, v)
                    
        # Any remaining edges become single-edge pieces
        for u, v in list(remaining):
            commit_single_edge(u, v)
            
        # Phase 2: Local Search / Repair
        # Try to combine single edges into cycles if possible.
        # This is complex to implement efficiently in the remaining time.
        # A simpler repair: check if any two single edges share a vertex and form a path of length 2, 
        # and if the endpoints are connected, form a triangle.
        
        # Identify single edges
        singles = [p for p in pieces if len(p) == 2]
        
        if len(singles) >= 2:
            # Build adjacency for singles
            single_adj = defaultdict(list)
            for i, e in enumerate(singles):
                u, v = e[0], e[1]
                single_adj[u].append((i, v))
                single_adj[v].append((i, u))
                
            # Try to find triangles formed by 3 single edges
            # Or extend a path of 2 singles with a 3rd?
            # Given time constraints, we skip complex repair and rely on the greedy phase being good enough.
            # The greedy longest cycle approach is significantly better than arbitrary cycle finding.
            
        return pieces

    def decompose_graph_multi_seed(edges, time_limit, base_seed):
        """
        Run decomposition with multiple seeds and keep the best result.
        """
        best_pieces = None
        best_count = float('inf')
        
        # Allocate time for trials
        num_trials = 5
        trial_time = time_limit / num_trials
        
        for t in range(num_trials):
            if time.time() > time_limit - 1:
                break
                
            seed = base_seed + t
            trial_deadline = time.time() + trial_time
            
            pieces = decompose_graph(edges, trial_deadline, seed)
            
            count = len(pieces)
            if count < best_count:
                best_count = count
                best_pieces = pieces
                
        if best_pieces is None:
            # Fallback: all edges as singles
            best_pieces = [list(e) for e in edges]
            
        return best_pieces

    # Main: decompose each instance
    results = []
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    
    for i, spec in enumerate(specs):
        if time.time() > deadline - 15:
            # Not enough time, use trivial solution
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        
        # Allocate time for this instance
        # Give more time to later instances if earlier ones are fast?
        # Or equal split?
        # Let's use equal split of remaining time minus safety margin
        remaining_time = deadline - time.time() - 5
        time_per_instance = remaining_time / (len(specs) - i)
        
        if time_per_instance < 5:
            time_per_instance = 5
            
        pieces = decompose_graph_multi_seed(edges, time_per_instance, 1000 + i)
        
        results.append(pieces)
        
    return results
# EVOLVE-BLOCK-END
