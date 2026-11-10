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
    
    def find_cycle_bfs(adj, start, time_limit):
        """Find a cycle containing start using BFS from neighbors of start."""
        # Try each neighbor of start as potential end of path
        neighbors = list(adj[start])
        random.shuffle(neighbors)
        
        for target in neighbors:
            # BFS from start to target, avoiding direct edge start-target
            visited = {start}
            parent = {start: None}
            queue = deque([start])
            
            while queue:
                if time.time() > time_limit:
                    return None
                current = queue.popleft()
                if current == target:
                    # Reconstruct path
                    path = []
                    node = current
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    # Check path length >= 3
                    if len(path) >= 3:
                        return path
                    continue
                for nb in adj[current]:
                    if nb not in visited:
                        visited.add(nb)
                        parent[nb] = current
                        queue.append(nb)
        return None
    
    def decompose_with_cycles(edges, time_limit, seed):
        """Decompose edges into cycles and single edges using greedy cycle extraction."""
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
        
        # Repeatedly find and remove cycles
        max_iterations = 1000
        iterations = 0
        
        while remaining and iterations < max_iterations:
            if time.time() > time_limit:
                break
            iterations += 1
            
            # Find vertices with degree >= 2
            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                # No cycles left, commit all remaining as singles
                for u, v in list(remaining):
                    commit_single_edge(u, v)
                break
            
            # Try to find a cycle from a random candidate
            random.shuffle(candidates)
            best_cycle = None
            
            # Try a few random starting vertices
            max_starts = min(10, len(candidates))
            for start in candidates[:max_starts]:
                if time.time() > time_limit:
                    break
                
                # Try to find a cycle using DFS with randomization
                # Use a randomized depth-first approach
                cycle = _find_cycle_dfs(adj, start, time_limit)
                if cycle is not None:
                    if best_cycle is None or len(cycle) > len(best_cycle):
                        best_cycle = cycle
                        if len(cycle) >= 10:
                            break  # Good enough, stop searching
            
            if best_cycle:
                commit_cycle(best_cycle)
            else:
                # If no cycle found, take a single edge to ensure progress
                u, v = list(remaining)[0]
                commit_single_edge(u, v)
        
        # Any remaining edges become single-edge pieces
        for u, v in list(remaining):
            commit_single_edge(u, v)
            
        return pieces
    
    def _find_cycle_dfs(adj, start, time_limit):
        """Find a cycle containing start using randomized DFS."""
        best_cycle = None
        best_len = 0
        
        path = [start]
        visited = {start}
        
        def dfs(current, depth):
            nonlocal best_cycle, best_len
            if time.time() > time_limit:
                return False
            if depth >= 30:  # Max cycle length
                return False
            
            neighbors = list(adj[current])
            random.shuffle(neighbors)
            
            for nb in neighbors:
                if nb == start and len(path) >= 3:
                    if len(path) > best_len:
                        best_len = len(path)
                        best_cycle = path[:]
                    return True
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    if dfs(nb, depth + 1):
                        return True
                    path.pop()
                    visited.discard(nb)
            return False
        
        dfs(start, 0)
        return best_cycle
    
    def greedy_matching_covers(edges, time_limit, seed):
        """Alternative: find long cycles via randomized walks."""
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
        
        max_iterations = 1000
        iterations = 0
        
        while remaining and iterations < max_iterations:
            if time.time() > time_limit:
                break
            iterations += 1
            
            # Pick a random vertex with degree >= 2
            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                for u, v in list(remaining):
                    commit_single_edge(u, v)
                break
            
            start = random.choice(candidates)
            
            # Random walk to find a cycle
            cycle = _random_walk_cycle(adj, start, time_limit)
            
            if cycle:
                commit_cycle(cycle)
            else:
                u, v = list(remaining)[0]
                commit_single_edge(u, v)
        
        for u, v in list(remaining):
            commit_single_edge(u, v)
            
        return pieces
    
    def _random_walk_cycle(adj, start, time_limit):
        """Find a cycle via random walk from start."""
        for _ in range(50):  # Try 50 random walks
            if time.time() > time_limit:
                return None
            
            path = [start]
            visited = {start}
            current = start
            
            for _ in range(100):  # Max walk length
                if time.time() > time_limit:
                    return None
                if not adj[current]:
                    break
                
                # Choose random neighbor
                neighbors = list(adj[current])
                nb = random.choice(neighbors)
                
                if nb == start and len(path) >= 3:
                    return path
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    current = nb
                else:
                    # Dead end, restart
                    break
        
        return None
    
    # Main: decompose each instance
    results = []
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    
    for spec in specs:
        if time.time() > deadline - 10:
            # Not enough time, use trivial solution
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        
        # Try multiple random seeds and keep the best
        best_pieces = None
        best_count = float('inf')
        
        # Allocate time for this instance
        time_per_instance = min(45, deadline - time.time() - 10)
        if time_per_instance < 5:
            time_per_instance = 5
            
        # Run multiple trials with different random seeds
        num_trials = 30
        trial_time = time_per_instance / num_trials
        
        for t in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            # Set a specific seed for reproducibility within the trial
            seed = 3000 + t
            
            # Time limit for this specific trial
            trial_deadline = time.time() + trial_time
            
            # Alternate between two decomposition strategies
            if t % 2 == 0:
                pieces = decompose_with_cycles(edges, trial_deadline, seed)
            else:
                pieces = greedy_matching_covers(edges, trial_deadline, seed)
            
            count = len(pieces)
            
            if count < best_count:
                best_count = count
                best_pieces = pieces
        
        if best_pieces is None:
            best_pieces = [list(e) for e in edges]
        
        results.append(best_pieces)
    
    return results
# EVOLVE-BLOCK-END
