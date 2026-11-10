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
    
    def find_cycles_by_eulerian(edges):
        """
        Find all cycles in the graph using Eulerian trail decomposition.
        Returns a list of cycles (each cycle is a list of vertices).
        """
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cycles = []
        remaining = set()
        for u, v in edges:
            remaining.add(edge_key(u, v))
        
        if not remaining:
            return cycles
        
        while remaining:
            # Find a vertex with odd degree or any vertex with degree >= 2
            # Actually, we just need to find any cycle in the remaining graph
            # Start from any vertex with degree >= 2
            start = None
            for v in adj:
                if len(adj[v]) >= 2:
                    start = v
                    break
            
            if start is None:
                # No cycles left, break
                break
            
            # Find a cycle using DFS
            cycle = _find_cycle_from_start(adj, start, remaining)
            if cycle is None:
                # Should not happen if start has degree >= 2, but just in case
                # Remove one edge to make progress
                u, v = list(remaining)[0]
                remaining.discard(edge_key(u, v))
                adj[u].discard(v)
                adj[v].discard(u)
                continue
            
            # Remove the cycle edges from remaining
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                ek = edge_key(u, v)
                remaining.discard(ek)
                adj[u].discard(v)
                adj[v].discard(u)
            
            cycles.append(cycle)
        
        return cycles
    
    def _find_cycle_from_start(adj, start, remaining):
        """Find a simple cycle starting from 'start' using DFS."""
        visited = {start}
        path = [start]
        
        def dfs(current):
            for nb in list(adj[current]):
                if nb == start and len(path) >= 3:
                    return path[:]
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    result = dfs(nb)
                    if result is not None:
                        return result
                    path.pop()
                    visited.discard(nb)
            return None
        
        return dfs(start)
    
    def decompose_via_cycles_and_remaining(edges, time_limit):
        """
        Decompose edges into cycles and single edges.
        First extract all cycles, then remaining edges become single-edge pieces.
        """
        cycles = find_cycles_by_eulerian(edges)
        
        # Edges used in cycles
        used_edges = set()
        for cycle in cycles:
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                used_edges.add(edge_key(u, v))
        
        # Remaining edges
        remaining_edges = set()
        for u, v in edges:
            ek = edge_key(u, v)
            if ek not in used_edges:
                remaining_edges.add(ek)
        
        pieces = []
        # Add cycles
        for cycle in cycles:
            pieces.append(cycle[:])
        
        # Add remaining edges as single-edge pieces
        for u, v in remaining_edges:
            pieces.append([u, v])
        
        return pieces
    
    def greedy_longest_cycle(edges, time_limit, seed):
        """
        Greedy approach: repeatedly find the longest cycle, remove it, repeat.
        """
        random.seed(seed)
        adj = defaultdict(set)
        remaining = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            remaining.add(edge_key(u, v))
        
        pieces = []
        max_iterations = 200
        
        for _ in range(max_iterations):
            if time.time() > time_limit:
                break
            if not remaining:
                break
            
            # Find a vertex with degree >= 2
            candidates = [v for v in adj if len(adj[v]) >= 2]
            if not candidates:
                break
            
            # Try to find the longest cycle starting from each candidate
            best_cycle = None
            best_len = 0
            
            for start in candidates:
                cycle = _find_longest_cycle_dfs(adj, start)
                if cycle is not None and len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle
                if best_len >= 20:  # Good enough
                    break
            
            if best_cycle is None:
                # No cycle found, break
                break
            
            # Remove the cycle
            for i in range(len(best_cycle)):
                u = best_cycle[i]
                v = best_cycle[(i + 1) % len(best_cycle)]
                ek = edge_key(u, v)
                remaining.discard(ek)
                adj[u].discard(v)
                adj[v].discard(u)
            
            pieces.append(best_cycle[:])
        
        # Add remaining edges as single-edge pieces
        for u, v in remaining:
            pieces.append([u, v])
        
        return pieces
    
    def _find_longest_cycle_dfs(adj, start):
        """Find the longest simple cycle starting from 'start' using DFS."""
        best_cycle = None
        best_len = 0
        
        def dfs(current, path, visited):
            nonlocal best_cycle, best_len
            for nb in adj[current]:
                if nb == start and len(path) >= 3:
                    if len(path) > best_len:
                        best_len = len(path)
                        best_cycle = path[:]
                elif nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    dfs(nb, path, visited)
                    path.pop()
                    visited.discard(nb)
        
        visited = {start}
        path = [start]
        dfs(start, path, visited)
        
        return best_cycle
    
    def local_search_improve(pieces, edges, time_limit):
        """
        Try to merge adjacent single-edge pieces into cycles if possible.
        """
        # Build a map from edge to piece index
        edge_to_piece = {}
        for i, piece in enumerate(pieces):
            if len(piece) == 2:
                ek = edge_key(piece[0], piece[1])
                edge_to_piece[ek] = i
            else:
                for j in range(len(piece)):
                    u = piece[j]
                    v = piece[(j + 1) % len(piece)]
                    ek = edge_key(u, v)
                    edge_to_piece[ek] = i
        
        # Try to merge two single edges into a path, then check if the closing edge exists
        # This is complex, so we'll skip for now and just return pieces as-is
        # A more sophisticated approach would try to find paths that can be closed
        
        return pieces
    
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
        
        # Time allocation for this instance
        time_per_instance = min(45, deadline - time.time() - 5)
        if time_per_instance < 5:
            time_per_instance = 5
        
        # Try multiple approaches and keep the best
        best_pieces = None
        best_count = float('inf')
        
        # Approach 1: Eulerian cycle extraction
        trial_deadline = time.time() + min(10, time_per_instance / 3)
        pieces1 = decompose_via_cycles_and_remaining(edges, trial_deadline)
        count1 = len(pieces1)
        if count1 < best_count:
            best_count = count1
            best_pieces = pieces1
        
        # Approach 2: Greedy longest cycle with different seeds
        for seed in [42, 123, 456, 789, 1011]:
            if time.time() > deadline - 10:
                break
            
            trial_deadline = time.time() + min(8, time_per_instance / 5)
            pieces2 = greedy_longest_cycle(edges, trial_deadline, seed)
            count2 = len(pieces2)
            if count2 < best_count:
                best_count = count2
                best_pieces = pieces2
        
        # Approach 3: Randomized greedy with more trials
        for t in range(20):
            if time.time() > deadline - 5:
                break
            
            seed = 5000 + t
            trial_deadline = time.time() + min(5, time_per_instance / 10)
            pieces3 = greedy_longest_cycle(edges, trial_deadline, seed)
            count3 = len(pieces3)
            if count3 < best_count:
                best_count = count3
                best_pieces = pieces3
        
        if best_pieces is None:
            best_pieces = [list(e) for e in edges]
        
        results.append(best_pieces)
    
    return results
# EVOLVE-BLOCK-END
