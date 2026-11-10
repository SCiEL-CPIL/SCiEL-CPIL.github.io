"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    """
    Decompose each instance graph into edge-disjoint cycles and single edges,
    minimizing the total number of pieces (cycles + single edges).
    
    Strategy:
    1. Generate the instance graphs.
    2. For each graph, use a randomized local search approach:
       a. Start with a trivial decomposition into single edges.
       b. Iteratively attempt to merge edges into cycles to reduce the piece count.
       c. Use randomized restarts and local moves to escape local optima.
    3. Return the list of pieces for each graph.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin within 150s limit
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def find_long_cycle(adj, n, rng, max_iters=50):
        """
        Find a long simple cycle in the graph.
        Uses randomized DFS with backtracking.
        Returns list of vertices in the cycle or None.
        """
        if not adj:
            return None
        
        # Start from vertices with degree >= 2
        candidates = [v for v in range(n) if len(adj[v]) >= 2]
        if not candidates:
            return None
        
        best_cycle = None
        best_len = 2
        
        # Try multiple random starts
        for _ in range(max_iters):
            start = rng.choice(candidates)
            
            # DFS to find a cycle back to start
            path = [start]
            on_path = {start}
            found = False
            
            def dfs(current):
                nonlocal found, best_cycle, best_len, path, on_path
                if found:
                    return
                
                # Check if we can close the cycle
                if start in adj[current] and len(path) >= 3:
                    cur_len = len(path)
                    if cur_len > best_len:
                        best_len = cur_len
                        best_cycle = path[:]
                    found = True
                    return
                
                # Explore neighbors
                neighbors = list(adj[current])
                rng.shuffle(neighbors)
                
                for nb in neighbors:
                    if nb == start:
                        continue
                    if nb not in on_path:
                        on_path.add(nb)
                        path.append(nb)
                        dfs(nb)
                        path.pop()
                        on_path.discard(nb)
                        if found or best_len == n:
                            return
            
            dfs(start)
            if best_len == n:
                break
        
        return best_cycle
    
    def decompose_graph(edges, n, rng, time_budget=10.0):
        """
        Decompose the graph into cycles and single edges using randomized greedy + local search.
        Returns a list of pieces.
        """
        start_time = time.time()
        rem_edges = set(tuple(sorted(e)) for e in edges)
        
        # Build initial adjacency
        adj = {i: set() for i in range(n)}
        for u, v in rem_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        pieces = []
        
        # Phase 1: Greedy cycle extraction
        while rem_edges:
            if time.time() - start_time > time_budget - 2.0:
                break
            
            # Find a long cycle
            cycle = find_long_cycle(adj, n, rng, max_iters=20)
            
            if cycle is None:
                break
            
            # Remove the cycle's edges
            valid_cycle = True
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                if e not in rem_edges:
                    valid_cycle = False
                    break
            
            if valid_cycle:
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = tuple(sorted((u, v)))
                    rem_edges.discard(e)
                    adj[u].discard(v)
                    adj[v].discard(u)
                pieces.append(cycle)
            else:
                # If cycle is invalid, just take one edge
                if rem_edges:
                    e = next(iter(rem_edges))
                    rem_edges.discard(e)
                    u, v = e
                    adj[u].discard(v)
                    adj[v].discard(u)
                    pieces.append(list(e))
        
        # Phase 2: Local search - try to merge single edges into cycles
        # This is a heuristic attempt to improve the decomposition
        if time.time() - start_time < time_budget - 1.0:
            # Try to find cycles among the remaining single edges
            single_edges = [p for p in pieces if len(p) == 2]
            cycles = [p for p in pieces if len(p) >= 3]
            
            # Build adjacency from single edges only
            adj_single = {i: set() for i in range(n)}
            for e in single_edges:
                u, v = e
                adj_single[u].add(v)
                adj_single[v].add(u)
            
            # Try to find cycles in the single-edge subgraph
            new_pieces = []
            used_edges = set()
            
            for e in single_edges:
                if e in used_edges:
                    continue
                # Try to extend this edge into a cycle
                u, v = e
                # Look for a path from v back to u using only unused single edges
                # This is simplified; in practice, we'd do a proper search
                
            # For now, just keep the greedy result as it's already decent
            # A more sophisticated local search would go here
        
        # Remaining edges become single-edge pieces
        for e in rem_edges:
            pieces.append(list(e))
        
        return pieces
    
    def validate_pieces(pieces, edges_set):
        """Validate that pieces are edge-disjoint and cover all edges."""
        used_edges = set()
        for p in pieces:
            if len(p) == 2:
                e = tuple(sorted(p))
                if e in used_edges or e not in edges_set:
                    return False
                used_edges.add(e)
            else:
                if len(p) < 3 or len(set(p)) != len(p):
                    return False
                for i in range(len(p)):
                    u = p[i]
                    v = p[(i + 1) % len(p)]
                    e = tuple(sorted((u, v)))
                    if e in used_edges or e not in edges_set:
                        return False
                    used_edges.add(e)
        return used_edges == edges_set
    
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for spec in specs:
        if time.time() > deadline - 10.0:
            # Not enough time, return trivial solution for remaining
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        n = spec[0]
        edges_set = set(tuple(sorted(e)) for e in edges)
        
        # Try multiple random restarts
        best_pieces = None
        best_count = float('inf')
        
        # Allocate time for this graph
        graph_time = min(20.0, deadline - time.time() - 5.0)
        
        for attempt in range(5):
            if time.time() > deadline - 10.0:
                break
            
            rng = random.Random(spec[2] * 1000 + attempt)
            t_start = time.time()
            pieces = decompose_graph(edges, n, rng, time_budget=graph_time / 5.0)
            
            # Validate
            if validate_pieces(pieces, edges_set):
                count = len(pieces)
                if count < best_count:
                    best_count = count
                    best_pieces = pieces
            
            if time.time() - t_start > graph_time / 5.0:
                break
        
        if best_pieces is None:
            # Fallback
            best_pieces = [list(e) for e in edges]
        
        results.append(best_pieces)
    
    return results

# EVOLVE-BLOCK-END
