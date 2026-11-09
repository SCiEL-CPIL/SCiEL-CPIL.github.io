import random
import time
from collections import defaultdict
from random import Random

def construct():
    start_time = time.time()
    deadline = time.time() + 140  # 140s limit, 10s buffer

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_cycle_in_subgraph(adj, current_edges, n, rng):
        best_cycle = []

        def dfs(u, path, visited):
            nonlocal best_cycle
            
            # Get valid neighbors
            neighbors = []
            for v in adj[u]:
                e = tuple(sorted((u, v)))
                if e in current_edges:
                    neighbors.append(v)
            
            if not neighbors:
                return

            # Separate unvisited and start
            unvisited = []
            start_found = False
            for v in neighbors:
                if v == path[0]:
                    start_found = True
                elif v not in visited:
                    unvisited.append(v)
            
            rng.shuffle(unvisited)
            
            # Explore unvisited
            for v in unvisited:
                visited.add(v)
                dfs(v, path + [v], visited)
                visited.remove(v)
                
                # Optional: Stop if we found a Hamiltonian cycle
                if len(best_cycle) >= n:
                    return

            # Close cycle if possible
            if start_found:
                cycle = path + [path[0]]
                if len(cycle) > len(best_cycle):
                    best_cycle = cycle

        # Pick a start node
        # Prefer nodes with higher degree in current subgraph?
        # For simplicity, random node with degree > 0.
        active_nodes = [u for u in range(n) if any(tuple(sorted((u, v))) in current_edges for v in adj[u])]
        if not active_nodes:
            return None
            
        # Randomize start
        start_node = rng.choice(active_nodes)
        
        dfs(start_node, [start_node], {start_node})
        
        if best_cycle:
            return best_cycle
        return None

    def decompose_graph(edges, n):
        if not edges:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        best_pieces = None
        best_score = float('inf')
        
        # Time-based restarts
        while time.time() < deadline - 2:  # Leave 2s for other graphs/overhead
            rng = Random(int(time.time() * 1000) % 100000) # Pseudo-random seed for variety
            # Better: use a counter for deterministic reproducibility in testing?
            # No, we want diversity.
            
            current_edges = set(edges)
            pieces = []
            
            while current_edges:
                cycle = find_cycle_in_subgraph(adj, current_edges, n, rng)
                
                if not cycle:
                    # No cycle, add remaining
                    for e in current_edges:
                        pieces.append(list(e))
                    break
                else:
                    pieces.append(cycle)
                    # Remove edges
                    for i in range(len(cycle) - 1):
                        u, v = cycle[i], cycle[i+1]
                        e = tuple(sorted((u, v)))
                        current_edges.remove(e)
                    u, v = cycle[-1], cycle[0]
                    e = tuple(sorted((u, v)))
                    current_edges.remove(e)
                    
            score = len(pieces)
            if score < best_score:
                best_score = score
                best_pieces = [list(p) for p in pieces]
                
        if best_pieces is None:
            # Fallback: all edges
            return [list(e) for e in edges]
            
        return best_pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    # Allocate time roughly proportionally to complexity?
    # Or just let the while loop handle it.
    # To ensure all graphs are processed, we should reserve time.
    # Let's do 100s for graph 1, 100s for graph 2, 100s for graph 3?
    # No, total 140s.
    # Let's do 40s, 40s, 60s.
    
    time_alloc = [40, 40, 60]
    
    for i, spec in enumerate(specs):
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        # Temporarily adjust deadline for this graph
        local_deadline = time.time() + time_alloc[i]
        
        # We need to pass local_deadline to decompose_graph
        # Let's refactor decompose_graph to take a deadline.
        
        # ... (Refactored below)
        
        pieces = decompose_graph_with_deadline(edges, n, local_deadline)
        results.append(pieces)
        
    return results

# Refactored decompose_graph to accept deadline
def construct():
    start_time = time.time()
    total_deadline = time.time() + 140

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_cycle_in_subgraph(adj, current_edges, n, rng):
        best_cycle = []

        def dfs(u, path, visited):
            nonlocal best_cycle
            
            neighbors = []
            for v in adj[u]:
                e = tuple(sorted((u, v)))
                if e in current_edges:
                    neighbors.append(v)
            
            if not neighbors:
                return

            unvisited = []
            start_found = False
            for v in neighbors:
                if v == path[0]:
                    start_found = True
                elif v not in visited:
                    unvisited.append(v)
            
            rng.shuffle(unvisited)
            
            for v in unvisited:
                visited.add(v)
                dfs(v, path + [v], visited)
                visited.remove(v)
                
                if len(best_cycle) >= n:
                    return

            if start_found:
                cycle = path + [path[0]]
                if len(cycle) > len(best_cycle):
                    best_cycle = cycle

        active_nodes = [u for u in range(n) if any(tuple(sorted((u, v))) in current_edges for v in adj[u])]
        if not active_nodes:
            return None
            
        start_node = rng.choice(active_nodes)
        dfs(start_node, [start_node], {start_node})
        
        if best_cycle:
            return best_cycle
        return None

    def decompose_graph(edges, n, deadline):
        if not edges:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        best_pieces = None
        best_score = float('inf')
        
        while time.time() < deadline:
            rng = Random()
            current_edges = set(edges)
            pieces = []
            
            while current_edges:
                cycle = find_cycle_in_subgraph(adj, current_edges, n, rng)
                
                if not cycle:
                    for e in current_edges:
                        pieces.append(list(e))
                    break
                else:
                    pieces.append(cycle)
                    for i in range(len(cycle) - 1):
                        u, v = cycle[i], cycle[i+1]
                        e = tuple(sorted((u, v)))
                        current_edges.remove(e)
                    u, v = cycle[-1], cycle[0]
                    e = tuple(sorted((u, v)))
                    current_edges.remove(e)
                    
            score = len(pieces)
            if score < best_score:
                best_score = score
                best_pieces = [list(p) for p in pieces]
                
        if best_pieces is None:
            return [list(e) for e in edges]
            
        return best_pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    # Allocate time: 45, 45, 50
    time_alloc = [45, 45, 50]
    
    for i, spec in enumerate(specs):
        n, p, seed = spec
        edges = instance(n, p, seed)
        local_deadline = time.time() + time_alloc[i]
        pieces = decompose_graph(edges, n, local_deadline)
        results.append(pieces)
        
    return results
