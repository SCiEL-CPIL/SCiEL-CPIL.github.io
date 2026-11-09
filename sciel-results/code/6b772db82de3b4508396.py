import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    graphs = [instance(*spec) for spec in specs]
    
    def solve_graph(edges, max_iters=200):
        if not edges:
            return []
        
        best_paths = None
        best_count = float('inf')
        
        # Initial degree calculation for reference, but we will update dynamically
        # We will use a randomized greedy approach with multiple restarts
        
        for iter_idx in range(max_iters):
            # Check time
            if time.time() > deadline - 5:
                break
                
            # Copy edges to a mutable structure
            # Using a set of tuples for fast lookup and removal
            remaining_edges = set(edges)
            
            # Build adjacency list
            adj = defaultdict(set)
            for u, v in remaining_edges:
                adj[u].add(v)
                adj[v].add(u)
            
            paths = []
            
            while remaining_edges:
                # Find a vertex with the highest degree
                # If multiple, pick one randomly to allow different outcomes in restarts
                max_deg = -1
                candidates = []
                for v in adj:
                    deg = len(adj[v])
                    if deg > 0:
                        if deg > max_deg:
                            max_deg = deg
                            candidates = [v]
                        elif deg == max_deg:
                            candidates.append(v)
                
                if not candidates:
                    break # Should not happen if remaining_edges is not empty and adj is consistent
                
                start_v = random.choice(candidates)
                
                # Construct a path starting from start_v
                path = [start_v]
                prev_v = start_v
                used_edges_in_path = set()
                
                # Mark start_v as visited for this path to ensure simplicity
                visited_in_path = {start_v}
                
                while True:
                    # Find neighbors of prev_v in remaining_edges
                    # We need to find an edge (prev_v, next_v) in remaining_edges
                    # such that next_v is not in visited_in_path
                    next_v = None
                    best_next_deg = -1
                    
                    # Get all neighbors from the global adj list
                    # Note: adj contains all edges, but we must check if the specific edge is in remaining_edges
                    # Actually, it's more efficient to iterate over remaining_edges or maintain a proper dynamic adj.
                    # For n<=30, iterating over adj[prev_v] is fine.
                    
                    for neighbor in list(adj[prev_v]):
                        edge = tuple(sorted((prev_v, neighbor)))
                        if edge in remaining_edges:
                            if neighbor not in visited_in_path:
                                deg = len(adj[neighbor])
                                # Heuristic: Prefer neighbors with higher degree
                                # Add some randomness if degrees are equal
                                if deg > best_next_deg:
                                    best_next_deg = deg
                                    next_v = neighbor
                                elif deg == best_next_deg:
                                    # Random tie-break
                                    if random.random() < 0.5:
                                        next_v = neighbor
                                        
                    if next_v is None:
                        break
                    
                    # Commit to moving to next_v
                    path.append(next_v)
                    visited_in_path.add(next_v)
                    edge_to_remove = tuple(sorted((prev_v, next_v)))
                    remaining_edges.remove(edge_to_remove)
                    
                    # Update adj
                    adj[prev_v].remove(next_v)
                    adj[next_v].remove(prev_v)
                    
                    prev_v = next_v
            
            paths.append(path)
            
            count = len(paths)
            if count < best_count:
                best_count = count
                best_paths = paths
                
            # Early termination if we hit the theoretical lower bound?
            # Lower bound is roughly max(1, ceil((n-1)/2)) ? No, Gallai is (n+1)//2.
            # But we don't know the exact optimal for these specific graphs.
            # We can stop if we find a solution that is very good, e.g., <= (n+1)//2.
            # Let's just run the iterations.
            
        if best_paths is None:
            # Fallback: trivial solution
            return [[u, v] for u, v in edges]
            
        return best_paths

    results = []
    for i, edges in enumerate(graphs):
        if time.time() > deadline - 10:
            # Not enough time, use trivial solution for remaining
            results.append([[u, v] for u, v in edges])
            continue
            
        # Increase iterations for smaller graphs or if time allows
        # 200 iterations is usually fast enough for n=30
        paths = solve_graph(edges, max_iters=300)
        results.append(paths)
        
    return results

