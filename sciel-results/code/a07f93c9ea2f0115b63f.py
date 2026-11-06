import time
import sys
from collections import defaultdict
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_greedy_longest(edges, seed_offset=0):
        if not edges:
            return []

        # Build graph structure
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(tuple(sorted(e)) for e in edges)
        paths = []
        
        # We will repeatedly extract the longest simple path from the remaining graph
        # To find a "long" path, we use a greedy DFS with a specific ordering.
        # We try multiple starting nodes and keep the best path found in each iteration.
        
        while remaining_edges:
            # Get all nodes with remaining edges
            active_nodes = set()
            for u, v in remaining_edges:
                active_nodes.add(u)
                active_nodes.add(v)
            
            if not active_nodes:
                break

            best_path = []
            
            # Try starting from each active node (or a subset if N is large, but N<=30 is small)
            # To save time, we can just try a few random starts or all if fast enough.
            # With N=30, trying all 30 starts with a greedy DFS is fast.
            
            # Shuffle active nodes for randomization if needed, but deterministic is fine for first pass
            # We use a specific heuristic: start from high degree nodes
            nodes_sorted = sorted(active_nodes, key=lambda x: -len(adj[x]))
            
            for start_node in nodes_sorted:
                # Greedy DFS to find a long simple path
                # We want to maximize the length of the path.
                # Heuristic: Always move to the neighbor that has the most remaining edges (degree in remaining graph)
                # to keep options open, or least? 
                # "Most remaining edges" usually leads to longer paths in dense graphs.
                # "Least remaining edges" (Dead-end first) is used for Eulerian paths, but here we want simple paths.
                # Let's try "Max Degree" in remaining graph.
                
                path = [start_node]
                visited = {start_node}
                current = start_node
                
                while True:
                    # Get neighbors in the remaining graph
                    candidates = []
                    for nb in adj[current]:
                        edge = tuple(sorted((current, nb)))
                        if edge in remaining_edges and nb not in visited:
                            # Calculate "value" of this move
                            # Degree of nb in remaining graph
                            deg_nb = 0
                            for u, v in remaining_edges:
                                if u == nb or v == nb:
                                    deg_nb += 1
                            candidates.append((deg_nb, nb))
                    
                    if not candidates:
                        break
                    
                    # Sort by degree descending (greedy choice to stay in dense area)
                    candidates.sort(reverse=True)
                    next_node = candidates[0][1]
                    
                    # Move
                    edge = tuple(sorted((current, next_node)))
                    remaining_edges_copy = remaining_edges # We don't remove yet, just checking validity
                    # We must ensure the edge is actually available
                    if edge in remaining_edges:
                        path.append(next_node)
                        visited.add(next_node)
                        current = next_node
                    else:
                        break
                
                if len(path) > len(best_path):
                    best_path = path
            
            if not best_path:
                # Fallback: pick any edge
                e = next(iter(remaining_edges))
                best_path = [e[0], e[1]]
            else:
                # Ensure best_path is valid and uses existing edges
                # (The DFS logic above ensures this)
                pass

            # Extract the best_path found
            if len(best_path) > 1:
                paths.append(best_path)
                # Remove edges from remaining_edges
                for i in range(len(best_path) - 1):
                    u, v = best_path[i], best_path[i+1]
                    edge = tuple(sorted((u, v)))
                    remaining_edges.discard(edge)
            else:
                # Should not happen if remaining_edges is not empty, but safe guard
                break

        return paths

    def decompose_multi_start(edges, num_restarts=5):
        if not edges:
            return []
        
        best_result = None
        best_count = float('inf')
        
        # Run greedy longest path multiple times with different randomizations
        # To randomize, we can shuffle the order of nodes considered or the tie-breaking in candidates
        
        for i in range(num_restarts):
            # We need a version of decompose_greedy_longest that accepts a random seed
            # For simplicity, we'll just call the deterministic one first.
            # If we have time, we can add randomization.
            
            # Let's implement a randomized version inline
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            
            remaining_edges = set(tuple(sorted(e)) for e in edges)
            paths = []
            rng = random.Random(42 + i)
            
            while remaining_edges:
                active_nodes = set()
                for u, v in remaining_edges:
                    active_nodes.add(u)
                    active_nodes.add(v)
                
                if not active_nodes:
                    break

                best_path = []
                
                # Randomly select a subset of start nodes to try, or all
                # For N=30, all is fine.
                # Shuffle the order of processing starts to introduce randomness in "first best"
                nodes_list = list(active_nodes)
                rng.shuffle(nodes_list)
                
                for start_node in nodes_list:
                    path = [start_node]
                    visited = {start_node}
                    current = start_node
                    
                    while True:
                        candidates = []
                        for nb in adj[current]:
                            edge = tuple(sorted((current, nb)))
                            if edge in remaining_edges and nb not in visited:
                                deg_nb = 0
                                for u, v in remaining_edges:
                                    if u == nb or v == nb:
                                        deg_nb += 1
                                # Add random noise to break ties
                                noise = rng.random() * 0.1
                                candidates.append((deg_nb + noise, nb))
                        
                        if not candidates:
                            break
                        
                        candidates.sort(reverse=True)
                        next_node = candidates[0][1]
                        
                        edge = tuple(sorted((current, next_node)))
                        if edge in remaining_edges:
                            path.append(next_node)
                            visited.add(next_node)
                            current = next_node
                        else:
                            break
                    
                    if len(path) > len(best_path):
                        best_path = path
                
                if not best_path:
                    e = next(iter(remaining_edges))
                    best_path = [e[0], e[1]]

                if len(best_path) > 1:
                    paths.append(best_path)
                    for i in range(len(best_path) - 1):
                        u, v = best_path[i], best_path[i+1]
                        edge = tuple(sorted((u, v)))
                        remaining_edges.discard(edge)
                else:
                    break
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_result = paths
                
            # Check time
            if time.time() > deadline - 10:
                break

        return best_result if best_result else []

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            break
        edges = instance(*spec)
        
        # Use multi-start greedy
        # Adjust restarts based on time remaining
        time_left = deadline - time.time()
        if time_left > 50:
            restarts = 10
        elif time_left > 20:
            restarts = 5
        else:
            restarts = 2
            
        paths = decompose_multi_start(edges, restarts)
        results.append(paths)
    
    # If we ran out of time, fill remaining with trivial edge paths
    while len(results) < 3:
        spec = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])
    
    return results
