import random
from collections import defaultdict

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_path_cover(edges):
        # Build adjacency list
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(edges)
        paths = []
        
        while remaining_edges:
            # Find the longest path in the graph defined by remaining_edges
            # We use a randomized greedy approach: try many random walks and keep the longest
            best_path = []
            best_len = 0
            
            # Number of trials
            trials = 100 
            nodes = list(adj.keys())
            
            for _ in range(trials):
                # Pick a random start node
                start = random.choice(nodes)
                visited = {start}
                path = [start]
                
                # Greedy extension
                current = start
                while True:
                    # Get neighbors of current in remaining graph
                    # We need to check if edge (current, neighbor) is in remaining_edges
                    neighbors = [n for n in adj[current] if (min(current, n), max(current, n)) in remaining_edges and n not in visited]
                    
                    if not neighbors:
                        break
                    
                    # Pick a random neighbor (or one with highest degree in remaining graph?)
                    # Let's pick the one with the most remaining edges (greedy)
                    # This is often better than random
                    def count_edges(node):
                        count = 0
                        for n in adj[node]:
                            if (min(node, n), max(node, n)) in remaining_edges:
                                count += 1
                        return count
                            
                    # Sort by remaining degree
                    neighbors.sort(key=count_edges, reverse=True)
                    
                    # Add some randomness: pick from top 3 or just the best?
                    # Let's pick the best to be deterministic-ish but fast
                    next_node = neighbors[0]
                    
                    # Add edge to path
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node
                    
            # Update best path
            if len(path) > best_len:
                best_len = len(path)
                best_path = path

            if not best_path:
                # Fallback: pick any edge
                e = next(iter(remaining_edges))
                paths.append(list(e))
                remaining_edges.remove(e)
                continue
            
            # Extract edges from best_path and remove them
            for i in range(len(best_path) - 1):
                u, v = best_path[i], best_path[i+1]
                edge = tuple(sorted((u, v)))
                remaining_edges.remove(edge)
                
            paths.append(best_path)
            
        return paths

    result = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = find_path_cover(edges)
        result.append(paths)
        
    return result
