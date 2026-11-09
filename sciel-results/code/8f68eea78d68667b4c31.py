import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit, 10s safety margin
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_graph(edges):
        if not edges:
            return []
        
        # Use a set for fast removal
        remaining = set(edges)
        paths = []
        
        # Build adjacency list for quick neighbor lookup
        # Note: We need to update this as edges are removed, 
        # or just check membership in 'remaining' each time.
        # For n=30, checking membership is O(1) and fast enough.
        
        while remaining:
            if time.time() > deadline:
                # Return whatever we have so far, but we need a valid decomposition.
                # If we timeout, we should probably just append remaining edges as single paths to be valid,
                # though the score will be worse. But given the small size, this won't happen.
                for e in remaining:
                    paths.append([e[0], e[1]])
                break
            
            extended = False
            # Shuffle paths to avoid bias towards early paths
            indices = list(range(len(paths)))
            random.shuffle(indices)
            
            for i in indices:
                if not remaining:
                    break
                path = paths[i]
                if len(path) < 2:
                    continue
                    
                # Check start
                u = path[0]
                # Find a neighbor v such that (u,v) or (v,u) is in remaining
                # We can just iterate over remaining edges? No, that's O(E).
                # Better: iterate over neighbors of u in the original graph?
                # We don't have the original graph structure easily accessible here without rebuilding.
                # Let's build a neighbor map once for the full graph.
                
                # Wait, the loop above is inside while remaining. 
                # We should build the neighbor map outside.
                
                # Let's refactor solve_graph to accept the graph structure.
                # Actually, the current inner loop is inefficient if we scan all remaining edges for each endpoint.
                # Let's pass the adjacency list.
                
                # Re-implementing solve_graph with adjacency list is better.
                # But for now, let's stick to a simpler logic that is still fast enough for n=30.
                # E is ~100. Paths ~10. Endpoints ~20.
                # Checking each endpoint against all remaining edges is 20 * 100 = 2000 ops per iteration.
                # Iterations ~ E = 100. Total 200,000 ops. Very fast.
                
                # Find edge connected to u
                for e in list(remaining):
                    if e[0] == u or e[1] == u:
                        # Found an edge
                        v = e[1] if e[0] == u else e[0]
                        # Extend path from start
                        path.insert(0, v)
                        remaining.remove(e)
                        extended = True
                        break
                        
                if extended:
                    break
                    
                # Check end
                if extended:
                    continue
                    
                u = path[-1]
                for e in list(remaining):
                    if e[0] == u or e[1] == u:
                        v = e[1] if e[0] == u else e[0]
                        # Extend path from end
                        path.append(v)
                        remaining.remove(e)
                        extended = True
                        break
                        
                if extended:
                    break
            
            if not extended:
                # No path could be extended. Start a new one.
                # Pick an arbitrary edge
                e = next(iter(remaining))
                paths.append([e[0], e[1]])
                remaining.remove(e)
                
        return paths

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback: just use single edges for remaining specs
            edges = instance(*spec)
            results.append([[e[0], e[1]] for e in edges])
            continue
            
        edges = instance(*spec)
        # Optimization: The simple greedy above might get stuck in local minima where it creates many short paths.
        # A better heuristic: Prioritize extending paths that have endpoints with high degree in the remaining graph.
        # Let's implement a slightly smarter version.
        
        # Build adjacency list for the full graph
        adj = {i: set() for i in range(spec[0])}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining = set(edges)
        paths = []
        
        # Pre-compute degrees? Degrees change.
        # We will just use the simple greedy but with a twist:
        # When choosing which path to extend, pick the one with the endpoint having the highest remaining degree.
        
        while remaining:
            if time.time() > deadline:
                for e in remaining:
                    paths.append([e[0], e[1]])
                break
                
            best_path_idx = -1
            best_endpoint = -1
            best_neighbor = -1
            best_degree = -1
            
            # Evaluate all possible extensions
            for i, path in enumerate(paths):
                if len(path) < 2:
                    continue
                    
                # Check start
                u = path[0]
                # Find neighbors of u in remaining graph
                # We need to know which neighbors are still connected via remaining edges.
                # This requires checking edges.
                # To speed this up, we can maintain a set of remaining neighbors for each node?
                # That's complex to update.
                # Given n=30, E~100, checking is fine.
                
                # For each potential neighbor v of u (from original adj), check if edge (u,v) is in remaining
                for v in adj[u]:
                    e = tuple(sorted((u, v)))
                    if e in remaining:
                        # Calculate "score" for this extension.
                        # Higher remaining degree of v is better?
                        # Or just pick one.
                        # Let's pick the one with the highest remaining degree of the new endpoint.
                        # Remaining degree of v: count edges in remaining incident to v.
                        # This is O(E) per check.
                        # Total checks: Paths * 2 * Degree.
                        # 10 * 2 * 5 = 100 checks. Each check O(E)=100. Total 10k ops per iteration.
                        # 100 iterations. 1M ops. Fast.
                        
                        # Count remaining degree of v
                        deg_v = 0
                        for e2 in remaining:
                            if e2[0] == v or e2[1] == v:
                                deg_v += 1
                                
                        if deg_v > best_degree:
                            best_degree = deg_v
                            best_path_idx = i
                            best_endpoint = 0 # Start
                            best_neighbor = v
                            
                # Check end
                u = path[-1]
                for v in adj[u]:
                    e = tuple(sorted((u, v)))
                    if e in remaining:
                        deg_v = 0
                        for e2 in remaining:
                            if e2[0] == v or e2[1] == v:
                                deg_v += 1
                                
                        if deg_v > best_degree:
                            best_degree = deg_v
                            best_path_idx = i
                            best_endpoint = 1 # End
                            best_neighbor = v
            
            if best_path_idx != -1:
                path = paths[best_path_idx]
                if best_endpoint == 0:
                    path.insert(0, best_neighbor)
                else:
                    path.append(best_neighbor)
                
                e = tuple(sorted((path[best_endpoint], best_neighbor)))
                remaining.remove(e)
            else:
                # Start new path
                e = next(iter(remaining))
                paths.append([e[0], e[1]])
                remaining.remove(e)
                
        results.append(paths)

    return results
