import time
import random
from collections import defaultdict

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Generate the three fixed instance graphs
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    graphs = [instance(*spec) for spec in specs]
    
    results = []
    
    for edges in graphs:
        if time.time() > deadline - 10.0:
            # Fallback: just return edges if we are out of time
            results.append([list(e) for e in edges])
            continue
            
        # Build adjacency list
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Set of remaining edges
        remaining_edges = set(tuple(e) for e in edges)
        
        # Function to check if an edge is available
        def is_edge(u, v):
            return (u, v) in remaining_edges or (v, u) in remaining_edges
            
        # Function to remove an edge
        def remove_edge(u, v):
            if (u, v) in remaining_edges:
                remaining_edges.remove((u, v))
            else:
                remaining_edges.remove((v, u))
                
        pieces = []
        
        # Try to find cycles
        # We will use a heuristic DFS to find cycles.
        # To encourage longer cycles, we will try multiple random starts and neighbor orderings.
        
        while remaining_edges:
            if time.time() > deadline - 5.0:
                break
                
            # Find a vertex with max degree in remaining graph
            # Calculate degrees
            deg = defaultdict(int)
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
                
            # Pick start vertex with highest degree
            if not deg:
                break
            start = max(deg.keys(), key=lambda x: deg[x])
            
            found_cycle = None
            
            # Try a few random permutations of neighbors to find a long cycle
            # We limit the number of attempts per cycle extraction to avoid wasting time
            attempts = 0
            max_attempts = 50
            
            while attempts < max_attempts and not found_cycle:
                attempts += 1
                
                # DFS state
                visited = {start}
                path = [start]
                
                def dfs(u):
                    nonlocal found_cycle
                    if found_cycle:
                        return True
                    
                    # Get neighbors
                    neighbors = [v for v in adj[u] if is_edge(u, v)]
                    
                    # Sort neighbors by degree (descending) to prefer dense areas
                    # Add some randomness to break ties
                    neighbors.sort(key=lambda v: (deg[v], random.random()), reverse=True)
                    
                    for v in neighbors:
                        if v == start and len(path) >= 3:
                            found_cycle = path + [start]
                            return True
                        if v not in visited:
                            visited.add(v)
                            path.append(v)
                            if dfs(v):
                                return True
                            path.pop()
                            visited.remove(v)
                    return False
                
                dfs(start)
                
                if found_cycle:
                    break
                # If not found, the DFS exhausted all simple paths from start.
                # Since the graph is small, this is fast.
                # We can break early if we know no cycle exists? 
                # If DFS from a max-degree node finds no cycle, it might be in a tree component.
                # But there might be a cycle elsewhere.
                # So we should pick a different start if we fail?
                # Actually, if we pick the max degree node and find no cycle, it's likely a tree.
                # Let's just break the inner loop and pick a new start from remaining edges.
                # To do this, we can remove the component of 'start' from consideration?
                # Simpler: just pick a random edge from remaining and try to form a cycle through it?
                # Or just pick the next max degree node.
                
                # To avoid infinite loop if no cycle exists, we should detect if we've tried all possible starts.
                # But if no cycle exists, all edges are in trees.
                # We can just break the outer loop if we fail to find a cycle from several starts.
                pass
            
            if found_cycle:
                # Remove edges of the cycle
                for i in range(len(found_cycle) - 1):
                    u, v = found_cycle[i], found_cycle[i+1]
                    remove_edge(u, v)
                pieces.append(found_cycle[:-1]) # Store without closing vertex
            else:
                # No cycle found starting from this 'start' in these attempts.
                # It's possible no cycle exists in the remaining graph.
                # To be safe, we can check if any cycle exists at all using a standard cycle detection?
                # Or just assume if max-degree start fails, maybe we should try another start.
                # Let's try picking a random edge and trying to close a cycle through it.
                # If that fails too, we assume no cycles left.
                
                # Try one more specific heuristic: Pick a random edge (u, v) and try to find a path from v to u.
                if remaining_edges:
                    sample_edge = random.choice(list(remaining_edges))
                    u, v = sample_edge
                    # Try to find path from v to u avoiding the direct edge
                    # This is effectively the same as the DFS above but constrained.
                    # Given the small size, let's just try a few more random starts.
                    found_cycle = None
                    for _ in range(10):
                        if not remaining_edges: break
                        s = random.choice(list(remaining_edges))[0]
                        visited = {s}
                        path = [s]
                        def dfs2(cur):
                            nonlocal found_cycle
                            if found_cycle: return True
                            neighbors = [x for x in adj[cur] if is_edge(cur, x)]
                            neighbors.sort(key=lambda x: (deg[x], random.random()), reverse=True)
                            for n in neighbors:
                                if n == s and len(path) >= 3:
                                    found_cycle = path + [s]
                                    return True
                                if n not in visited:
                                    visited.add(n)
                                    path.append(n)
                                    if dfs2(n): return True
                                    path.pop()
                                    visited.remove(n)
                            return False
                        dfs2(s)
                        if found_cycle: break
                        
                    if found_cycle:
                        for i in range(len(found_cycle) - 1):
                            u2, v2 = found_cycle[i], found_cycle[i+1]
                            remove_edge(u2, v2)
                        pieces.append(found_cycle[:-1])
                    else:
                        # No cycle found. All remaining edges are single pieces.
                        break

        # Add remaining edges as single pieces
        for e in remaining_edges:
            pieces.append(list(e))
            
        results.append(pieces)
        
    return results

# Need to import Random
from random import Random
