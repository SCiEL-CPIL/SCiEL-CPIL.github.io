import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_cycle_heuristic(adj, edges_set):
        # adj: dict of list of neighbors
        # edges_set: set of (u,v) with u<v
        # Returns a list of vertices representing a cycle, or None
        
        # Try to find a cycle starting from a random edge
        # We want a long cycle.
        # Heuristic: Randomized DFS
        
        if not edges_set:
            return None
            
        # Pick a random edge to close the cycle
        edge_list = list(edges_set)
        random.shuffle(edge_list)
        
        best_cycle = None
        
        # Try a few random edges
        for _ in range(min(10, len(edge_list))):
            if time.time() > deadline:
                break
            u, v = edge_list.pop()
            # Ensure u < v
            if u > v: u, v = v, u
            
            # We want a path from v to u in G \ {(u,v)}
            # Use DFS with backtracking to find a long path
            # State: current_node, path
            # To avoid exponential blowup, we limit the search or use a greedy approach
            
            # Greedy approach:
            # Start at v.
            # Try to go to a neighbor that leads to u.
            # This is essentially finding a path.
            
            # Let's use a simple DFS that tries to find *any* path, 
            # but prioritizes unvisited nodes to keep the path long.
            
            visited = set()
            path = [v]
            visited.add(v)
            
            # We cannot use the edge (u,v)
            # We want to reach u.
            
            # This simple DFS might get stuck in a local minimum.
            # Let's try a few random restarts of the path construction.
            
            found = False
            for _ in range(20): # 20 random attempts for this edge
                if time.time() > deadline:
                    break
                visited = set()
                path = [v]
                visited.add(v)
                current = v
                
                # Greedy walk
                while True:
                    neighbors = adj[current]
                    # Filter out visited and the closing edge
                    candidates = []
                    for nb in neighbors:
                        if nb in visited:
                            continue
                        if nb == u:
                            # Check if edge (current, u) exists and is not (v,u) if current==v?
                            # If current == v, we just started.
                            # The edge is (min(current, u), max(current, u))
                            e = tuple(sorted((current, u)))
                            if e == (u, v):
                                # This is the edge we are closing with.
                                # If current == v, we can't use it to go to u immediately 
                                # unless we are closing the cycle of length 1 (impossible) 
                                # or we are at the end.
                                # If current != v, we can use it to close.
                                if current != v:
                                    candidates.append((nb, 1)) # Priority: close cycle
                                else:
                                    pass # Can't use the start edge to go back immediately
                            else:
                                candidates.append((nb, 0))
                        
                    if not candidates:
                        break # Stuck
                        
                    # If we can close the cycle, do it
                    close_candidates = [c for c in candidates if c[1] == 1]
                    if close_candidates:
                        # Found a cycle
                        cycle = path + [u]
                        if not best_cycle or len(cycle) > len(best_cycle):
                            best_cycle = cycle
                        found = True
                        break
                        
                    # Otherwise, pick a random candidate
                    # Prefer unvisited
                    random.shuffle(candidates)
                    next_node = candidates[0][0]
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node
                    
                    if len(path) > 50: # Safety
                        break

                if found:
                    break

        return best_cycle

    def decompose(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency
        adj = defaultdict(list)
        edge_set = set()
        for u, v in edges:
            if u > v: u, v = v, u
            edge_set.add((u, v))
            adj[u].append(v)
            adj[v].append(u)
            
        pieces = []
        
        while edge_set:
            if time.time() > deadline:
                break
            
            cycle = find_cycle_heuristic(adj, edge_set)
            
            if cycle:
                # Remove edges of the cycle
                for i in range(len(cycle) - 1):
                    u, v = cycle[i], cycle[i+1]
                    e = tuple(sorted((u, v)))
                    if e in edge_set:
                        edge_set.remove(e)
                        adj[u].remove(v)
                        adj[v].remove(u)
                # Add cycle to pieces
                # The cycle is given by vertices. The closing edge is implied.
                # We need to output the list of vertices.
                pieces.append(cycle)
            else:
                # No cycle found, take one edge
                e = next(iter(edge_set))
                edge_set.remove(e)
                u, v = e
                adj[u].remove(v)
                adj[v].remove(u)
                pieces.append([u, v])
                
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline:
            # If we run out of time, return whatever we have or empty?
            # Better to return a valid decomposition if possible.
            # But if we haven't started, we can't.
            # The loop above breaks if time runs out.
            # If we break out of the while loop, we still have edge_set remaining.
            # We must add them as single edges to be valid.
            pass
        
        res = decompose(*spec)
        results.append(res)
        
    return results
