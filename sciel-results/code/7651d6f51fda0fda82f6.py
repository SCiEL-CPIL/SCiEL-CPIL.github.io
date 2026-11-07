import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit
    
    best_edges = None
    best_raw_value = float('inf')
    
    n = 12
    min_degree = 3
    
    # Helper to check if graph is valid and count cycle lengths
    def evaluate(edges):
        # Build adjacency list
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Check degree
        for i in range(n):
            if len(adj[i]) < min_degree:
                return None
                
        # Check triangles
        for u in range(n):
            for v in adj[u]:
                for w in adj[v]:
                    if w == u: continue
                    if w in adj[u]:
                        return None # Triangle found
        
        # Enumerate simple cycles
        # Since n=12, we can use DFS with backtracking
        cycle_lengths = set()
        
        def dfs(start, current, visited, path):
            if current == start and len(path) > 3:
                cycle_lengths.add(len(path))
                return
            for neighbor in adj[current]:
                if neighbor == start and len(path) > 3:
                    cycle_lengths.add(len(path))
                elif neighbor not in visited and neighbor > start:
                    # Only count cycles where start is the smallest vertex
                    # to avoid duplicates
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(start, neighbor, visited, path)
                    path.pop()
                    visited.remove(neighbor)
                    
        for i in range(n):
            visited = {i}
            dfs(i, i, visited, [i])
            
        if not cycle_lengths:
            return None # No cycles? But min degree 3 implies cycles.
            
        return len(cycle_lengths)

    # Generate random 3-regular graphs
    # Use configuration model or random permutation
    for _ in range(10000):
        if time.time() > deadline:
            break
            
        # Generate random 3-regular graph
        # Method: Random permutation of stubs
        stubs = [i for i in range(n) for _ in range(3)]
        random.shuffle(stubs)
        
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                valid = False
                break
            if (min(u,v), max(u,v)) in edges:
                valid = False
                break
            edges.add((min(u,v), max(u,v)))
            
        if not valid:
            continue
            
        # Check if simple (no loops, no multi-edges) - handled above
        # Convert to list
        edge_list = [list(e) for e in edges]
        
        raw = evaluate(edge_list)
        if raw is not None:
            if raw < best_raw_value:
                best_raw_value = raw
                best_edges = edge_list
                if best_raw_value == 1:
                    break # Can't do better
    
    if best_edges is None:
        # Fallback to seed
        es=set()
        for i in range(6):
            for u,v in [(i,(i+1)%6),(i+6,(i+1)%6+6),(i,i+6)]:
                es.add(tuple(sorted((u,v))))
        return [list(e) for e in sorted(es)]
        
    return best_edges
