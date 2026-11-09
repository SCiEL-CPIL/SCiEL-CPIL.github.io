import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # Try to build a graph with n vertices
    # We know n=30 works (10 triangles)
    # We want to try n=39, 38, ...
    
    for target_n in range(39, 29, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this target_n
        for restart in range(50):
            if time.time() > deadline:
                break
                
            # Start with 10 triangles if target_n >= 30
            # Otherwise start empty
            base_n = min(target_n, 30)
            edges = []
            adj = [0] * target_n  # Bitmasks
            
            # Add 10 triangles
            for t in range(base_n // 3):
                a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
                edges.append([a, b])
                edges.append([a, c])
                edges.append([b, c])
                adj[a] |= (1 << b) | (1 << c)
                adj[b] |= (1 << a) | (1 << c)
                adj[c] |= (1 << a) | (1 << b)
            
            # Add remaining vertices one by one
            for v in range(base_n, target_n):
                # Find a valid set of neighbors
                # Heuristic: Connect to random vertices that don't violate C4
                # C4 violation if u, w in N(v) and N[u] & N[w] != 0
                
                # Try to find a set of neighbors
                # We want to connect to as many as possible to reduce alpha
                # But must be careful
                
                # Shuffle vertices
                candidates = list(range(v))
                random.shuffle(candidates)
                
                chosen = []
                for u in candidates:
                    # Check if u can be added to chosen
                    # For all w in chosen, N[u] & N[w] must be 0
                    ok = True
                    for w in chosen:
                        if adj[u] & adj[w]:
                            ok = False
                            break
                    if ok:
                        chosen.append(u)
                        # Don't connect to too many?
                        # Limit degree to avoid over-constraining?
                        if len(chosen) >= 5: # Heuristic limit
                            break
                
                # Connect v to chosen
                for u in chosen:
                    edges.append([v, u])
                    adj[v] |= (1 << u)
                    adj[u] |= (1 << v)
            
            # Check C4 freeness (sanity check)
            # It should be C4 free by construction
            
            # Check independence number
            alpha = _max_independent_set_size(adj, target_n)
            
            if alpha <= 10:
                return {'n': target_n, 'edges': edges}
    
    # Fallback to known valid graph n=30
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

def _max_independent_set_size(adj, n):
    # Branch and bound for maximum independent set
    # adj is a list of bitmasks
    best = 0
    
    def backtrack(vertex, current_set, size):
        nonlocal best
        if size > best:
            best = size
        
        # Upper bound: size + number of remaining vertices
        if size + (n - vertex) <= best:
            return
        
        if vertex == n:
            return
        
        # Option 1: Don't include vertex
        backtrack(vertex + 1, current_set, size)
        
        # Option 2: Include vertex
        # Check if vertex is compatible with current_set
        # current_set is a bitmask
        if (adj[vertex] & current_set) == 0:
            backtrack(vertex + 1, current_set | (1 << vertex), size + 1)
    
    backtrack(0, 0, 0)
    return best
