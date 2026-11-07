import time
import sys

def construct():
    n = 15
    deadline = time.time() + 75  # Safety margin before 90s kill
    
    # Helper to check validity efficiently
    def is_valid(edge_set):
        if not edge_set:
            return True
        # Build adjacency and degrees
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        # Check all simple paths of length 3: s - u - v - t
        # For each edge (u, v), s in N(u) \ {v}, t in N(v) \ {u, s}
        for u, v in edge_set:
            for s in adj[u]:
                if s == v:
                    continue
                ds = deg[s]
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if ds == deg[t]:
                        return False
        return True
    
    # Start with K_{7,8}: A = {0..6}, B = {7..14}
    edges = set()
    for u in range(7):
        for v in range(7, 15):
            edges.add((u, v))
    
    # All possible edges
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    missing = list(all_edges - edges)
    
    # Prioritize intra-partition edges
    def priority(e):
        u, v = e
        if u < 7 and v < 7:
            return 0
        elif u >= 7 and v >= 7:
            return 1
        else:
            return 2
    
    missing.sort(key=priority)
    
    current = set(edges)
    best = set(current)
    
    # Greedy addition
    for e in missing:
        if time.time() > deadline:
            break
        candidate = current | {e}
        if is_valid(candidate):
            current = candidate
    
    best = set(current)
    
    # Try to add more edges with different orderings
    # Randomize and try again
    import random
    random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        missing = list(all_edges - current)
        random.shuffle(missing)
        
        trial = set(current)
        for e in missing:
            if time.time() > deadline:
                break
            candidate = trial | {e}
            if is_valid(candidate):
                trial = candidate
        
        if len(trial) > len(best):
            best = set(trial)
            current = set(trial)
    
    # Final polish: try adding any remaining edges
    remaining = list(all_edges - best)
    random.shuffle(remaining)
    for e in remaining:
        if time.time() > deadline:
            break
        candidate = best | {e}
        if is_valid(candidate):
            best = candidate
    
    # Convert to required format
    final_edges = [[int(u), int(v)] for u, v in best]
    return final_edges
