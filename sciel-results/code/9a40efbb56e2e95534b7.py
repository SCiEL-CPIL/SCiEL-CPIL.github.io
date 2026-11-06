import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 40
    
    # Base: 8 disjoint K5s
    base_edges = set()
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                base_edges.add((u, v) if u < v else (v, u))
                
    edges = set(base_edges)
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # All possible edges
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edges:
                all_edges.append((i, j))
                
    best_edge_count = len(edges)
    best_edges = set(edges)
    
    def can_add(u, v):
        # Check if adding (u, v) creates a C6.
        # This happens if there is a path of length 5 between u and v in G.
        # Path: u - x - a - b - y - v
        # x in N(u), y in N(v)
        # a in N(x), b in N(y)
        # (a, b) is an edge
        # All vertices distinct.
        
        if (u, v) in edges or (v, u) in edges:
            return False
            
        for x in adj[u]:
            if x == v: continue
            for y in adj[v]:
                if y == u: continue
                if x == y: continue # Path length would be 2 or 4, not 5 via this structure
                # Check for path x-a-b-y of length 3
                # a in N(x), b in N(y), (a,b) in E
                # a, b not in {u, v, x, y}
                for a in adj[x]:
                    if a == u or a == v or a == x or a == y:
                        continue
                    for b in adj[y]:
                        if b == u or b == v or b == x or b == y:
                            continue
                        if a == b:
                            continue
                        # Check if (a, b) is an edge
                        if (a, b) in edges or (b, a) in edges:
                            return False # Creates C6
        return True

    # Greedy with restarts
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        current_edges = set(base_edges)
        current_adj = [set() for _ in range(n)]
        for u, v in current_edges:
            current_adj[u].add(v)
            current_adj[v].add(u)
            
        candidates = list(all_edges)
        random.shuffle(candidates)
        
        for u, v in candidates:
            if (u, v) in current_edges:
                continue
            
            # Check validity
            # Re-implement can_add using current_adj and current_edges for speed/clarity?
            # The closure `can_add` uses `edges` and `adj` which are the base.
            # We need a version that works on current state.
            
            is_valid = True
            # Check C6 creation
            for x in current_adj[u]:
                if x == v: continue
                for y in current_adj[v]:
                    if y == u: continue
                    if x == y: continue
                    for a in current_adj[x]:
                        if a == u or a == v or a == x or a == y:
                            continue
                        for b in current_adj[y]:
                            if b == u or b == v or b == x or b == y:
                                continue
                            if a == b:
                                continue
                            if (a, b) in current_edges or (b, a) in current_edges:
                                is_valid = False
                                break
                    if not is_valid: break
                if not is_valid: break
            if not is_valid: break
            
            if is_valid:
                current_edges.add((u, v))
                current_adj[u].add(v)
                current_adj[v].add(u)
                
        if len(current_edges) > best_edge_count:
            best_edge_count = len(current_edges)
            best_edges = set(current_edges)
            
        # Optional: If we reach 105, we can stop early?
        # Or just keep trying to beat it.
        
    # Convert back to list of lists
    result = [[u, v] for u, v in best_edges]
    return result
