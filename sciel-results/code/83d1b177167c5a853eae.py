import math

def construct():
    N = 256
    # Squares up to 2*N
    squares = set()
    i = 1
    while i*i <= 2*N:
        squares.add(i*i)
        i += 1
        
    # Valid nodes: a+b is not square for a=b => 2a is not square
    # 2a = k^2 => k is even => 2a = 4j^2 => a = 2j^2
    forbidden = set()
    j = 1
    while 2*j*j <= N:
        forbidden.add(2*j*j)
        j += 1
        
    valid_nodes = [x for x in range(1, N+1) if x not in forbidden]
    
    # Build adjacency list
    # u ~ v if u+v in squares and u != v (u=v handled by forbidden)
    # Actually, if u+v is square, they are adjacent.
    # We only care about edges between valid nodes.
    adj = {v: set() for v in valid_nodes}
    for i, u in enumerate(valid_nodes):
        for v in valid_nodes[i+1:]:
            if u + v in squares:
                adj[u].add(v)
                adj[v].add(u)
                
    # Greedy Min-Degree Independent Set
    remaining = set(valid_nodes)
    selected = []
    
    while remaining:
        min_deg = float('inf')
        best_v = None
        for v in remaining:
            deg = len(adj[v] & remaining)
            if deg < min_deg:
                min_deg = deg
                best_v = v
        
        selected.append(best_v)
        # Remove best_v and its neighbors from remaining
        to_remove = set(adj[best_v] & remaining)
        to_remove.add(best_v)
        remaining -= to_remove
        
    S = set(selected)
    U = set(valid_nodes) - S
    
    # Local Search: 1-out, k-in
    improved = True
    while improved:
        improved = False
        # Try adding nodes if possible
        added_any = False
        for u in list(U):
            if not (adj[u] & S):
                S.add(u)
                U.remove(u)
                added_any = True
                improved = True
        if added_any:
            continue
            
        # Try 1-out, k-in swaps
        for s in list(S):
            # Temporarily remove s
            S.remove(s)
            U.add(s)
            
            # Try to add as many from U as possible
            added = []
            for u in list(U):
                if not (adj[u] & S):
                    S.add(u)
                    U.remove(u)
                    added.append(u)
                    
            # Check if we improved size
            # We removed 1 (s). We added len(added).
            # Net change = len(added) - 1
            if len(added) > 1:
                improved = True
            else:
                # Revert
                S.discard(s)
                U.add(s)
                for u in added:
                    S.discard(u)
                    U.add(u)
                    
    return list(S)
