import random
import time

def is_triangle_free(edges, n=16):
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    for u in range(n):
        for v in range(u+1, n):
            if adj[u][v]:
                for w in range(v+1, n):
                    if adj[u][w] and adj[v][w]:
                        return False
    return True

def is_3_colorable(edges, n=16):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    color = [-1]*n
    
    def dfs(u):
        if color[u] != -1:
            return True
        for c in range(3):
            if all(color[v] != c for v in adj[u]):
                color[u] = c
                if dfs(u+1):
                    return True
                color[u] = -1
        return False
    
    # Try to color vertices 0..n-1
    # Use a simple DFS
    # To speed up, order vertices by degree
    order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    # Remap colors according to order
    # Actually, simpler to just use the original order or a heuristic
    # Let's use the original order for simplicity, but it might be slow
    # Better to use a recursive backtracking with vertex ordering
    
    # Let's implement a proper backtracking with vertex ordering
    def backtrack(idx):
        if idx == n:
            return True
        u = order[idx]
        for c in range(3):
            if all(color[v] != c for v in adj[u]):
                color[u] = c
                if backtrack(idx+1):
                    return True
                color[u] = -1
        return False
    
    return backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140 seconds to be safe
    
    best_edges = None
    best_count = -1
    
    # Start with a known 4-chromatic triangle-free graph: Grötzsch
    # Grötzsch graph edges
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    groetzsch_edges = [list(e) for e in cycle]
    for u, w in cycle:
        groetzsch_edges += [[u, 5 + w], [w, 5 + u]]
    groetzsch_edges += [[5 + u, 10] for u in range(5)]
    
    # Add random edges to the Grötzsch graph
    # We'll do multiple restarts
    for restart in range(1000):
        if time.time() > deadline:
            break
        
        # Start with Grötzsch graph and 5 isolated vertices
        edges = list(groetzsch_edges)
        # Add random edges between vertices 11..15 and others, or among themselves
        # Actually, we can add any edge that doesn't create a triangle
        # Let's do a greedy random addition
        random.shuffle(edges) # Not needed, but for variety
        
        # Try to add all possible edges
        all_possible = [(u, v) for u in range(16) for v in range(u+1, 16)]
        random.shuffle(all_possible)
        
        current_edges = set(tuple(e) for e in groetzsch_edges)
        
        for u, v in all_possible:
            if time.time() > deadline:
                break
            if (u, v) in current_edges or (v, u) in current_edges:
                continue
            # Try adding edge (u, v)
            new_edges = list(current_edges) + [[u, v]]
            # Check triangle-free
            if not is_triangle_free(new_edges):
                continue
            # Check 3-colorable
            if is_3_colorable(new_edges):
                continue
            # Add the edge
            current_edges.add((u, v))
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = [list(e) for e in current_edges]
            
            # If we reached the maximum possible (64), we can stop
            if best_count == 64:
                break
    
    if best_edges is None:
        # Fallback to Grötzsch
        return groetzsch_edges
    return best_edges
