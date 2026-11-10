import time
import random
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 16
    
    # 1. Construct initial 45-edge graph
    edges = set()
    for i in range(5):
        edges.add((i, (i+1)%5))
    for i in range(5):
        edges.add((i, 5 + (i-1)%5))
        edges.add((i, 5 + (i+1)%5))
    for i in range(5):
        edges.add((5+i, 10))
        
    S = [5, 6, 7, 8, 9]
    for v in range(11, 16):
        for u in S:
            edges.add((v, u))
            
    best_edges = set(edges)
    best_count = len(best_edges)
    
    # Build initial adjacency
    adj = [set() for _ in range(n)]
    for u, v in best_edges:
        adj[u].add(v)
        adj[v].add(u)

    def is_not_3_colorable(edge_set, adj_list):
        # Backtracking
        # Use adj_list for speed
        order = sorted(range(n), key=lambda x: len(adj_list[x]), reverse=True)
        color = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Check colors of neighbors
            used_colors = set()
            for u in adj_list[v]:
                if color[u] != -1:
                    used_colors.add(color[u])
            
            for c in range(3):
                if c not in used_colors:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
            
        return not backtrack(0)

    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Phase 1: Greedy Add
    # Try to add as many edges as possible without swaps
    changed = True
    while changed and time.time() < deadline - 20.0:
        changed = False
        # Shuffle pairs to avoid bias
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if (u, v) in best_edges or (v, u) in best_edges:
                continue
            # Check triangle
            # Common neighbor?
            if adj[u].intersection(adj[v]):
                continue
            # No triangle, add it
            best_edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
            best_count += 1
            changed = True
            
    # Phase 2: Local Search with Swaps
    # Time remaining for swaps
    while time.time() < deadline:
        u, v = random.choice(all_pairs)
        if (u, v) in best_edges or (v, u) in best_edges:
            continue
            
        # Check if adding (u,v) creates a triangle
        common = adj[u].intersection(adj[v])
        
        if not common:
            # Can add directly
            best_edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
            best_count += 1
        else:
            # Triangle exists. Pick a random common neighbor w
            w = random.choice(list(common))
            
            # Try swap 1: Add (u,v), Remove (u,w)
            # Ensure (u,w) is an edge
            if w in adj[u]:
                new_edges = best_edges.copy()
                new_edges.add((u, v))
                new_edges.discard((u, w))
                new_edges.discard((w, u))
                
                # Check triangle-free for new_edges
                # We can do a full check or incremental
                # For simplicity and safety, full check is fine for n=16
                is_tf = True
                # Build temp adj
                temp_adj = [set() for _ in range(n)]
                for a, b in new_edges:
                    temp_adj[a].add(b)
                    temp_adj[b].add(a)
                    
                for a in range(n):
                    for b in temp_adj[a]:
                        if b > a:
                            if temp_adj[a].intersection(temp_adj[b]):
                                is_tf = False
                                break
                    if not is_tf: break
                
                if is_tf:
                    # Check 3-colorable
                    if is_not_3_colorable(new_edges, temp_adj):
                        best_edges = new_edges
                        adj = temp_adj
                        # Count unchanged
                        continue
                        
            # Try swap 2: Add (u,v), Remove (v,w)
            if w in adj[v]:
                new_edges = best_edges.copy()
                new_edges.add((u, v))
                new_edges.discard((v, w))
                new_edges.discard((w, v))
                
                is_tf = True
                temp_adj = [set() for _ in range(n)]
                for a, b in new_edges:
                    temp_adj[a].add(b)
                    temp_adj[b].add(a)
                    
                for a in range(n):
                    for b in temp_adj[a]:
                        if b > a:
                            if temp_adj[a].intersection(temp_adj[b]):
                                is_tf = False
                                break
                    if not is_tf: break
                
                if is_tf:
                    if is_not_3_colorable(new_edges, temp_adj):
                        best_edges = new_edges
                        adj = temp_adj
                        continue

    return [list(e) for e in best_edges]
