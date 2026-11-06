import time
import random

def construct():
    deadline = time.time() + 140
    n = 20
    best_edges = None
    best_count = 0
    
    def greedy_and_search():
        # Use adjacency matrix for speed
        adj = [[False] * n for _ in range(n)]
        # common[i][j] = number of common neighbors of i and j
        common = [[0] * n for _ in range(n)]
        edge_count = 0
        
        def add_edge(u, v):
            nonlocal edge_count
            adj[u][v] = True
            adj[v][u] = True
            edge_count += 1
            # Update common neighbor counts
            for w in range(n):
                if w == u or w == v:
                    continue
                if adj[u][w]:
                    common[v][w] += 1
                    common[w][v] += 1
                if adj[v][w]:
                    common[u][w] += 1
                    common[w][u] += 1
            # common[u][v] and common[v][u]: u and v are now neighbors
            # Their common neighbors are vertices w adjacent to both
            # Before adding, common[u][v] counted w in N(u)∩N(v)
            # After: same set (u not in N(u), v not in N(v) in simple graph context for common)
            # Actually common[u][v] should be |N(u) ∩ N(v)|. 
            # N(u) now includes v, N(v) now includes u.
            # But common neighbors exclude u and v themselves typically.
            # Let me redefine: common[i][j] = |{w : w != i, w != j, adj[i][w] and adj[j][w]}|
            # When we add edge (u,v):
            # For pair (u,v): new common = old common (since v was not in N(u) before, u not in N(v) before,
            #   and now v in N(u), u in N(v), but we exclude u,v from common neighbors)
            #   So common[u][v] unchanged.
            # For pair (u,w), w != v: if adj[v][w], then v becomes a common neighbor of u and w
            #   So common[u][w] += 1 if adj[v][w]
            # For pair (v,w), w != u: if adj[u][w], then u becomes a common neighbor of v and w
            #   So common[v][w] += 1 if adj[u][w]
            # I already did this above. Good.
        
        def remove_edge(u, v):
            nonlocal edge_count
            adj[u][v] = False
            adj[v][u] = False
            edge_count -= 1
            for w in range(n):
                if w == u or w == v:
                    continue
                if adj[u][w]:
                    common[v][w] -= 1
                    common[w][v] -= 1
                if adj[v][w]:
                    common[u][w] -= 1
                    common[w][u] -= 1
        
        def can_add(u, v):
            if u == v or adj[u][v]:
                return False
            # After adding (u,v), check if any pair has >= 3 common neighbors
            # Only pairs involving u or v can change
            # Pair (u,v): common unchanged (as argued)
            # Pair (u,w) for w in N(v), w != u: common[u][w] increases by 1
            #   Need common[u][w] + 1 < 3, i.e., common[u][w] <= 1
            # Pair (v,w) for w in N(u), w != v: common[v][w] increases by 1
            #   Need common[v][w] <= 1
            for w in range(n):
                if w == u or w == v:
                    continue
                if adj[v][w]:
                    if common[u][w] + 1 >= 3:
                        return False
                if adj[u][w]:
                    if common[v][w] + 1 >= 3:
                        return False
            return True
        
        # Greedy build
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(pairs)
        for u, v in pairs:
            if can_add(u, v):
                add_edge(u, v)
        
        # Local search: try to add more edges
        improved = True
        iters = 0
        while improved and iters < 3000 and time.time() < deadline - 3:
            improved = False
            iters += 1
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if not adj[i][j]]
            random.shuffle(pairs)
            for u, v in pairs:
                if can_add(u, v):
                    add_edge(u, v)
                    improved = True
        
        # Try swaps: remove one edge, add up to two
        if time.time() < deadline - 3:
            for _ in range(200):
                if time.time() >= deadline - 3:
                    break
                # Pick a random edge to remove
                existing = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i][j]]
                if not existing:
                    break
                a, b = random.choice(existing)
                remove_edge(a, b)
                # Try to add edges
                added = 0
                for u in range(n):
                    if u == a or u == b:
                        continue
                    if can_add(a, u):
                        add_edge(a, u)
                        added += 1
                        if added >= 2:
                            break
                if added < 2:
                    for u in range(n):
                        if u == a or u == b:
                            continue
                        if can_add(b, u):
                            add_edge(b, u)
                            added += 1
                            if added >= 2:
                                break
                if added < 1:
                    # Put back
                    add_edge(a, b)
        
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i][j]]
        return edges, edge_count
    
    # Multiple restarts
    while time.time() < deadline - 5:
        edges, cnt = greedy_and_search()
        if cnt > best_count:
            best_count = cnt
            best_edges = edges
    
    if best_edges is None:
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges
    
    return best_edges
