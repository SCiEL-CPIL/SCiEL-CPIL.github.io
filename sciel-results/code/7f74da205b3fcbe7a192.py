import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 40
    best_edges = []
    best_count = 0
    
    adj = np.zeros((n, n), dtype=bool)
    common = [[set() for _ in range(n)] for _ in range(n)]
    
    def update_common(u, v, added):
        for w in range(n):
            if w == u or w == v:
                continue
            if added:
                if adj[v, w]:
                    common[u, w].add(v)
                    common[w, u].add(v)
                if adj[u, w]:
                    common[v, w].add(u)
                    common[w, v].add(u)
            else:
                if adj[v, w]:
                    if v in common[u, w]: common[u, w].remove(v)
                    if v in common[w, u]: common[w, u].remove(v)
                if adj[u, w]:
                    if u in common[v, w]: common[v, w].remove(u)
                    if u in common[w, v]: common[w, v].remove(u)

    def creates_c6(u, v):
        nu = np.where(adj[u, :])[0]
        nv = np.where(adj[v, :])[0]
        
        for a in nu:
            if a == v: continue
            for c in nv:
                if a == c: continue
                if c == u: continue
                # Check if any b in common[a, c] is not u or v
                # common[a, c] is a set
                for b in common[a, c]:
                    if b != u and b != v:
                        return True
        return False

    def add_edge(u, v):
        adj[u, v] = True
        adj[v, u] = True
        update_common(u, v, True)

    def remove_edge(u, v):
        adj[u, v] = False
        adj[v, u] = False
        update_common(u, v, False)

    def get_edges():
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    edges.append([i, j])
        return edges

    while time.time() < deadline:
        # Reset
        for i in range(n):
            for j in range(n):
                adj[i, j] = False
                common[i, j].clear()
        
        # Greedy
        all_edges = [[i, j] for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if not creates_c6(u, v):
                add_edge(u, v)
                
        current_edges = get_edges()
        current_count = len(current_edges)
        
        if current_count > best_count:
            best_count = current_count
            best_edges = current_edges
            
        # Local Search: Swap
        # Try to remove 1 edge and add 2
        # We do this a few times per restart
        for _ in range(10):
            if time.time() > deadline:
                break
            if not current_edges:
                break
                
            # Pick random edge to remove
            idx = random.randint(0, len(current_edges)-1)
            u, v = current_edges[idx]
            
            # Tentative remove
            remove_edge(u, v)
            current_edges.pop(idx)
            
            # Try to add 2 edges
            added = []
            candidates = []
            for i in range(n):
                for j in range(i+1, n):
                    if not adj[i, j]:
                        candidates.append((i, j))
            random.shuffle(candidates)
            
            for i, j in candidates:
                if len(added) >= 2:
                    break
                if not creates_c6(i, j):
                    add_edge(i, j)
                    added.append((i, j))
                    
            if len(added) > 0:
                current_edges.extend(added)
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = get_edges()
            else:
                # Revert
                add_edge(u, v)
                current_edges.append([u, v])

    return best_edges
