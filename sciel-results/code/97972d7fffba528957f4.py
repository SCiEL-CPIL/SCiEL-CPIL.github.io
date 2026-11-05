import time
import random

def construct():
    deadline = time.time() + 140.0
    n = 18
    
    def creates_k4(adj, u, v):
        common = adj[u] & adj[v]
        m = common
        while m:
            lb = m & (-m)
            x = lb.bit_length() - 1
            m ^= lb
            if adj[x] & common:
                return True
        return False

    def eval_max_tf(adj):
        # Precompute for each vertex v: the set of vertices that form a triangle with v
        # tri_with[v] = bitmask of vertices w such that {v, w, x} is a triangle for some x
        # Actually, we need: can we add v to current_mask without creating a triangle?
        # A triangle is formed if v has two neighbors u1, u2 in mask where u1-u2 is an edge.
        # So: for v, check if there exist u1, u2 in (mask & adj[v]) such that adj[u1] & (1<<u2)
        
        # Precompute: for each pair (u,v) that is an edge, store it
        # Better: for each v, precompute the "conflict" - which vertices in adj[v] are connected to each other
        
        # Fast check: can_add(mask, v)
        # Let S = mask & adj[v]
        # We need: no edge within S
        # i.e., for all u in S, adj[u] & S == 0 (excluding u itself)
        
        def can_add(mask, v):
            S = mask & adj[v]
            m = S
            while m:
                lb = m & (-m)
                u = lb.bit_length() - 1
                m ^= lb
                if adj[u] & S & ~(1 << u):
                    return False
            return True
        
        best = 0
        verts = list(range(n))
        verts.sort(key=lambda v: adj[v].bit_count(), reverse=True)
        
        def bb(pos, current_mask, current_size):
            nonlocal best
            if pos >= n:
                if current_size > best:
                    best = current_size
                return
            if current_size + (n - pos) <= best:
                return
            v = verts[pos]
            if can_add(current_mask, v):
                bb(pos + 1, current_mask | (1 << v), current_size + 1)
            bb(pos + 1, current_mask, current_size)
        
        bb(0, 0, 0)
        return best

    best_score = n
    best_edges = []
    
    # Initial candidate: Turan Graph
    adj_turan = [0] * n
    edges_turan = []
    for i in range(n):
        for j in range(i + 1, n):
            if i // 6 != j // 6:
                adj_turan[i] |= (1 << j)
                adj_turan[j] |= (1 << i)
                edges_turan.append([i, j])
    
    score_turan = eval_max_tf(adj_turan)
    if score_turan < best_score:
        best_score = score_turan
        best_edges = edges_turan.copy()
    
    # Random Search
    for _ in range(300):
        if time.time() > deadline - 10:
            break
            
        adj = [0] * n
        edges = []
        
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if not creates_k4(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
        
        score = eval_max_tf(adj)
        if score < best_score:
            best_score = score
            best_edges = edges.copy()
            
        # Local Search: Remove edges from G (Add to H) to reduce alpha_2(G)
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            random.shuffle(edges)
            for ei in range(len(edges)):
                if time.time() > deadline - 15:
                    break
                u, v = edges[ei]
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edges.pop(ei)
                
                new_score = eval_max_tf(adj)
                if new_score < score:
                    score = new_score
                    if score < best_score:
                        best_score = score
                        best_edges = edges.copy()
                    improved = True
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.insert(ei, [u, v])
                    
    return best_edges
