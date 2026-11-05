import time
import random

def construct():
    deadline = time.time() + 140.0
    n = 18
    
    # Helper to check if adding edge (u, v) to H (removing from G) creates an independent set of size 4 in H
    # This is equivalent to checking if G \ {(u,v)} contains a K4?
    # No. G is K4-free. Removing an edge from G cannot create a K4.
    # So we can always remove edges from G.
    # The constraint is just that we want to minimize alpha_2(G).
    
    def creates_k4(adj, u, v):
        # Check if adding edge (u, v) to G creates a K4
        # This is used during the initial construction of G
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
        # Precompute triangles
        triangles = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    cm = common & (~((1 << (v + 1)) - 1))
                    while cm:
                        lb = cm & (-cm)
                        w = lb.bit_length() - 1
                        cm ^= lb
                        triangles.append((u, v, w))
        
        # tri_partners[v] = bitmask of vertices w such that v,w are in some triangle
        tri_partners = [0] * n
        for (u, v, w) in triangles:
            tri_partners[u] |= (1 << v) | (1 << w)
            tri_partners[v] |= (1 << u) | (1 << w)
            tri_partners[w] |= (1 << u) | (1 << v)
        
        def can_add(mask, v):
            nm = mask & adj[v]
            rm = nm
            while rm:
                lb = rm & (-rm)
                u = lb.bit_length() - 1
                rm ^= lb
                if adj[u] & mask & adj[v] & ~(1 << u):
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
            # Shuffle edges to avoid bias
            random.shuffle(edges)
            for ei in range(len(edges)):
                if time.time() > deadline - 15:
                    break
                u, v = edges[ei]
                # Remove edge
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
                    # Restore
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.insert(ei, [u, v])
                    
    return best_edges
