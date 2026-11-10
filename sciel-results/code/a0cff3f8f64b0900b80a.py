import time
import random

def construct():
    start = time.time()
    deadline = start + 110.0
    n = 20
    
    # Rook's graph on 4x4 grid (16 vertices)
    # Vertices 0..15
    # (i, j) -> i*4 + j
    # Adjacent if same row or same col
    adj = [0]*n
    for i in range(4):
        for j in range(4):
            u = i*4 + j
            for j2 in range(4):
                if j2 != j:
                    v = i*4 + j2
                    adj[u] |= 1<<v
                    adj[v] |= 1<<u
            for i2 in range(4):
                if i2 != i:
                    v = i2*4 + j
                    adj[u] |= 1<<v
                    adj[v] |= 1<<u
    
    # Add K4 on vertices 16, 17, 18, 19
    for u in range(16, 20):
        for v in range(u+1, 20):
            adj[u] |= 1<<v
            adj[v] |= 1<<u
            
    # Add matching 0-16, 1-17, 2-18, 3-19
    for i in range(4):
        u = i
        v = 16 + i
        adj[u] |= 1<<v
        adj[v] |= 1<<u
        
    # Verify and count
    def popcount(x):
        return bin(x).count('1')
        
    def verify(adj):
        for u in range(n):
            for v in range(u+1, n):
                if popcount(adj[u] & adj[v]) >= 3:
                    return False
        return True
        
    def edge_count(adj):
        return sum(popcount(a) for a in adj)//2

    if not verify(adj):
        # Fallback to random search if construction fails (should not)
        adj = [0]*n
        for _ in range(1000):
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and not ((adj[u]>>v)&1):
                # Check if can add
                ok = True
                for w in range(n):
                    if w != u and w != v:
                        if (adj[u]>>w)&1 and (adj[v]>>w)&1:
                            # Check common neighbors of u,v
                            pass
                # Simplified check
                cn = popcount(adj[u] & adj[v])
                if cn < 3:
                    # Check bad pairs
                    bad = False
                    for w in range(n):
                        if w == u or w == v: continue
                        if (adj[u]>>w)&1:
                            if popcount(adj[w] & adj[v]) >= 2:
                                bad = True
                                break
                        if (adj[v]>>w)&1:
                            if popcount(adj[u] & adj[w]) >= 2:
                                bad = True
                                break
                    if not bad:
                        adj[u] |= 1<<v
                        adj[v] |= 1<<u

    # Local Search
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    def can_add(adj, u, v):
        if (adj[u]>>v)&1: return True
        if popcount(adj[u] & adj[v]) >= 3:
            return False
        for w in range(n):
            if w == u or w == v: continue
            if (adj[u]>>w)&1:
                if popcount(adj[w] & adj[v]) >= 2:
                    return False
            if (adj[v]>>w)&1:
                if popcount(adj[u] & adj[w]) >= 2:
                    return False
        return True

    def do_add(adj, u, v):
        adj[u] |= 1<<v
        adj[v] |= 1<<u

    def do_remove(adj, u, v):
        adj[u] &= ~(1<<v)
        adj[v] &= ~(1<<u)

    best_adj = adj[:]
    best_count = edge_count(adj)
    
    rng = random.Random(42)
    
    # Try to improve
    for _ in range(500):
        if time.time() > deadline - 5.0:
            break
        cur_adj = best_adj[:]
        # Random perturb
        for _ in range(10):
            u, v = rng.choice(all_edges)
            if (cur_adj[u]>>v)&1:
                do_remove(cur_adj, u, v)
            elif can_add(cur_adj, u, v):
                do_add(cur_adj, u, v)
        
        # Local improve
        changed = True
        while changed:
            changed = False
            for u, v in all_edges:
                if not ((cur_adj[u]>>v)&1) and can_add(cur_adj, u, v):
                    do_add(cur_adj, u, v)
                    changed = True
        
        cnt = edge_count(cur_adj)
        if verify(cur_adj) and cnt > best_count:
            best_count = cnt
            best_adj = cur_adj[:]

    res = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u]>>v)&1:
                res.append([u,v])
    return res
