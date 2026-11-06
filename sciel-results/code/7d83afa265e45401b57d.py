import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 112  # Safety margin
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {(u, v): i for i, (u, v) in enumerate(all_edges)}
    m_total = len(all_edges)
    
    best_edges = []
    best_count = 0
    best_adj = [0] * n
    
    # Precompute Shrikhande Complement
    # Shrikhande: (i,j) ~ (i',j') if (i==i' and j-j' in {1,3}) or (j==j' and i-i' in {1,3}) or (i-i'==j-j' and i-i' in {1,3})
    # Map vertex v to (v//4, v%4)
    
    shrikhande_edges = set()
    for v in range(n):
        i, j = divmod(v, 4)
        for w in range(v+1, n):
            i2, j2 = divmod(w, 4)
            di = (i2 - i) % 4
            dj = (j2 - j) % 4
            
            is_shrikhande = False
            if i == i2 and dj in (1, 3):
                is_shrikhande = True
            elif j == j2 and di in (1, 3):
                is_shrikhande = True
            elif di == dj and di in (1, 3):
                is_shrikhande = True
            
            if is_shrikhande:
                shrikhande_edges.add((v, w))
                
    # Complement edges
    comp_edges = [e for e in all_edges if e not in shrikhande_edges]
    
    # Check if comp_edges is Q3-free
    def is_q3_free(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        # Check all subsets of 8 vertices? Too slow.
        # Use the specific structure check for each edge?
        # If we just want to verify the seed, we can do a full search for Q3.
        # But for 72 edges, it's likely to have Q3 if not designed for it.
        # Let's assume it might have Q3 and use it as a seed for local search.
        # But if it is Q3-free, we return it immediately.
        
        # Fast check: Iterate all edges, see if any is in a Q3.
        for u, v in edges:
            if has_q3_with_edge(adj, u, v):
                return False
        return True

    # Optimized has_q3_with_edge
    def has_q3_with_edge(adj, u, v):
        # Check if graph with edge (u,v) contains Q3 containing (u,v)
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)
        
        if nu.bit_count() < 2 or nv.bit_count() < 2:
            return False
            
        nu_list = [i for i in range(n) if (nu >> i) & 1]
        nv_list = [i for i in range(n) if (nv >> i) & 1]
        
        # Candidates for a,b from nu_list, c,d from nv_list
        for i in range(len(nu_list)):
            a = nu_list[i]
            for j in range(i+1, len(nu_list)):
                b = nu_list[j]
                
                # a,b must not be connected to v?
                # In Q3, a,b are in same part as v? No.
                # u (000), v (001). a (010), b (100).
                # v is 001. a is 010. dist(v,a) = 2. So a NOT connected to v.
                # So we require (adj[a] & (1<<v)) == 0 and (adj[b] & (1<<v)) == 0.
                if (adj[a] >> v) & 1: continue
                if (adj[b] >> v) & 1: continue
                
                for k in range(len(nv_list)):
                    c = nv_list[k]
                    for l in range(k+1, len(nv_list)):
                        d = nv_list[l]
                        
                        if (adj[c] >> u) & 1: continue
                        if (adj[d] >> u) & 1: continue
                        
                        # Check connections a-c, b-d or a-d, b-c
                        ac = (adj[a] >> c) & 1
                        bd = (adj[b] >> d) & 1
                        ad = (adj[a] >> d) & 1
                        bc = (adj[b] >> c) & 1
                        
                        if not ((ac and bd) or (ad and bc)):
                            continue
                        
                        # Determine e, f
                        # e is common neighbor of a, b (excluding u, c, d)
                        # f is common neighbor of c, d (excluding v, a, b)
                        common_ab = adj[a] & adj[b] & ~(1 << u) & ~(1 << c) & ~(1 << d)
                        common_cd = adj[c] & adj[d] & ~(1 << v) & ~(1 << a) & ~(1 << b)
                        
                        # Check if there is edge between common_ab and common_cd
                        # e in common_ab, f in common_cd, e != f, e ~ f
                        # Since common_ab and common_cd are disjoint (bipartite parts), e != f is automatic if they are in different parts.
                        # But we must ensure e, f are valid vertices.
                        
                        # Iterate e in common_ab
                        e_list = [x for x in range(n) if (common_ab >> x) & 1]
                        f_list = [x for x in range(n) if (common_cd >> x) & 1]
                        
                        for e in e_list:
                            for f in f_list:
                                if (adj[e] >> f) & 1:
                                    return True
                                    
        return False

    # Use Shrikhande Complement as initial seed
    initial_edges = set(comp_edges)
    adj = [0] * n
    for u, v in initial_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Verify and repair if necessary
    # If it has Q3, remove edges
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for u, v in list(initial_edges):
            if has_q3_with_edge(adj, u, v):
                initial_edges.discard((u, v))
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                changed = True
                
    best_count = len(initial_edges)
    best_edges = list(initial_edges)
    
    # Local Search to improve
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u, v in all_edges:
            if (u, v) not in initial_edges:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if not has_q3_with_edge(adj, u, v):
                    initial_edges.add((u, v))
                    improved = True
                    best_count = len(initial_edges)
                    best_edges = list(initial_edges)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
        # Random restarts if stuck?
        if not improved:
             # Try removing random edge
             if initial_edges:
                 u, v = random.choice(list(initial_edges))
                 initial_edges.discard((u, v))
                 adj[u] &= ~(1 << v)
                 adj[v] &= ~(1 << u)
                 # Re-add
                 for u2, v2 in all_edges:
                     if (u2, v2) not in initial_edges:
                         adj[u2] |= (1 << v2)
                         adj[v2] |= (1 << u2)
                         if not has_q3_with_edge(adj, u2, v2):
                             initial_edges.add((u2, v2))
                             improved = True
                             best_count = len(initial_edges)
                             best_edges = list(initial_edges)
                             break
                         else:
                             adj[u2] &= ~(1 << v2)
                             adj[v2] &= ~(1 << u2)

    return [[u, v] for u, v in best_edges]

if __name__ == "__main__":
    print(construct())
