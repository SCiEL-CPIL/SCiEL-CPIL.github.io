import time
import random

N = 40
DEADLINE = 110.0  # safety margin

def construct():
    start = time.time()
    random.seed(42)
    
    # adjacency as list of ints
    adj = [0]*N
    
    def add_edge(u,v):
        adj[u] |= 1<<v
        adj[v] |= 1<<u
    
    def remove_edge(u,v):
        adj[u] &= ~(1<<v)
        adj[v] &= ~(1<<u)
    
    def edge_count():
        return sum(bin(a).count('1') for a in adj)//2
    
    def path5(u, v):
        # check simple path of length 5 from u to v
        # u-a-b-c-d-v, all distinct
        a_mask = adj[u] & ~(1<<v)
        while a_mask:
            ab = a_mask & (-a_mask)
            a = ab.bit_length()-1
            a_mask ^= ab
            b_mask = adj[a] & ~(1<<u) & ~(1<<v)
            while b_mask:
                bb = b_mask & (-b_mask)
                b = bb.bit_length()-1
                b_mask ^= bb
                c_mask = adj[b] & ~(1<<u) & ~(1<<a) & ~(1<<v)
                while c_mask:
                    cb = c_mask & (-c_mask)
                    c = cb.bit_length()-1
                    c_mask ^= cb
                    d_mask = adj[c] & ~(1<<u) & ~(1<<a) & ~(1<<b) & ~(1<<v)
                    # d must be neighbor of v
                    d_mask &= adj[v]
                    if d_mask:
                        return True
        return False
    
    def is_c6_free():
        # verify no C6 (for debugging)
        for u in range(N):
            for v in range(u+1, N):
                if (adj[u]>>v)&1:
                    # check path5 in graph without edge u-v
                    # temporarily remove
                    adj[u] &= ~(1<<v)
                    adj[v] &= ~(1<<u)
                    has = path5(u,v)
                    adj[u] |= 1<<v
                    adj[v] |= 1<<u
                    if has:
                        return False
        return True
    
    # Initialize: 8 K5's
    for k in range(8):
        base = 5*k
        for a in range(5):
            for b in range(a+1,5):
                add_edge(base+a, base+b)
    
    best_edges = edge_count()
    
    # Greedy: try to add missing edges
    improved = True
    while improved:
        improved = False
        for u in range(N):
            for v in range(u+1, N):
                if time.time() > DEADLINE:
                    break
                if not ((adj[u]>>v)&1):
                    if not path5(u,v):
                        add_edge(u,v)
                        improved = True
                        best_edges = edge_count()
        if time.time() > DEADLINE:
            break
    
    # Swap-based improvement: remove one edge, add up to 2
    for it in range(200):
        if time.time() > DEADLINE:
            break
        found_swap = False
        for u in range(N):
            for v in range(u+1, N):
                if time.time() > DEADLINE:
                    break
                if not ((adj[u]>>v)&1):
                    continue
                # try removing (u,v) and adding two new edges
                remove_edge(u,v)
                added = 0
                added_edges = []
                for u2 in range(N):
                    for v2 in range(u2+1, N):
                        if time.time() > DEADLINE:
                            break
                        if not ((adj[u2]>>v2)&1):
                            # check if adding (u2,v2) creates C6
                            if not path5(u2,v2):
                                add_edge(u2,v2)
                                added_edges.append((u2,v2))
                                added += 1
                                if added == 2:
                                    break
                    if added == 2:
                        break
                if added > 1:
                    found_swap = True
                    best_edges = edge_count()
                else:
                    # revert
                    for (a,b) in added_edges:
                        remove_edge(a,b)
                    add_edge(u,v)
                if found_swap:
                    break
            if found_swap:
                break
        if not found_swap:
            break
    
    # Build edge list
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (adj[u]>>v)&1:
                edges.append([u,v])
    return edges
