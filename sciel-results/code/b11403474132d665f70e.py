import random
import time

def is_k4_free(adj):
    n = len(adj)
    for a in range(n):
        for b in range(a+1, n):
            if not (adj[a] & (1 << b)): continue
            for c in range(b+1, n):
                if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)): continue
                if adj[a] & adj[b] & adj[c]:
                    return False
    return True

def get_largest_triangle_free(adj, deadline=None):
    n = len(adj)
    best = 0
    
    # Precompute common neighbors for each edge
    cn = [[0]*n for _ in range(n)]
    for u in range(n):
        for v in range(u+1, n):
            if adj[u] & (1 << v):
                cn[u][v] = adj[u] & adj[v] & ~((1<<u)|(1<<v))
                cn[v][u] = cn[u][v]
    
    # Enumerate subsets from largest to smallest
    for mask in range((1 << n) - 1, -1, -1):
        if deadline and time.time() > deadline:
            break
        k = bin(mask).count('1')
        if k <= best:
            continue
        
        # Check if mask is triangle-free
        is_tf = True
        m = mask
        while m:
            v_bit = m & (-m)
            v = v_bit.bit_length() - 1
            m ^= v_bit
            m2 = mask & ~v_bit
            while m2:
                u_bit = m2 & (-m2)
                u = u_bit.bit_length() - 1
                m2 ^= u_bit
                if u >= v:
                    continue
                if adj[u] & (1 << v):
                    if cn[u][v] & mask:
                        is_tf = False
                        break
            if not is_tf:
                break
        
        if is_tf:
            best = k
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 18
    
    # Initial graph: Turan T(18,3)
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            part_i = 0 if i < 6 else (1 if i < 12 else 2)
            part_j = 0 if j < 6 else (1 if j < 12 else 2)
            if part_i != part_j:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    best_l = get_largest_triangle_free(adj, deadline)
    best_adj = adj[:]
    
    while time.time() < deadline:
        move_type = random.randint(0, 3)
        
        if move_type == 0:
            all_edges = [(i,j) for i in range(n) for j in range(i+1,n) if adj[i] & (1 << j)]
            if not all_edges:
                continue
            u, v = random.choice(all_edges)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            l = get_largest_triangle_free(adj, deadline)
            if l < best_l:
                best_l = l
                best_adj = adj[:]
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        elif move_type == 1:
            all_non_edges = [(i,j) for i in range(n) for j in range(i+1,n) if not (adj[i] & (1 << j))]
            if not all_non_edges:
                continue
            u, v = random.choice(all_non_edges)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if is_k4_free(adj):
                l = get_largest_triangle_free(adj, deadline)
                if l < best_l:
                    best_l = l
                    best_adj = adj[:]
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
        elif move_type == 2:
            all_edges = [(i,j) for i in range(n) for j in range(i+1,n) if adj[i] & (1 << j)]
            all_non_edges = [(i,j) for i in range(n) for j in range(i+1,n) if not (adj[i] & (1 << j))]
            if not all_edges or not all_non_edges:
                continue
            u, v = random.choice(all_edges)
            x, y = random.choice(all_non_edges)
            if (u,v) == (x,y) or (u,v) == (y,x):
                continue
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            adj[x] |= (1 << y)
            adj[y] |= (1 << x)
            if is_k4_free(adj):
                l = get_largest_triangle_free(adj, deadline)
                if l < best_l:
                    best_l = l
                    best_adj = adj[:]
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    adj[x] &= ~(1 << y)
                    adj[y] &= ~(1 << x)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                adj[x] &= ~(1 << y)
                adj[y] &= ~(1 << x)
                
        elif move_type == 3:
            all_edges = [(i,j) for i in range(n) for j in range(i+1,n) if adj[i] & (1 << j)]
            all_non_edges = [(i,j) for i in range(n) for j in range(i+1,n) if not (adj[i] & (1 << j))]
            if len(all_edges) < 2 or len(all_non_edges) < 2:
                continue
            (u1,v1), (u2,v2) = random.sample(all_edges, 2)
            (x1,y1), (x2,y2) = random.sample(all_non_edges, 2)
            if len({(u1,v1),(u2,v2),(x1,y1),(x2,y2)}) < 4:
                continue
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            adj[u2] &= ~(1 << v2)
            adj[v2] &= ~(1 << u2)
            adj[x1] |= (1 << y1)
            adj[y1] |= (1 << x1)
            adj[x2] |= (1 << y2)
            adj[y2] |= (1 << x2)
            if is_k4_free(adj):
                l = get_largest_triangle_free(adj, deadline)
                if l < best_l:
                    best_l = l
                    best_adj = adj[:]
                else:
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    adj[x1] &= ~(1 << y1)
                    adj[y1] &= ~(1 << x1)
                    adj[x2] &= ~(1 << y2)
                    adj[y2] &= ~(1 << x2)
            else:
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                adj[u2] |= (1 << v2)
                adj[v2] |= (1 << u2)
                adj[x1] &= ~(1 << y1)
                adj[y1] &= ~(1 << x1)
                adj[x2] &= ~(1 << y2)
                adj[y2] &= ~(1 << x2)
        
        if random.random() < 0.05:
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.4:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            for _ in range(100):
                if is_k4_free(adj): break
                found = False
                for a in range(n):
                    for b in range(a+1, n):
                        if not (adj[a] & (1 << b)): continue
                        for c in range(b+1, n):
                            if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)): continue
                            common = adj[a] & adj[b] & adj[c]
                            if common:
                                d = (common & -common).bit_length() - 1
                                adj[a] &= ~(1 << d)
                                adj[d] &= ~(1 << a)
                                found = True
                                break
                        if found: break
                    if found: break
                if not found: break
            l = get_largest_triangle_free(adj, deadline)
            if l < best_l:
                best_l = l
                best_adj = adj[:]
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    return edges
