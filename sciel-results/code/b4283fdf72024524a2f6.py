import time

def construct():
    deadline = time.time() + 115.0
    
    # Build polarity graph of PG(2,5): 31 vertices
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    points.append((1, (y * pow(x, q-2)) % q, (z * pow(x, q-2)) % q))
                elif y != 0:
                    points.append((0, 1, (z * pow(y, q-2)) % q))
                else:
                    points.append((0, 0, 1))
    
    n_base = len(points)  # 31
    adj = [0] * n_base
    for i in range(n_base):
        for j in range(i + 1, n_base):
            u, v = points[i], points[j]
            if (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    n = n_base
    adj_ext = adj[:]
    
    def can_add_vertex(S, adj, n):
        """Check if adding a new vertex adjacent to set S maintains C4-freeness."""
        s_list = [i for i in range(n) if (S >> i) & 1]
        # No two vertices in S may share a common neighbor
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                if adj[s_list[i]] & adj[s_list[j]]:
                    return False
        # Each vertex outside S is adjacent to at most one vertex in S
        for c in range(n):
            if (S >> c) & 1:
                continue
            cnt = 0
            for v in s_list:
                if (adj[c] >> v) & 1:
                    cnt += 1
                    if cnt > 1:
                        return False
        return True
    
    def has_indep_set_11(adj, n):
        """Check if graph has an independent set of size 11."""
        if n < 11:
            return False
        start = time.time()
        degs = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: degs[x])
        adj_ord = [0] * n
        for i in range(n):
            for j in range(n):
                if (adj[order[i]] >> order[j]) & 1:
                    adj_ord[i] |= (1 << j)
        
        def search(rem, sz):
            if time.time() - start > 10:
                return -1
            if sz >= 11:
                return True
            if not rem or sz + bin(rem).count('1') < 11:
                return False
            bv, bd = -1, n + 1
            t = rem
            while t:
                lsb = t & (-t)
                v = lsb.bit_length() - 1
                d = bin(adj_ord[v] & rem).count('1')
                if d < bd:
                    bd, bv = d, v
                t &= t - 1
            if bv < 0:
                return False
            r = search(rem & ~(adj_ord[bv] | (1 << bv)), sz + 1)
            if r != -1:
                return r
            return search(rem & ~(1 << bv), sz)
        
        return search((1 << n) - 1, 0)
    
    # Greedily extend the graph
    for _ in range(12):
        if time.time() > deadline - 30:
            break
        S = 0
        cands = sorted(range(n), key=lambda x: -bin(adj_ext[x]).count('1'))
        for c in cands:
            if time.time() > deadline - 25:
                break
            if can_add_vertex(S | (1 << c), adj_ext, n):
                S |= (1 << c)
        if not S:
            break
        s_sz = bin(S).count('1')
        nn = n + 1
        na = adj_ext[:] + [0]
        for i in range(n):
            if (S >> i) & 1:
                na[i] |= (1 << (nn - 1))
                na[nn - 1] |= (1 << i)
        if n - s_sz >= 10:
            if has_indep_set_11(na, nn):
                break
        adj_ext = na
        n = nn
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj_ext[u] >> v) & 1:
                edges.append([u, v])
    return {'n': n, 'edges': edges}
