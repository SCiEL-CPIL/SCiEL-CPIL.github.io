import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n = 20
    best_edges = 0
    best_adj = None
    
    triples = [(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)]
    
    def is_k33_fast(adj):
        for (i, j, k) in triples:
            common = adj[i] & adj[j] & adj[k]
            if bin(common).count('1') >= 3:
                return False
        return True
    
    def edge_count(adj):
        return sum(bin(x).count('1') for x in adj) // 2
    
    def can_add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        result = is_k33_fast(adj)
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return result
    
    def structured_seed():
        adj = [0] * n
        conn_set = {1, 2, 4, 5, 8, 10, 12, 15, 16, 18}
        for i in range(n):
            for d in conn_set:
                j = (i + d) % n
                if j > i:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj
    
    def random_seed(p):
        adj = [0] * n
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj
    
    def purge(adj):
        for _ in range(200):
            violated = False
            for (i, j, k) in triples:
                common = adj[i] & adj[j] & adj[k]
                if bin(common).count('1') >= 3:
                    violated = True
                    c = common
                    bits = []
                    while c:
                        lsb = c & -c
                        bits.append(lsb.bit_length() - 1)
                        c -= lsb
                    x = random.choice(bits)
                    adj[i] &= ~(1 << x)
                    adj[x] &= ~(1 << i)
                    break
            if not violated:
                break
        return adj
    
    def greedy_fill(adj):
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj[u] >> v) & 1:
                        if can_add_edge(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            changed = True
        return adj
    
    def local_search(adj, max_iters=50):
        for _ in range(max_iters):
            if time.time() > deadline - 3.0:
                break
            missing = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] >> v) & 1]
            existing = [(u, v) for u in range(n) for v in range(u+1, n) if (adj[u] >> v) & 1]
            if not missing or not existing:
                break
            random.shuffle(missing)
            added = False
            for u, v in missing[:30]:
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added = True
                    break
            if added:
                continue
            for _ in range(20):
                u1, v1 = random.choice(existing)
                u2, v2 = random.choice(missing)
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                adj[u2] |= (1 << v2)
                adj[v2] |= (1 << u2)
                if is_k33_fast(adj):
                    for uu in range(n):
                        for vv in range(uu+1, n):
                            if not (adj[uu] >> vv) & 1:
                                if can_add_edge(adj, uu, vv):
                                    adj[uu] |= (1 << vv)
                                    adj[vv] |= (1 << uu)
                    break
                else:
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    adj[u2] &= ~(1 << v2)
                    adj[v2] &= ~(1 << u2)
        return adj
    
    # Structured seed first
    adj = structured_seed()
    adj = purge(adj)
    adj = greedy_fill(adj)
    ec = edge_count(adj)
    if ec > best_edges:
        best_edges = ec
        best_adj = adj[:]
    adj = local_search(adj, 40)
    ec = edge_count(adj)
    if ec > best_edges:
        best_edges = ec
        best_adj = adj[:]
    
    # Random restarts
    for restart in range(500):
        if time.time() > deadline - 5.0:
            break
        p = 0.2 + 0.5 * (restart % 6) / 5.0
        adj = random_seed(p)
        adj = purge(adj)
        adj = greedy_fill(adj)
        ec = edge_count(adj)
        if ec > best_edges:
            best_edges = ec
            best_adj = adj[:]
        adj = local_search(adj, 25)
        ec = edge_count(adj)
        if ec > best_edges:
            best_edges = ec
            best_adj = adj[:]
    
    if best_adj:
        for u in range(n):
            for v in range(u+1, n):
                if not (best_adj[u] >> v) & 1:
                    if can_add_edge(best_adj, u, v):
                        best_adj[u] |= (1 << v)
                        best_adj[v] |= (1 << u)
        return [[u, v] for u in range(n) for v in range(u+1, n) if (best_adj[u] >> v) & 1]
    else:
        return [[u, v] for u in range(10) for v in range(10, 20) if (u + v) % 3 == 0]
