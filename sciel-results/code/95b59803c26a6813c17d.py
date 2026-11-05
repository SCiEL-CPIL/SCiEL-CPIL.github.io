"""K3,3-free dense graph on 20 vertices via randomized local search."""
import time
import random

N = 20
DEADLINE = 135.0

def construct():
    start = time.time()
    random.seed(42)

    def edge_count(adj):
        return sum(bin(a).count('1') for a in adj) // 2

    def is_k33_free(adj):
        for a in range(N):
            for b in range(a + 1, N):
                ab = adj[a] & adj[b]
                if bin(ab).count('1') < 3:
                    continue
                for c in range(b + 1, N):
                    abc = ab & adj[c]
                    x = abc
                    cnt = 0
                    while x:
                        x &= x - 1
                        cnt += 1
                        if cnt >= 3:
                            return False
        return True

    def try_add_edge(adj, u, v):
        if u == v or (adj[u] >> v) & 1:
            return False
        # Tentatively add
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if is_k33_free(adj):
            return True
        else:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            return False

    def greedy_build():
        adj = [0] * N
        edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        random.shuffle(edges)
        for u, v in edges:
            if time.time() - start > DEADLINE:
                break
            try_add_edge(adj, u, v)
        return adj

    def local_search(adj, max_iters=20000):
        best_adj = adj[:]
        best_count = edge_count(adj)
        current = adj[:]
        
        for it in range(max_iters):
            if time.time() - start > DEADLINE:
                break
            improved = False
            # Try adding random missing edges
            for _ in range(15):
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u >= v or (current[u] >> v) & 1:
                    continue
                if try_add_edge(current, u, v):
                    improved = True
                    break
            if improved:
                c = edge_count(current)
                if c > best_count:
                    best_count = c
                    best_adj = current[:]
                continue
            # Try swap: remove one edge, add another
            for _ in range(10):
                a = random.randint(0, N - 1)
                b = random.randint(0, N - 1)
                if a >= b or not ((current[a] >> b) & 1):
                    continue
                c = random.randint(0, N - 1)
                d = random.randint(0, N - 1)
                if c >= d or (current[c] >> d) & 1:
                    continue
                # Remove (a,b)
                current[a] &= ~(1 << b)
                current[b] &= ~(1 << a)
                if try_add_edge(current, c, d):
                    pass
                else:
                    current[c] &= ~(1 << d)
                    current[d] &= ~(1 << c)
                    current[a] |= (1 << b)
                    current[b] |= (1 << a)
                improved = True
                break
            c = edge_count(current)
            if c > best_count:
                best_count = c
                best_adj = current[:]
        
        return best_adj, best_count

    best_overall = None
    best_count_overall = 0

    # Multiple diverse starts
    starts = []
    
    # Greedy
    adj_g = greedy_build()
    starts.append(adj_g)
    
    # Random dense
    for trial in range(3):
        adj_r = [0] * N
        density = 0.6 + 0.1 * trial
        for i in range(N):
            for j in range(i + 1, N):
                if random.random() < density:
                    adj_r[i] |= (1 << j)
                    adj_r[j] |= (1 << i)
        # Fix by removing edges from violations
        for _ in range(300):
            if time.time() - start > DEADLINE:
                break
            if is_k33_free(adj_r):
                break
            # Find violation and remove an edge
            found = False
            for a in range(N):
                if found: break
                for b in range(a + 1, N):
                    if found: break
                    ab = adj_r[a] & adj_r[b]
                    if bin(ab).count('1') < 3: continue
                    for c in range(b + 1, N):
                        abc = ab & adj_r[c]
                        x = abc; cnt = 0
                        while x:
                            x &= x - 1; cnt += 1
                            if cnt >= 3:
                                found = True; break
                        if found: break
                    if found: break
                if found: break
            if not found: break
            adj_r[a] &= ~(1 << b)
            adj_r[b] &= ~(1 << a)
        starts.append(adj_r)

    for adj_s in starts:
        if time.time() - start > DEADLINE:
            break
        c_s = edge_count(adj_s)
        if c_s > best_count_overall:
            best_count_overall = c_s
            best_overall = adj_s[:]
        adj_ls, c_ls = local_search(adj_s, max_iters=15000)
        if c_ls > best_count_overall:
            best_count_overall = c_ls
            best_overall = adj_ls[:]

    if best_overall is None:
        return [[a, b] for a in range(2) for b in range(2, 20)]

    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if (best_overall[u] >> v) & 1:
                edges.append([u, v])
    return edges
