import random
import time

def construct():
    n = 18
    deadline = time.time() + 140
    
    def try_add_edge(adj, u, v):
        if adj[u] & (1 << v):
            return None
        common = adj[u] & adj[v] & ~(1 << u) & ~(1 << v)
        for w in range(n):
            if not (common & (1 << w)):
                continue
            for x in range(w+1, n):
                if not (common & (1 << x)):
                    continue
                if adj[w] & (1 << x):
                    return None
        new_adj = adj[:]
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
        return new_adj
    
    def find_largest_tf(adj):
        best = 0
        for mask in range(1, 1 << n):
            if time.time() > deadline:
                return best
            sz = mask.bit_count()
            if sz <= best:
                continue
            tf = True
            for i in range(n):
                if not (mask & (1 << i)):
                    continue
                neigh = adj[i] & mask
                nn = neigh
                while nn:
                    lsb = nn & (-nn)
                    j = lsb.bit_length() - 1
                    nn ^= lsb
                    if adj[j] & neigh:
                        tf = False
                        break
                if not tf:
                    break
            if tf:
                best = sz
        return best
    
    def get_edges(adj):
        return [[i, j] for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
    
    random.seed(42)
    
    def random_init():
        adj = [0] * n
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        for u, v in pairs:
            new_adj = try_add_edge(adj, u, v)
            if new_adj is not None:
                adj = new_adj
        return adj
    
    best_adj = None
    best_score = n + 1
    
    for trial in range(10):
        if time.time() > deadline - 60:
            break
        
        adj = random_init()
        score = find_largest_tf(adj)
        
        if score < best_score:
            best_score = score
            best_adj = adj[:]
    
    return get_edges(best_adj)
