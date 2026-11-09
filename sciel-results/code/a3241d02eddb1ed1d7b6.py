import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    def time_up():
        return time.time() > deadline
    
    def max_independent_set(adj_bits, n, timeout=6):
        start = time.time()
        best = [0]
        non_adj = [((1 << n) - 1) ^ (1 << i) ^ adj_bits[i] for i in range(n)]
        
        def bnb(cands, size):
            nonlocal best
            if time.time() - start > timeout:
                return
            if size >= best[0]:
                return
            if cands == 0:
                if size > best[0]:
                    best[0] = size
                return
            if size + bin(cands).count('1') <= best[0]:
                return
            # Branch on vertex with max degree in candidate subgraph
            v = -1
            max_deg = -1
            c = cands
            while c:
                v_bit = c & -c
                vi = v_bit.bit_length() - 1
                deg = bin(cands & non_adj[vi]).count('1')
                if deg > max_deg:
                    max_deg = deg
                    v = vi
                c ^= v_bit
            bnb(cands & non_adj[v], size + 1)
            bnb(cands ^ (1 << v), size)
        
        bnb((1 << n) - 1, 0)
        return best[0]
    
    def try_n(n, attempts=400):
        for att in range(attempts):
            if time_up():
                return None
            adj_bits = [0] * n
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < 0.35:
                        if adj_bits[u] & adj_bits[v] == 0:
                            adj_bits[u] |= (1 << v)
                            adj_bits[v] |= (1 << u)
            
            for step in range(400):
                if time_up():
                    break
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v or (adj_bits[u] >> v) & 1:
                    continue
                if adj_bits[u] & adj_bits[v]:
                    continue
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
            
            alpha = max_independent_set(adj_bits, n, timeout=4)
            if alpha <= 9:
                return adj_bits
        return None
    
    for n in range(40, 25, -1):
        if time_up():
            break
        result = try_n(n)
        if result is not None:
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if (result[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
    
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i+1, n) if (j-i) % n % 3 == 1]}
