import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 5.0
    
    n_target = 50
    n_edges = n_target * (n_target - 1) // 2
    
    # Precompute edge indices and pairs
    edge_pairs = []
    for i in range(n_target):
        for j in range(i + 1, n_target):
            edge_pairs.append((i, j))
            
    edge_index = [[-1] * n_target for _ in range(n_target)]
    for idx, (u, v) in enumerate(edge_pairs):
        edge_index[u][v] = idx
        edge_index[v][u] = idx

    def get_idx(u, v):
        return edge_index[u][v]

    # Base 32 construction
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    def build_base_32():
        colors_32 = [0] * (32 * 31 // 2)
        # Map local index to global index? 
        # Easier: create a 50-length array, fill 32 part, return the 32 part extracted.
        # But we need the 50-length array for search.
        pass

    colors = [0] * n_edges
    
    # Fill base 32
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors[get_idx(i, j)] = log[i ^ j] % 3
            else:
                colors[get_idx(i, j)] = 3
                
    # Fill rest randomly
    for i in range(32):
        for j in range(i + 1, n_target):
            if j >= 32:
                colors[get_idx(i, j)] = random.randint(0, 3)
    for i in range(32, n_target):
        for j in range(i + 1, n_target):
            colors[get_idx(i, j)] = random.randint(0, 3)

    def build_adjacency(colors):
        adj = [[0] * n_target for _ in range(4)]
        for u in range(n_target):
            for v in range(u + 1, n_target):
                c = colors[get_idx(u, v)]
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
        return adj

    def initial_total_conflicts(adj):
        total = 0
        for u in range(n_target):
            for v in range(u + 1, n_target):
                for c in range(4):
                    total += (adj[c][u] & adj[c][v]).bit_count()
        return total // 3

    adj = build_adjacency(colors)
    total_conflicts = initial_total_conflicts(adj)
    
    if total_conflicts == 0:
        return {'n': n_target, 'colors': colors}

    # Min-Conflicts
    max_iters = 1000000
    T = 1.0
    
    for it in range(max_iters):
        if time.time() - start_time > deadline - safety_margin:
            break
            
        idx = random.randint(0, n_edges - 1)
        u, v = edge_pairs[idx]
        old_c = colors[idx]
        
        best_new_c = old_c
        min_delta = float('inf')
        
        old_conf = (adj[old_c][u] & adj[old_c][v]).bit_count()
        
        for new_c in range(4):
            if new_c == old_c:
                continue
            new_conf = (adj[new_c][u] & adj[new_c][v]).bit_count()
            delta = new_conf - old_conf
            if delta < min_delta:
                min_delta = delta
                best_new_c = new_c
        
        if min_delta <= 0 or (random.random() < math.exp(-min_delta / T)):
            colors[idx] = best_new_c
            adj[old_c][u] ^= (1 << v)
            adj[old_c][v] ^= (1 << u)
            adj[best_new_c][u] ^= (1 << v)
            adj[best_new_c][v] ^= (1 << u)
            total_conflicts += min_delta
            
            if total_conflicts == 0:
                return {'n': n_target, 'colors': colors}
        
        T *= 0.999
        if T < 0.1:
            T = 0.1

    # Fallback: Return 32
    # Extract 32-vertex coloring
    colors_32 = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors_32.append(colors[get_idx(i, j)])
            
    return {'n': 32, 'colors': colors_32}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
