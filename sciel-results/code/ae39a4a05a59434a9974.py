import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 20
    adj = [0] * n
    
    # Fast popcount
    if hasattr(int, 'bit_count'):
        def popcount(x):
            return x.bit_count()
    else:
        _pc = [0] * 256
        for i in range(256):
            _pc[i] = bin(i).count('1')
        def popcount(x):
            c = 0
            while x:
                c += _pc[x & 0xFF]
                x >>= 8
            return c
        
    def can_add(u, v):
        if (adj[u] >> v) & 1:
            return False
            
        # 1. Pair (u,v) common neighbors must be <= 2
        if popcount(adj[u] & adj[v]) > 2:
            return False
            
        # 2. For all w in N(v), pair (u,w) gains v as common neighbor.
        #    So current common(u,w) must be <= 1.
        neighbors_v = adj[v]
        while neighbors_v:
            lsb = neighbors_v & -neighbors_v
            w = lsb.bit_length() - 1
            neighbors_v ^= lsb
            if w == u:
                continue
            if popcount(adj[u] & adj[w]) > 1:
                return False
                
        # 3. For all w in N(u), pair (v,w) gains u as common neighbor.
        #    So current common(v,w) must be <= 1.
        neighbors_u = adj[u]
        while neighbors_u:
            lsb = neighbors_u & -neighbors_u
            w = lsb.bit_length() - 1
            neighbors_u ^= lsb
            if w == v:
                continue
            if popcount(adj[v] & adj[w]) > 1:
                return False
                
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def edge_count():
        total_deg = 0
        for i in range(n):
            total_deg += popcount(adj[i])
        return total_deg // 2

    def get_neighbors(u):
        result = []
        temp = adj[u]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            result.append(w)
            temp ^= lsb
        return result

    def greedy_init():
        nonlocal adj
        adj = [0] * n
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if can_add(u, v):
                add_edge(u, v)
        return edge_count()

    # Multiple greedy restarts
    best_adj = [0] * n
    best_count = 0
    
    for _ in range(10):
        cnt = greedy_init()
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
            
    # Simulated Annealing
    T = 10.0
    cooling = 0.995
    min_T = 0.1
    stagnation = 0
    max_stagnation = 1000
    
    while time.time() < deadline:
        if T < min_T:
            T = 5.0
            adj = best_adj[:]
            stagnation = 0
            continue
            
        current_count = edge_count()
        r = random.random()
        
        if r < 0.6:
            # Add edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or (adj[u] >> v) & 1:
                continue
            if can_add(u, v):
                add_edge(u, v)
                new_count = edge_count()
                if new_count > best_count:
                    best_count = new_count
                    best_adj = adj[:]
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                stagnation += 1
        elif r < 0.8:
            # Remove edge
            u = random.randint(0, n - 1)
            neigh_list = get_neighbors(u)
            if not neigh_list:
                continue
            v = random.choice(neigh_list)
            remove_edge(u, v)
            new_count = edge_count()
            if random.random() < math.exp(-1.0 / T):
                if new_count > best_count:
                    best_count = new_count
                    best_adj = adj[:]
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                add_edge(u, v)
                stagnation += 1
        else:
            # Swap
            u1 = random.randint(0, n - 1)
            neigh_list1 = get_neighbors(u1)
            if not neigh_list1:
                continue
            v1 = random.choice(neigh_list1)
            
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2 or (adj[u2] >> v2) & 1:
                continue
            
            remove_edge(u1, v1)
            if can_add(u2, v2):
                add_edge(u2, v2)
                new_count = edge_count()
                if new_count > best_count:
                    best_count = new_count
                    best_adj = adj[:]
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                add_edge(u1, v1)
                stagnation += 1
                
        T *= cooling
        
        if stagnation > max_stagnation:
            # Kick: remove 5 random edges
            adj = best_adj[:]
            for _ in range(5):
                u = random.randint(0, n - 1)
                neigh_list = get_neighbors(u)
                if neigh_list:
                    v = random.choice(neigh_list)
                    remove_edge(u, v)
            T = 5.0
            stagnation = 0

    # Final polish
    adj = best_adj[:]
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    continue
                if can_add(u, v):
                    add_edge(u, v)
                    improved = True
                    if edge_count() > best_count:
                        best_count = edge_count()
                        best_adj = adj[:]
        if improved:
            adj = best_adj[:]
            
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
