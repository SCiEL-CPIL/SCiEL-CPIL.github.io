import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    n = 40
    
    def bitset_to_edges(adj):
        return [[u, v] for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]

    def count_edges(adj):
        return sum(bin(x).count('1') for x in adj) // 2

    def can_add_edge(adj, u, v):
        S1 = adj[u] & ~(1 << u) & ~(1 << v)
        if S1 == 0:
            return True
        
        S2 = 0
        a = S1
        while a:
            lsb = a & -a
            a ^= lsb
            idx = lsb.bit_length() - 1
            S2 |= adj[idx]
        S2 &= ~(1 << u) & ~(1 << v)
        
        if S2 == 0:
            return True
        
        S3 = 0
        b = S2
        while b:
            lsb = b & -b
            b ^= lsb
            idx = lsb.bit_length() - 1
            S3 |= adj[idx]
        S3 &= ~(1 << u) & ~(1 << v)
        S3 &= ~S1
        S3 &= ~S2
        
        if S3 == 0:
            return True
        
        S4 = 0
        c = S3
        while c:
            lsb = c & -c
            c ^= lsb
            idx = lsb.bit_length() - 1
            S4 |= adj[idx]
        S4 &= ~(1 << u) & ~(1 << v)
        S4 &= ~S1
        S4 &= ~S2
        S4 &= ~S3
        
        if (S4 & adj[v]) != 0:
            return False
        
        return True

    def greedy_fill(adj, rand):
        candidates = [(u, v) for u in range(n) for v in range(u + 1, n) if not ((adj[u] >> v) & 1)]
        rand.shuffle(candidates)
        for u, v in candidates:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

    def init_graph():
        adj = [0] * n
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    rand = random.Random(12345)
    
    # Initial solution
    best_adj = init_graph()
    greedy_fill(best_adj, rand)
    best_count = count_edges(best_adj)
    
    # SA Loop
    current_adj = best_adj[:]
    current_count = best_count
    
    T = 1.0
    T_min = 0.01
    alpha = 0.995
    tabu = []
    tabu_size = 10
    
    while time.time() < DEADLINE - 2:
        if T < T_min:
            # Reheat or break?
            # If we break, we are done.
            # If we reheat, we can explore more.
            # Let's break to save time for safety.
            break
        
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (current_adj[u] >> v) & 1:
                    edges.append((u, v))
        
        if not edges:
            break
            
        valid_edges = [e for e in edges if e not in tabu]
        if not valid_edges:
            valid_edges = edges
            tabu = []
            
        u, v = rand.choice(valid_edges)
        
        old_adj = current_adj[:]
        old_count = current_count
        
        current_adj[u] &= ~(1 << v)
        current_adj[v] &= ~(1 << u)
        
        greedy_fill(current_adj, rand)
        
        new_count = count_edges(current_adj)
        
        delta = new_count - old_count
        if delta >= 0 or rand.random() < math.exp(delta / T):
            current_count = new_count
            if new_count > best_count:
                best_count = new_count
                best_adj = current_adj[:]
            
            tabu.append((u, v))
            if len(tabu) > tabu_size:
                tabu.pop(0)
        else:
            current_adj = old_adj
            current_count = old_count
            
        T *= alpha
        
    return bitset_to_edges(best_adj)
