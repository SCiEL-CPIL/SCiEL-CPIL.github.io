import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 145.0 # Safety margin for 150s limit
    n = 40
    
    adj = [0] * n
    
    def update_adj(u, v, add=True):
        if add:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        else:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
    def can_add_edge(u, v):
        """
        Check if adding edge (u, v) creates a C6.
        """
        mask_u = adj[u]
        mask_v = adj[v]
        
        ma = mask_u
        while ma:
            lsb_a = ma & -ma
            ma ^= lsb_a
            a = lsb_a.bit_length() - 1
            
            md = mask_v
            while md:
                lsb_d = md & -md
                md ^= lsb_d
                d = lsb_d.bit_length() - 1
                
                if a == d:
                    continue
                
                # Candidates for b: neighbor of a, not u, v, a, d
                mask_b_cands = adj[a] & ~mask_u & ~mask_v & ~(1 << a) & ~(1 << d)
                mb = mask_b_cands
                while mb:
                    lsb_b = mb & -mb
                    mb ^= lsb_b
                    b = lsb_b.bit_length() - 1
                    
                    # Candidates for c: neighbor of b and d, not u, v, a, b, d
                    mask_c = adj[b] & adj[d] & ~mask_u & ~mask_v & ~(1 << a) & ~(1 << b) & ~(1 << d)
                    if mask_c:
                        return False
        
        return True

    def count_edges():
        return sum(x.bit_count() for x in adj) // 2

    def get_edges():
        return [[u, v] for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]

    def get_non_edges():
        return [[u, v] for u in range(n) for v in range(u + 1, n) if not ((adj[u] >> v) & 1)]

    def greedy_fill():
        changed = True
        while changed:
            changed = False
            non_edges = get_non_edges()
            random.shuffle(non_edges)
            for u, v in non_edges:
                if can_add_edge(u, v):
                    update_adj(u, v, True)
                    changed = True
                    # Break to restart shuffle? Or just continue?
                    # Continuing is fine, but shuffling again might be better for diversity.
                    # Let's just continue in this pass.
        return count_edges()

    def local_search(seed_val):
        random.seed(seed_val)
        
        best_edges = count_edges()
        best_adj = adj[:]
        
        T = 1.0
        T_min = 0.01
        cooling = 0.9999
        
        steps = 0
        max_steps = 1000000
        
        while time.time() < DEADLINE - 5:
            steps += 1
            if steps % 1000 == 0 and T > T_min:
                T *= cooling
                if T < T_min: T = T_min
                
            r = random.random()
            
            if r < 0.6:
                # Try to add edge
                non_edges = get_non_edges()
                if not non_edges:
                    continue
                u, v = random.choice(non_edges)
                if can_add_edge(u, v):
                    update_adj(u, v, True)
                    current_edges = count_edges()
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                        
            elif r < 0.8:
                # Try to remove edge
                edges = get_edges()
                if not edges:
                    continue
                u, v = random.choice(edges)
                update_adj(u, v, False)
                # Accept with some probability to explore
                if random.random() < 0.3:
                    pass
                else:
                    update_adj(u, v, True)
                    
            else:
                # Try swap
                edges = get_edges()
                non_edges = get_non_edges()
                if not edges or not non_edges:
                    continue
                u1, v1 = random.choice(edges)
                u2, v2 = random.choice(non_edges)
                
                update_adj(u1, v1, False)
                
                if can_add_edge(u2, v2):
                    update_adj(u2, v2, True)
                else:
                    update_adj(u1, v1, True)
            
            # Periodic greedy fill to reach local max
            if steps % 1000 == 0:
                greedy_fill()
                current_edges = count_edges()
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]

        return best_adj, best_edges

    best_adj = None
    best_m = 0
    
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > DEADLINE - 15:
            break
            
        adj = [0] * n
        # Random initialization
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.15:
                    update_adj(u, v, True)
                    
        m = greedy_fill()
        if m > best_m:
            best_m = m
            best_adj = adj[:]
            
        remaining = DEADLINE - 15 - time.time()
        if remaining > 5:
            ls_adj, ls_m = local_search(10000 + i)
            if ls_m > best_m:
                best_m = ls_m
                best_adj = ls_adj

    if best_adj is None:
        return []
        
    return get_edges()
