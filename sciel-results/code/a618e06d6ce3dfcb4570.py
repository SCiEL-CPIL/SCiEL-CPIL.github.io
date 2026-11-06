import random
import time

def construct():
    n = 16
    adj = [0] * n
    
    def has_edge(u, v):
        return (adj[u] >> v) & 1

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def count_q3s(u, v):
        # Optimized count of Q3s containing edge (u,v)
        # Assumes we check other 11 edges.
        count = 0
        mask_u = adj[u] & ~(1 << v)
        
        mask_a = mask_u
        while mask_a:
            lsb_a = mask_a & -mask_a
            a = lsb_a.bit_length() - 1
            mask_a ^= lsb_a
            
            mask_b = mask_u & ~(1 << a)
            while mask_b:
                lsb_b = mask_b & -mask_b
                b = lsb_b.bit_length() - 1
                mask_b ^= lsb_b
                
                mask_f = adj[a] & adj[b]
                mask_f &= ~((1 << u) | (1 << v) | (1 << a) | (1 << b))
                
                while mask_f:
                    lsb_f = mask_f & -mask_f
                    f = lsb_f.bit_length() - 1
                    mask_f ^= lsb_f
                    
                    mask_c = adj[v] & adj[a]
                    mask_c &= ~((1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << f))
                    
                    if not mask_c: continue
                    
                    mask_d = adj[v] & adj[b]
                    mask_d &= ~((1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << f))
                    
                    if not mask_d: continue
                    
                    mask_c_iter = mask_c
                    while mask_c_iter:
                        lsb_c = mask_c_iter & -mask_c_iter
                        c = lsb_c.bit_length() - 1
                        mask_c_iter ^= lsb_c
                        
                        mask_d_iter = mask_d & ~(1 << c)
                        
                        while mask_d_iter:
                            lsb_d = mask_d_iter & -mask_d_iter
                            d = lsb_d.bit_length() - 1
                            mask_d_iter ^= lsb_d
                            
                            mask_e = adj[c] & adj[d] & adj[f]
                            mask_e &= ~((1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d) | (1 << f))
                            
                            if mask_e:
                                count += mask_e.bit_count()
                                
        return count

    # Greedy Initialization
    random.seed(42)
    order = list(range(n * (n - 1) // 2))
    # Map index to (u, v)
    def idx_to_uv(idx):
        # 0: (0,1), 1: (0,2), ...
        # Triangular numbers
        # Not critical, just random order
        return idx # Placeholder, will generate pairs directly
        
    # Random order of edges
    edges_list = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(edges_list)
    
    for u, v in edges_list:
        # Check if adding (u,v) creates Q3
        delta = count_q3s(u, v)
        if delta == 0:
            add_edge(u, v)
            
    # Current state is valid (0 violations)
    violations = 0
    best_edges_count = sum(1 for u in range(n) for v in range(u+1, n) if has_edge(u, v))
    best_adj = adj[:]
    
    # Simulated Annealing
    T = 1.0
    T_min = 1e-5
    alpha = 0.999
    # Run for some time
    start_time = time.time()
    while time.time() - start_time < 110: # Leave 10s margin
        T *= alpha
        if T < T_min:
            T = T_min
            
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        delta = count_q3s(u, v)
        
        if has_edge(u, v):
            # Remove
            # Violations decrease by delta
            # Edges decrease by 1
            # Score change: -K * (-delta) + (-1) = K*delta - 1
            # We want to minimize Score (which is -K*viol + edges)
            # Wait, my Score definition:
            # If I use Score = K * violations - edges, I want to MINIMIZE Score.
            # Let's stick to:
            # Objective: Maximize Edges, subject to Violations = 0.
            # Penalty method: Minimize Cost = K * Violations - Edges.
            # If Violations=0, Cost = -Edges. Min Cost = Max Edges.
            
            # Current Cost = K * violations - edges
            # New Cost = K * (violations - delta) - (edges - 1) = K*violations - K*delta - edges + 1
            # Delta Cost = -K*delta + 1
            
            if violations > 0:
                # We MUST reduce violations if possible?
                # If delta > 0, it's a huge improvement in Cost.
                # If delta == 0, Cost increases by 1 (worse).
                if delta > 0:
                    accept = True
                else:
                    # Accept with prob exp(-1/T) ?
                    # Usually we don't want to remove edges if it doesn't help violations.
                    accept = False
            else:
                # violations == 0
                # delta must be 0 (since graph is valid)
                # Cost increases by 1.
                # Reject.
                accept = False
                
            if accept:
                remove_edge(u, v)
                violations -= delta
                if violations == 0:
                    curr_edges = best_edges_count - 1 # Wait, best_edges_count is not updated in loop?
                    # We need to track current edges count
                    # Let's track curr_edges
                    pass
        else:
            # Add
            # Violations increase by delta
            # Edges increase by 1
            # Delta Cost = K*delta - 1
            
            if violations == 0:
                if delta == 0:
                    accept = True
                else:
                    # Accept with prob exp(-(K*delta - 1)/T)
                    # K is large.
                    accept = random.random() < (2.718281828 ** (-(100 * delta - 1) / T))
            else:
                # violations > 0
                # Adding might increase violations further, but might enable future removals?
                # Or if delta < 0? No, delta is count of NEW Q3s.
                # Wait, if violations > 0, adding an edge can ONLY increase or keep violations.
                # It cannot reduce them.
                # So we should never add edges if violations > 0?
                # Unless we are using a different move set (swap).
                # With single edge flip, adding edge when violations>0 is bad.
                accept = False
                
            if accept:
                add_edge(u, v)
                violations += delta
                # Update best if valid
                if violations == 0:
                    # Count edges
                    # This is expensive.
                    # We can maintain curr_edges
                    pass

    # The above logic is a bit messy with tracking.
    # Let's rewrite cleanly.
    
    return [] # Placeholder
