import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110
    n = 40
    
    # Initial state: 8 disjoint K5s (104 edges)
    edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append((5 * k + a, 5 * k + b))
    
    # Bitset representation of the graph for fast C6 checking
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    def has_c6_fast(adj_mask, u, v):
        """
        Check if adding edge (u, v) creates a C6.
        Path: u - w - x - y - z - v
        Optimized: iterate w, x, then check if any y in N(x) has a neighbor in N(v)
        with proper exclusions.
        """
        mask_u = adj_mask[u]
        mask_v = adj_mask[v]
        
        w_mask = mask_u & ~(1 << v)
        
        while w_mask:
            w_bit = w_mask & -w_mask
            w = w_bit.bit_length() - 1
            w_mask ^= w_bit
            
            mask_w = adj_mask[w]
            x_mask = mask_w & ~(1 << u) & ~(1 << v) & ~(1 << w)
            
            while x_mask:
                x_bit = x_mask & -x_mask
                x = x_bit.bit_length() - 1
                x_mask ^= x_bit
                
                # For each y in N(x) \ {u, v, w, x}, check if N(y) intersects N(v) \ {u, w, x, y}
                mask_x = adj_mask[x]
                y_mask = mask_x & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x)
                
                while y_mask:
                    y_bit = y_mask & -y_mask
                    y = y_bit.bit_length() - 1
                    y_mask ^= y_bit
                    
                    # Check if N(y) has any neighbor in N(v) excluding {u, w, x, y}
                    # z must be in N(y) and N(v), z not in {u, v, w, x, y}
                    z_mask = adj_mask[y] & mask_v & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x) & ~(1 << y)
                    if z_mask:
                        return True
        
        return False

    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)

    current_edges = set(edges)
    best_edges = set(edges)
    best_score = len(best_edges)
    
    # Simulated Annealing Parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    steps = 0
    last_improvement = 0
    
    while time.time() < deadline:
        # Pick a random move: add or remove an edge
        # Bias towards adding edges since we want to maximize
        if random.random() < 0.75:
            # Try to add a random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            if (u, v) in current_edges:
                continue
                
            # Check C6
            if has_c6_fast(adj_mask, u, v):
                continue
            
            # Move is valid (graph remains C6-free)
            add_edge(u, v)
            current_edges.add((u, v))
            new_score = len(current_edges)
            
            if new_score > best_score:
                best_score = new_score
                best_edges = set(current_edges)
                last_improvement = steps
        
        else:
            # Try to remove a random edge (to escape local optima)
            if not current_edges:
                continue
            e = random.choice(list(current_edges))
            u, v = e
            remove_edge(u, v)
            current_edges.remove(e)
            
            # Acceptance criterion for worsening move
            if random.random() < np.exp(-1.0 / T):
                pass # Accepted
            else:
                # Reject, revert
                add_edge(u, v)
                current_edges.add(e)
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Restart with higher temperature if stuck
            if steps - last_improvement > 500:
                T = max(0.5, T)
        
        steps += 1
    
    # Final greedy pass: try to add any remaining edges that don't create C6
    all_possible = [(a, b) for a in range(n) for b in range(a + 1, n)]
    for e in all_possible:
        if e in best_edges:
            continue
        u, v = e
        # Build temp adj_mask from best_edges
        temp_adj = [0] * n
        for a, b in best_edges:
            temp_adj[a] |= (1 << b)
            temp_adj[b] |= (1 << a)
        if not has_c6_fast(temp_adj, u, v):
            best_edges.add(e)
    
    return [[a, b] for (a, b) in sorted(best_edges)]
