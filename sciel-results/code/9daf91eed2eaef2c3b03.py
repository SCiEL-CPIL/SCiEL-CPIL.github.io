import random
import time
import sys

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 115  # Safety margin
    
    # Adjacency as list of bitmasks
    adj = [0] * n
    
    def add_edge(u, v):
        if u > v: u, v = v, u
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        if u > v: u, v = v, u
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def has_edge(u, v):
        if u > v: u, v = v, u
        return (adj[u] >> v) & 1
        
    def get_max_book():
        max_bk = 0
        for u in range(n):
            mask = adj[u]
            # Only consider v > u
            mask &= ~((1 << (u + 1)) - 1)
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                bk = (adj[u] & adj[v]).bit_count()
                if bk > max_bk:
                    max_bk = bk
                mask &= mask - 1
        return max_bk

    # Initialize: K_{20,20} + (0,1)
    for a in range(20):
        for b in range(20, 40):
            add_edge(a, b)
    add_edge(0, 1)
    
    current_edges = [(a, b) for a in range(20) for b in range(20, 40)] + [(0, 1)]
    
    best_adj = adj[:]
    best_max = get_max_book()
    
    # Precompute all possible edges for random selection
    all_possible_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    while time.time() < deadline:
        # Find an edge with high book size to remove
        max_edge = None
        max_bk_val = 0
        for u, v in current_edges:
            bk = (adj[u] & adj[v]).bit_count()
            if bk > max_bk_val:
                max_bk_val = bk
                max_edge = (u, v)
                
        if max_edge is None:
            break
            
        u, v = max_edge
        
        # Remove it
        remove_edge(u, v)
        current_edges.remove((u, v))
        
        # Find best non-edge to add
        best_add_edge = None
        best_add_bk = n + 1
        
        # Iterate non-edges
        for x in range(n):
            for y in range(x + 1, n):
                if has_edge(x, y):
                    continue
                bk = (adj[x] & adj[y]).bit_count()
                if bk < best_add_bk:
                    best_add_bk = bk
                    best_add_edge = (x, y)
                    
        if best_add_edge is None:
            add_edge(u, v)
            current_edges.append((u, v))
            continue
            
        # Add the best edge
        add_edge(*best_add_edge)
        current_edges.append(best_add_edge)
        
        # Check if we improved
        new_max = get_max_book()
        if new_max < best_max:
            best_max = new_max
            best_adj = adj[:]
            if best_max <= 7:
                break
        
        # If we didn't improve, revert if it worsened
        if new_max > best_max:
            remove_edge(*best_add_edge)
            current_edges.remove(best_add_edge)
            add_edge(u, v)
            current_edges.append((u, v))
            
            # Try a random swap to escape local minima
            if random.random() < 0.1:
                e_rm = random.choice(current_edges)
                remove_edge(*e_rm)
                current_edges.remove(e_rm)
                
                while True:
                    e_add = random.choice(all_possible_edges)
                    if not has_edge(*e_add):
                        break
                add_edge(*e_add)
                current_edges.append(e_add)
                
                rand_max = get_max_book()
                if rand_max > best_max:
                    remove_edge(*e_add)
                    current_edges.remove(e_add)
                    add_edge(*e_rm)
                    current_edges.append(e_rm)

    # Convert best_adj to edge list
    result = []
    for u in range(n):
        mask = best_adj[u]
        mask &= ~((1 << (u + 1)) - 1)
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            result.append([u, v])
            mask &= mask - 1
            
    if len(result) != 401:
        result = []
        for a in range(20):
            for b in range(20, 40):
                result.append([a, b])
        result.append([0, 1])
        
    return result
