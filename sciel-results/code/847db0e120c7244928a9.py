import time
import random
import sys

def construct():
    n = 40
    start_time = time.time()
    deadline = 115  # Safety margin: 5s buffer before 120s limit
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Bitset adjacency
    adj = [0] * n
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def edge_count():
        return sum(bin(a).count("1") for a in adj) // 2

    def can_add(u, v):
        """
        Returns True if adding edge (u,v) does not create a C6.
        """
        Nv = adj[v]
        Nu = adj[u]
        
        x_mask = Nv
        while x_mask:
            x = (x_mask & -x_mask).bit_length() - 1
            x_mask &= x_mask - 1
            
            if x == u: continue
            
            w_mask = Nu
            while w_mask:
                w = (w_mask & -w_mask).bit_length() - 1
                w_mask &= w_mask - 1
                
                if w == v or w == x: continue
                
                S_x = adj[x] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                S_w = adj[w] & ~(1 << u) & ~(1 << v) & ~(1 << x)
                
                y_mask = S_x
                while y_mask:
                    y = (y_mask & -y_mask).bit_length() - 1
                    y_mask &= y_mask - 1
                    
                    if adj[y] & (S_w & ~(1 << y)):
                        return False
        return True

    def get_edges():
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                j = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if i < j:
                    edges.append([i, j])
        return edges

    random.seed(42)
    
    best_edges = 0
    best_adj = [0] * n
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Phase 1: Random Greedy
    # Try several random orderings
    for trial in range(30):
        if time_left() < 15:
            break
            
        adj_try = [0] * n
        ec = 0
        
        shuffled = all_pairs[:]
        random.shuffle(shuffled)
        
        for u, v in shuffled:
            if time_left() < 5:
                break
            
            if can_add(u, v):
                adj_try[u] |= (1 << v)
                adj_try[v] |= (1 << u)
                ec += 1
        
        if ec > best_edges:
            best_edges = ec
            best_adj = adj_try[:]

    if best_edges == 0:
        # Fallback: 8 disjoint K5s
        for k in range(8):
            for a in range(5):
                for b in range(a+1, 5):
                    u = 5*k+a
                    v = 5*k+b
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
        best_edges = 8 * 10

    # Phase 2: Simulated Annealing
    adj = best_adj[:]
    current_edges = best_edges
    
    # List of current edges for fast removal/swap
    def get_edge_list():
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                j = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if i < j:
                    edges.append((i, j))
        return edges

    current_edges_list = get_edge_list()
    
    T = 1.0
    cooling = 0.99
    min_T = 0.01
    
    while time_left() > 2 and T > min_T:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.7:
            # Try to add a random non-edge
            # Sample until we find a non-edge or give up (unlikely to give up)
            found = False
            for _ in range(10):
                u, v = random.choice(all_pairs)
                if not (adj[u] & (1 << v)):
                    found = True
                    break
            
            if found:
                if can_add(u, v):
                    add_edge(u, v)
                    current_edges += 1
                    current_edges_list.append((u, v))
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                else:
                    # Reject (no change)
                    pass
        elif move_type < 0.9:
            # Try to remove a random edge
            if current_edges_list:
                u, v = random.choice(current_edges_list)
                remove_edge(u, v)
                current_edges -= 1
                current_edges_list.remove((u, v))
                # Accept if random() < exp(-1/T)
                if random.random() < (2.71828 ** (-1.0 / T)):
                    pass # Accept
                else:
                    # Reject: Restore
                    add_edge(u, v)
                    current_edges += 1
                    current_edges_list.append((u, v))
        else:
            # Try a swap: Remove one edge, add another
            if current_edges_list:
                u, v = random.choice(current_edges_list)
                remove_edge(u, v)
                current_edges -= 1
                current_edges_list.remove((u, v))
                
                # Try to add a random non-edge
                added = False
                for _ in range(10):
                    a, b = random.choice(all_pairs)
                    if a == u and b == v: continue
                    if a == v and b == u: continue
                    if not (adj[a] & (1 << b)):
                        if can_add(a, b):
                            add_edge(a, b)
                            current_edges += 1
                            current_edges_list.append((a, b))
                            added = True
                            # Accept swap (edge count same as before move, but structure changed)
                            if current_edges > best_edges:
                                best_edges = current_edges
                                best_adj = adj[:]
                            break
                
                if not added:
                    # Reject swap: Restore original edge
                    add_edge(u, v)
                    current_edges += 1
                    current_edges_list.append((u, v))
        
        T *= cooling

    # Final cleanup: try to add any remaining possible edges
    # Do a full pass to ensure we are at a local maximum
    changed = True
    while changed and time_left() > 1:
        changed = False
        for u, v in all_pairs:
            if not (adj[u] & (1 << v)):
                if can_add(u, v):
                    add_edge(u, v)
                    current_edges += 1
                    changed = True
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]

    return get_edges()
