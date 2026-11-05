import time
import random
import numpy as np

def construct():
    n = 16
    t_start = time.time()
    deadline = t_start + 115.0

    # Precompute Q3 edges
    _Q3_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    _Q3_ADJ = [0] * 8
    for a, b in _Q3_EDGES:
        _Q3_ADJ[a] |= (1 << b)
        _Q3_ADJ[b] |= (1 << a)

    def has_q3_fast(adj):
        """
        Check if graph with adjacency masks 'adj' contains Q3.
        """
        for u in range(n):
            neu = adj[u]
            if bin(neu).count('1') < 3:
                continue
                
            assignment = [-1] * 8
            assignment[0] = u
            
            def bt(q):
                if q == 8:
                    for a, b in _Q3_EDGES:
                        if not (adj[assignment[a]] & (1 << assignment[b])):
                            return False
                    return True
                
                candidates = 0
                assigned_neighbors_in_q3 = 0
                for y in range(8):
                    if assignment[y] != -1 and (_Q3_ADJ[q] & (1 << y)):
                        assigned_neighbors_in_q3 |= (1 << y)
                        
                if assigned_neighbors_in_q3 == 0:
                    candidates = (1 << n) - 1
                else:
                    mask = assigned_neighbors_in_q3
                    first = True
                    while mask:
                        lsb = mask & (-mask)
                        y = lsb.bit_length() - 1
                        if first:
                            candidates = adj[assignment[y]]
                            first = False
                        else:
                            candidates &= adj[assignment[y]]
                        mask ^= lsb

                used_mask = 0
                for g in assignment:
                    if g != -1:
                        used_mask |= (1 << g)
                candidates &= ~used_mask
                
                while candidates:
                    lsb = candidates & (-candidates)
                    g = lsb.bit_length() - 1
                    candidates ^= lsb
                    
                    assignment[q] = g
                    if bt(q + 1):
                        return True
                    assignment[q] = -1
                return False

            if bt(1):
                return True
        
        return False

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def edge_list(adj):
        es = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    es.append([u, v])
        return es

    def find_q3_witness(adj):
        for u in range(n):
            if bin(adj[u]).count('1') < 3:
                continue
            assignment = [-1] * 8
            assignment[0] = u
            
            def bt(q):
                if q == 8:
                    for a, b in _Q3_EDGES:
                        if not (adj[assignment[a]] & (1 << assignment[b])):
                            return None
                    return list(assignment)
                
                assigned_neighbors_in_q3 = 0
                for y in range(8):
                    if assignment[y] != -1 and (_Q3_ADJ[q] & (1 << y)):
                        assigned_neighbors_in_q3 |= (1 << y)
                        
                if assigned_neighbors_in_q3 == 0:
                    candidates = (1 << n) - 1
                else:
                    mask = assigned_neighbors_in_q3
                    first = True
                    candidates = 0
                    while mask:
                        lsb = mask & (-mask)
                        y = lsb.bit_length() - 1
                        if first:
                            candidates = adj[assignment[y]]
                            first = False
                        else:
                            candidates &= adj[assignment[y]]
                        mask ^= lsb

                used_mask = 0
                for g in assignment:
                    if g != -1:
                        used_mask |= (1 << g)
                candidates &= ~used_mask
                
                while candidates:
                    lsb = candidates & (-candidates)
                    g = lsb.bit_length() - 1
                    candidates ^= lsb
                    
                    assignment[q] = g
                    res = bt(q + 1)
                    if res:
                        return res
                    assignment[q] = -1
                return None

            res = bt(1)
            if res:
                return res
        return None

    # Start with K16
    adj = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                adj[i] |= (1 << j)
    
    best_count = -1
    best_edges = [] # Empty graph is always Q3-free

    # Main loop: Remove edges until Q3-free
    while time.time() < deadline - 10:
        if not has_q3_fast(adj):
            current_count = count_edges(adj)
            if current_count > best_count:
                best_count = current_count
                best_edges = edge_list(adj)
            break
        
        witness = find_q3_witness(adj)
        if not witness:
            break
            
        q3_edges_in_g = set()
        for a, b in _Q3_EDGES:
            u, v = witness[a], witness[b]
            if u > v: u, v = v, u
            q3_edges_in_g.add((u, v))
        
        # Choose edge to remove
        edge_to_remove = random.choice(list(q3_edges_in_g))
        u, v = edge_to_remove
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # If we exited due to time, best_edges might be empty or from a previous valid state.
    # If best_edges is empty, we might have never found a Q3-free state.
    # But if we started with K16 and removed edges, we should eventually hit Q3-free.
    # However, if we timed out *before* hitting Q3-free, best_edges is [] (score 0).
    
    # To improve, we can try to add edges back from the *current* adj if it is Q3-free?
    # But if we timed out, adj is NOT Q3-free.
    
    # So we must ensure we return a valid graph.
    # If best_edges is empty, return empty graph.
    
    if best_edges:
        adj = [0] * n
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        current_count = count_edges(adj)

        # Greedy add back
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline - 5:
                break
            if not (adj[u] & (1 << v)):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if has_q3_fast(adj):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_edges = edge_list(adj)

    return best_edges
