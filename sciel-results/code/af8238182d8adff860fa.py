import time
import random

def _is_triangle_free(adj_masks):
    n = 16
    for u in range(n):
        v_mask = adj_masks[u]
        while v_mask:
            v_bit = v_mask & -v_mask
            v = v_bit.bit_length() - 1
            v_mask ^= v_bit
            if u < v: # Only check each edge once
                if adj_masks[u] & adj_masks[v]:
                    return False
    return True

def _is_3_colorable(adj_masks):
    n = 16
    degrees = [bin(mask).count('1') for mask in adj_masks]
    order = sorted(range(n), key=lambda v: degrees[v], reverse=True)
    
    color = [-1] * n
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        forbidden = 0
        neighbors = adj_masks[v]
        while neighbors:
            u_bit = neighbors & -neighbors
            u = u_bit.bit_length() - 1
            neighbors ^= u_bit
            if color[u] != -1:
                forbidden |= (1 << color[u])
        
        for c in range(3):
            if not (forbidden & (1 << c)):
                color[v] = c
                if backtrack(idx + 1):
                    return True
                color[v] = -1
        return False
    
    return backtrack(0)

def construct():
    deadline = time.time() + 110
    n = 16
    
    adj_masks = [0] * n
    def add_edge(u, v):
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    # Correct Grötzsch Graph (Mycielski of C5) on vertices 0-10
    # C5: 0-1-2-3-4-0
    add_edge(0, 1)
    add_edge(1, 2)
    add_edge(2, 3)
    add_edge(3, 4)
    add_edge(4, 0)
    
    # Copies 5-9
    # 5 (copy of 0) connects to neighbors of 0 in C5: 1, 4
    add_edge(5, 1)
    add_edge(5, 4)
    
    # 6 (copy of 1) connects to neighbors of 1 in C5: 0, 2
    add_edge(6, 0)
    add_edge(6, 2)
    
    # 7 (copy of 2) connects to neighbors of 2 in C5: 1, 3
    add_edge(7, 1)
    add_edge(7, 3)
    
    # 8 (copy of 3) connects to neighbors of 3 in C5: 2, 4
    add_edge(8, 2)
    add_edge(8, 4)
    
    # 9 (copy of 4) connects to neighbors of 4 in C5: 3, 0
    add_edge(9, 3)
    add_edge(9, 0)
    
    # Apex 10 connects to all copies 5-9
    add_edge(10, 5)
    add_edge(10, 6)
    add_edge(10, 7)
    add_edge(10, 8)
    add_edge(10, 9)
    
    # Vertices 11-15 are isolated
    
    # Verify initial graph is valid
    if not _is_triangle_free(adj_masks) or _is_3_colorable(adj_masks):
        # Fallback to empty graph if construction is flawed (shouldn't happen)
        adj_masks = [0] * n
        
    best_adj = [m for m in adj_masks]
    best_edges = sum(bin(m).count('1') for m in best_adj) // 2
    
    current_adj = [m for m in adj_masks]
    current_edges = best_edges
    
    def get_edge_count(adj):
        return sum(bin(m).count('1') for m in adj) // 2

    temp = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        move_type = random.choice([0, 1, 2])
        new_adj = [m for m in current_adj]
        
        if move_type == 0: # Add edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v or (new_adj[u] & (1 << v)):
                continue
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
        elif move_type == 1: # Remove edge
            edges = []
            for u in range(n):
                v_mask = new_adj[u]
                while v_mask:
                    v_bit = v_mask & -v_mask
                    v = v_bit.bit_length() - 1
                    v_mask ^= v_bit
                    if u < v:
                        edges.append((u, v))
            if not edges:
                continue
            u, v = random.choice(edges)
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
        else: # Swap
            edges = []
            for u in range(n):
                v_mask = new_adj[u]
                while v_mask:
                    v_bit = v_mask & -v_mask
                    v = v_bit.bit_length() - 1
                    v_mask ^= v_bit
                    if u < v:
                        edges.append((u, v))
            if not edges:
                continue
            u1, v1 = random.choice(edges)
            new_adj[u1] &= ~(1 << v1)
            new_adj[v1] &= ~(1 << u1)
            
            u2 = random.randint(0, n-1)
            v2 = random.randint(0, n-1)
            if u2 == v2 or (new_adj[u2] & (1 << v2)):
                continue
            new_adj[u2] |= (1 << v2)
            new_adj[v2] |= (1 << u2)
            
        # Check validity
        if not _is_triangle_free(new_adj):
            continue
        if _is_3_colorable(new_adj):
            continue
            
        new_edges = get_edge_count(new_adj)
        
        if new_edges >= current_edges:
            current_adj = new_adj
            current_edges = new_edges
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = [m for m in current_adj]
        else:
            diff = current_edges - new_edges
            prob = 2.71828 ** (-diff / temp)
            if random.random() < prob:
                current_adj = new_adj
                current_edges = new_edges
        
        temp *= cooling
        if temp < 0.01:
            temp = 0.01

    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
