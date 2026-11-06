import numpy as np
import time
import random

def _count_c6_through_edge(adj_mask, n, u, v):
    """
    Count the number of C6s in the graph defined by adj_mask that contain the edge (u, v).
    adj_mask is a list of n integers, where adj_mask[i] is a bitmask of neighbors of i.
    We assume (u, v) is NOT in adj_mask currently (or we treat it as removed).
    Actually, it's easier to pass the mask WITHOUT the edge (u,v).
    """
    count = 0
    # We need paths v - a - b - c - d - u of length 4.
    # a in N(v)
    # b in N(a) \ {v}
    # c in N(b) \ {v, a}
    # d in N(c) \ {v, a, b}
    # Check if u in N(d)
    
    neighbors_v = adj_mask[v]
    # Iterate a
    a = neighbors_v
    while a:
        # Get lowest set bit
        la = a & -a
        a_idx = la.bit_length() - 1
        a ^= la
        
        if a_idx == u: continue # Path cannot go v-u...
        
        neighbors_a = adj_mask[a_idx]
        # Remove v and u from neighbors_a for b
        neighbors_b = neighbors_a & ~(1 << v) & ~(1 << u)
        
        b = neighbors_b
        while b:
            lb = b & -b
            b_idx = lb.bit_length() - 1
            b ^= lb
            
            neighbors_b_idx = adj_mask[b_idx]
            # Remove v, u, a for c
            neighbors_c = neighbors_b_idx & ~(1 << v) & ~(1 << u) & ~(1 << a_idx)
            
            c = neighbors_c
            while c:
                lc = c & -c
                c_idx = lc.bit_length() - 1
                c ^= lc
                
                neighbors_c_idx = adj_mask[c_idx]
                # Remove v, u, a, b for d
                neighbors_d = neighbors_c_idx & ~(1 << v) & ~(1 << u) & ~(1 << a_idx) & ~(1 << b_idx)
                
                d = neighbors_d
                while d:
                    ld = d & -d
                    d_idx = ld.bit_length() - 1
                    d ^= ld
                    
                    # Check if u is neighbor of d
                    if adj_mask[d_idx] & (1 << u):
                        count += 1
                        
    return count

def _build_masks_from_colors(colors, n):
    masks = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            masks[c][i] |= (1 << j)
            masks[c][j] |= (1 << i)
            idx += 1
    return masks

def _count_total_c6s(masks, n):
    total = 0
    for c in range(4):
        adj = masks[c]
        # Count C6s. To avoid double counting, we can sum counts for edges (u,v) with u<v and divide by 6?
        # Each C6 has 6 edges.
        # If we sum count_c6_through_edge for all edges, each C6 is counted 6 times.
        # So total = sum / 6.
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    # Remove edge (u,v) temporarily for counting
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    total += _count_c6_through_edge(adj, n, u, v)
                    # Restore
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
    return total // 6

def _spectral_init(n, seed):
    """
    Generate initial edge colors using spectral embedding of a random regular graph.
    """
    rng = np.random.RandomState(seed)
    
    # Generate a random d-regular graph
    d = min(10, n - 1)
    if d % 2 == 1: d -= 1
    if d < 2: d = 2
    
    adj = np.zeros((n, n), dtype=int)
    for _ in range(d // 2):
        perm = rng.permutation(n)
        for i in range(n // 2):
            u = perm[2*i]
            v = perm[2*i+1]
            if u != v and adj[u, v] == 0:
                adj[u, v] = 1
                adj[v, u] = 1
    
    np.fill_diagonal(adj, 0)
    
    # Laplacian
    degree = adj.sum(axis=1)
    D = np.diag(degree)
    L = D - adj
    
    eigvals, eigvecs = np.linalg.eigh(L)
    
    if n <= 4:
        return [(i + j) % 4 for i in range(n) for j in range(i+1, n)]
        
    # Use top 4 non-trivial eigenvectors
    vecs = eigvecs[:, 1:5]
    
    # Random rotation
    R = rng.randn(4, 4)
    R, _ = np.linalg.qr(R)
    
    rotated_points = R @ vecs.T # Shape (4, n)
    
    edge_colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = rotated_points[:, j] - rotated_points[:, i]
            signs = np.sign(diff)
            signs[signs == 0] = 1
            b0 = 1 if signs[0] > 0 else 0
            b1 = 1 if signs[1] > 0 else 0
            b2 = 1 if signs[2] > 0 else 0
            b3 = 1 if signs[3] > 0 else 0
            idx = b0*8 + b1*4 + b2*2 + b3
            color = idx % 4
            edge_colors.append(color)
            
    return edge_colors

def _simulated_annealing(colors, n, time_limit=50.0):
    """
    Minimize the number of C6s using Simulated Annealing.
    """
    masks = _build_masks_from_colors(colors, n)
    current_energy = _count_total_c6s(masks, n)
    
    if current_energy == 0:
        return colors
    
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    start_time = time.time()
    rng = random.Random()
    
    while T > T_min and (time.time() - start_time) < time_limit:
        # Pick a random edge
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        # Find current color of (i, j)
        # We need to map (i,j) to index in colors list?
        # Or just look at masks.
        # Let's find which mask has the edge.
        old_color = -1
        for c in range(4):
            if masks[c][i] & (1 << j):
                old_color = c
                break
        
        if old_color == -1:
            continue # Should not happen for K_n coloring
            
        # Pick new color
        new_color = rng.randint(0, 3)
        if new_color == old_color:
            new_color = (new_color + 1) % 4
            
        # Calculate Delta Energy
        # Remove (i,j) from old_color
        masks[old_color][i] ^= (1 << j)
        masks[old_color][j] ^= (1 << i)
        
        c6_removed = _count_c6_through_edge(masks[old_color], n, i, j)
        
        # Add (i,j) to new_color
        # Before adding, count C6s that would be formed
        c6_added = _count_c6_through_edge(masks[new_color], n, i, j)
        
        # Add edge to new_color
        masks[new_color][i] |= (1 << j)
        masks[new_color][j] |= (1 << i)
        
        delta_E = c6_added - c6_removed
        
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            # Accept
            current_energy += delta_E
            if current_energy == 0:
                break
        else:
            # Reject: revert
            masks[new_color][i] ^= (1 << j)
            masks[new_color][j] ^= (1 << i)
            masks[old_color][i] |= (1 << j)
            masks[old_color][j] |= (1 << i)
            
        T *= cooling
        
    # Extract final colors from masks
    final_colors = []
    for i in range(n):
        for j in range(i + 1, n):
            for c in range(4):
                if masks[c][i] & (1 << j):
                    final_colors.append(c)
                    break
            else:
                final_colors.append(0) # Fallback, shouldn't happen
                
    return final_colors

def construct():
    """Construct a 4-coloring of K_n without C6 in any color class, maximizing n."""
    
    best_n = 1
    best_colors = [0]
    
    # Try n from 17 down to 10
    for n in range(17, 9, -1):
        found = False
        # Try a few seeds
        for seed in range(10):
            colors = _spectral_init(n, seed)
            colors = _simulated_annealing(colors, n, time_limit=5.0)
            
            # Verify
            masks = _build_masks_from_colors(colors, n)
            if _count_total_c6s(masks, n) == 0:
                best_n = n
                best_colors = colors
                found = True
                break
        
        if found:
            return {'n': best_n, 'colors': best_colors}
            
    # Fallback
    for n in range(9, 1, -1):
        for seed in range(5):
            colors = _spectral_init(n, seed)
            colors = _simulated_annealing(colors, n, time_limit=2.0)
            masks = _build_masks_from_colors(colors, n)
            if _count_total_c6s(masks, n) == 0:
                best_n = n
                best_colors = colors
                break
        if best_n == n:
            break
            
    return {'n': best_n, 'colors': best_colors}
