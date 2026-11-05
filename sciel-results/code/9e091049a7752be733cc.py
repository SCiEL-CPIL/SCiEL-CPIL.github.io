import numpy as np
import math
import time

def _jc_distance_matrix(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n == 0:
        return np.array([]), labels
    if n == 1:
        return np.array([[0.0]]), labels
        
    # Convert sequences to numpy array of chars
    # Using a list of lists is fine for small n
    seqs = [sequences[label] for label in labels]
    L = len(seqs[0])
    
    # Calculate mismatches
    # To save memory, we can do it in a loop if L is large, but N is small.
    # N=32, L=2000.
    # N^2 * L = 32*32*2000 = 2M operations.
    # Python loop might be slow.
    # Let's use numpy broadcasting.
    
    # Convert to array of integers for faster comparison?
    # Or just use string comparison?
    # np.array(seqs) creates a (n, L) array of strings/chars.
    
    seq_arr = np.array(seqs, dtype='U1')
    
    # (n, 1, L) vs (1, n, L)
    # This creates a (n, n, L) array.
    # 32 * 32 * 2000 = 2,048,000 elements.
    # Each element is 1 byte (if we use uint8) or 4 bytes (if 'U1').
    # 'U1' is 4 bytes usually. 8MB. Fine.
    
    mismatches = (seq_arr[:, None, :] != seq_arr[None, :, :]).sum(axis=2)
    p = mismatches / L
    
    d = np.zeros((n, n))
    mask = p < 0.75
    d[mask] = -0.75 * np.log(1.0 - 4.0 * p[mask] / 3.0)
    d[~mask] = 10.0
    np.fill_diagonal(d, 0.0)
    
    return d, labels

def _neighbor_joining(dist, labels):
    n = len(labels)
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    
    # Initialize distance dictionary
    dist_dict = {}
    for i in range(n):
        for j in range(i+1, n):
            dist_dict[(i, j)] = dist[i, j]
            dist_dict[(j, i)] = dist[i, j]
            
    while len(active) > 3:
        m = len(active)
        
        # Calculate r
        r = {i: 0.0 for i in active}
        for i in active:
            for j in active:
                if i != j:
                    r[i] += dist_dict[(i, j)]
        
        # Find min Q
        best_q = float('inf')
        best_i = -1
        best_j = -1
        
        for idx_i, i in enumerate(active):
            for j in active[idx_i+1:]:
                q = (m - 2) * dist_dict[(i, j)] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i = i
                    best_j = j
        
        # Join
        new_id = f"n{len(active)}_{best_i}_{best_j}"
        new_tree = f"({trees[best_i]},{trees[best_j]})"
        
        # Update distances
        for x in active:
            if x not in (best_i, best_j):
                d_new_x = 0.5 * (dist_dict[(best_i, x)] + dist_dict[(best_j, x)] - dist_dict[(best_i, best_j)])
                dist_dict[(new_id, x)] = d_new_x
                dist_dict[(x, new_id)] = d_new_x
        
        # Update active and trees
        active = [x for x in active if x not in (best_i, best_j)]
        active.append(new_id)
        trees[new_id] = new_tree
        
    if len(active) == 3:
        a, b, c = active
        return f"({trees[a]},({trees[b]},{trees[c]}));"
    return "(" + ",".join(trees[x] for x in active) + ");"

def reconstruct(sequences):
    start_time = time.time()
    
    d, labels = _jc_distance_matrix(sequences)
    
    tree = _neighbor_joining(d, labels)
    
    # Simple sanity check
    if time.time() - start_time > 260:
         # Should not happen for this algorithm
         pass
         
    return tree
