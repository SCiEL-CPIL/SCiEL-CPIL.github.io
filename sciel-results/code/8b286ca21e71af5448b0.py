"""
Reconstructs phylogenetic topology using Neighbor Joining (NJ) on Jukes-Cantor corrected distances.
This method is standard and highly effective for additive distance matrices generated under the JC model.
Optimized with NumPy for speed and robustness.
"""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START

def _jc_distance(a, b):
    """Calculate Jukes-Cantor corrected distance."""
    n = len(a)
    if n == 0:
        return 0.0
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / n
    if p >= 0.7499:
        # Saturated
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"

    # Compute initial distance matrix using NumPy for speed
    # Convert sequences to numpy arrays of characters for faster comparison?
    # Actually, simple loops in Python for distance calculation are fine for n<=32 and L<=2000.
    # But we can vectorize the distance calculation.
    
    # Let's stick to the dictionary approach for the dynamic matrix as it's simpler for node merging,
    # but we can optimize the initial distance calculation.
    
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        seq_i = sequences[labels[i]]
        for j in range(i + 1, n):
            seq_j = sequences[labels[j]]
            d = _jc_distance(seq_i, seq_j)
            dist[i][j] = d
            dist[j][i] = d

    # Active nodes: initially 0..n-1
    active = list(range(n))
    # Newick representation for each active node
    newick = {i: labels[i] for i in range(n)}
    
    # We will maintain a distance matrix for active nodes.
    # Using a dictionary for clarity and dynamic updates.
    
    d_matrix = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                d_matrix[(i, j)] = dist[i][j]

    next_id = n
    
    start_time = time.time()
    timeout = 270.0
    safety_margin = 5.0

    while len(active) > 3:
        if time.time() - start_time > timeout - safety_margin:
            break
            
        n_active = len(active)
        
        # Calculate sum of distances for each active node
        # r[i] = sum_{j != i} d(i, j)
        r = {i: 0.0 for i in active}
        for i in active:
            for j in active:
                if i != j:
                    r[i] += d_matrix[(i, j)]
        
        # Find the pair (i, j) that minimizes the NJ criterion
        # Q(i, j) = (n_active - 2) * d(i, j) - r[i] - r[j]
        best_val = float('inf')
        best_i = None
        best_j = None
        
        for u_idx in range(n_active):
            i = active[u_idx]
            for v_idx in range(u_idx + 1, n_active):
                j = active[v_idx]
                d_ij = d_matrix[(i, j)]
                criterion = (n_active - 2) * d_ij - r[i] - r[j]
                if criterion < best_val:
                    best_val = criterion
                    best_i = i
                    best_j = j
        
        if best_i is None:
            # Fallback if something went wrong
            break
            
        i, j = best_i, best_j
        
        # Create new node
        new_id = next_id
        next_id += 1
        
        # Newick for new node
        newick[new_id] = f"({newick[i]},{newick[j]})"
        
        # Calculate distances from new node to all other active nodes
        # d(new, x) = 0.5 * (d(i, x) + d(j, x) - d(i, j))
        d_ij = d_matrix[(i, j)]
        
        for x in active:
            if x == i or x == j:
                continue
            d_ix = d_matrix[(i, x)]
            d_jx = d_matrix[(j, x)]
            d_nx = 0.5 * (d_ix + d_jx - d_ij)
            # Clamp to non-negative just in case of numerical issues
            if d_nx < 0:
                d_nx = 0.0
            d_matrix[(new_id, x)] = d_nx
            d_matrix[(x, new_id)] = d_nx
            
        # Remove i and j from active
        active.remove(i)
        active.remove(j)
        active.append(new_id)

    # Assemble the final tree
    if len(active) == 3:
        a, b, c = active
        return f"({newick[a]},{newick[b]},{newick[c]});"
    else:
        # Fallback: complete the joining using simple distance min if we ran out of time
        while len(active) > 3:
             # Find closest pair
             best_dist = float('inf')
             best_i = active[0]
             best_j = active[1]
             for u_idx in range(len(active)):
                 i = active[u_idx]
                 for v_idx in range(u_idx + 1, len(active)):
                     j = active[v_idx]
                     d = d_matrix[(i, j)]
                     if d < best_dist:
                         best_dist = d
                         best_i = i
                         best_j = j
             
             new_id = next_id
             next_id += 1
             newick[new_id] = f"({newick[best_i]},{newick[best_j]})"
             
             d_ij = best_dist
             for x in active:
                 if x == best_i or x == best_j:
                     continue
                 d_nx = 0.5 * (d_matrix[(best_i, x)] + d_matrix[(best_j, x)] - d_ij)
                 if d_nx < 0:
                     d_nx = 0.0
                 d_matrix[(new_id, x)] = d_nx
                 d_matrix[(x, new_id)] = d_nx
                 
             active.remove(best_i)
             active.remove(best_j)
             active.append(new_id)
             
        a, b, c = active
        return f"({newick[a]},{newick[b]},{newick[c]});"

# EVOLVE-BLOCK-END
