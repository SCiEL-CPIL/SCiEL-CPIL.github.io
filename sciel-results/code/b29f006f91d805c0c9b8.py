import numpy as np
import math
import time

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Convert sequences to numpy arrays of integers for fast comparison
    # A=0, C=1, G=2, T=3
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([list(map(lambda x: seq_map[x], s)) for s in sequences.values()], dtype=np.int8)
    # seqs shape: (n, L)
    
    # Compute pairwise Hamming distances
    # Use broadcasting: seqs[:, None, :] vs seqs[None, :, :]
    # This creates an (n, n, L) array, which might be large for L=2000, n=32.
    # 32 * 32 * 2000 = 2M elements. 2M * 1 byte = 2MB. This is fine.
    # But comparing element-wise might be slow in pure numpy without optimization.
    # Alternative: loop over sites or use dot products if we encode properly.
    # Let's stick to a loop over sites or a vectorized comparison.
    
    # Vectorized comparison:
    # diff = (seqs[:, None, :] != seqs[None, :, :]).sum(axis=2)
    # This is efficient enough for n=32, L=2000.
    
    diff = (seqs[:, None, :] != seqs[None, :, :]).sum(axis=2)
    p = diff / diff.shape[1]
    
    # JC Correction
    # Handle p >= 0.75
    p_safe = np.clip(p, 0, 0.7499)
    D = -0.75 * np.log(1.0 - 4.0 * p_safe / 3.0)
    np.fill_diagonal(D, 0)
    
    # Neighbour Joining
    # nodes: list of current nodes (indices or strings)
    # tree: dict mapping node -> newick string
    # D: current distance matrix
    
    # We will maintain D as a numpy array and update it.
    # Since n is small, we can also use a list of lists, but numpy is faster.
    # However, updating a subset of rows/cols in numpy is easy.
    
    nodes = list(range(n))
    tree = {i: labels[i] for i in range(n)}
    D_curr = D.copy()
    
    # Mapping from original index to current "active" index in the reduced matrix?
    # It's easier to just track which original nodes are still active.
    # Actually, NJ creates new nodes. We need to map new nodes to distances.
    # Let's use a dictionary for distances or a sparse representation?
    # With n<=32, a full matrix is fine.
    
    # Let's use a list of active node ids.
    # And a dictionary `dist` where keys are tuples of node ids?
    # Or just keep a matrix and track active indices.
    
    # Let's use a standard NJ implementation with a distance dictionary for clarity and correctness, 
    # as n is small.
    
    # dist[(i, j)] = distance
    dist = {}
    for i in range(n):
        for j in range(i+1, n):
            dist[(i, j)] = D_curr[i, j]
            dist[(j, i)] = D_curr[i, j]
            
    active_nodes = list(range(n))
    
    while len(active_nodes) > 3:
        n_curr = len(active_nodes)
        
        # Calculate row sums
        row_sums = {}
        for i in active_nodes:
            s = 0.0
            for j in active_nodes:
                if i != j:
                    s += dist[(i, j)]
            row_sums[i] = s
            
        # Find min Q
        min_q = float('inf')
        min_pair = None
        for i_idx, i in enumerate(active_nodes):
            for j in active_nodes[i_idx+1:]:
                q = (n_curr - 2) * dist[(i, j)] - row_sums[i] - row_sums[j]
                if q < min_q:
                    min_q = q
                    min_pair = (i, j)
                    
        a, b = min_pair
        
        # Create new node c
        c = -1 # Use negative indices for new nodes to avoid collision
        # Actually, let's just use a counter or string.
        # Let's use a new integer id.
        new_id = -len(active_nodes) - 1 
        # Ensure uniqueness.
        # Simpler: just use a string id for the new node.
        new_node = f"n_{a}_{b}"
        
        # Update tree
        tree[new_node] = f"({tree[a]},{tree[b]})"
        
        # Update distances
        # D(c, x) = 0.5 * (D(a, x) + D(b, x) - D(a, b))
        d_ab = dist[(a, b)]
        for x in active_nodes:
            if x not in (a, b):
                d_cx = 0.5 * (dist[(a, x)] + dist[(b, x)] - d_ab)
                dist[(new_node, x)] = d_cx
                dist[(x, new_node)] = d_cx
        
        # Remove a, b, add new_node
        active_nodes.remove(a)
        active_nodes.remove(b)
        active_nodes.append(new_node)
        
    # Connect remaining 3 nodes
    # Any topology of 3 nodes is the same unrooted tree (a star or a triangle, which are topologically equivalent for 3 taxa)
    # Wait, for 3 taxa, the unrooted tree is just a star.
    # The Newick string for 3 taxa (A, B, C) is ((A,B),C) or (A,(B,C)) etc.
    # All are topologically equivalent for 3 leaves.
    # We just need to output a valid Newick string.
    
    a, b, c = active_nodes
    return f"({tree[a]},{tree[b]},{tree[c]});"
