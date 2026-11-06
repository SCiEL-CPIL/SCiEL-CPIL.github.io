"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Neighbour joining on Jukes-Cantor corrected distances (Saitou and Nei 1987).'''
import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 265.0  # safety margin before 270s kill

def _elapsed():
    return time.time() - _START_TIME

def _build_distance_matrix(sequences, labels):
    """Build pairwise JC distance matrix using numpy for speed."""
    n = len(labels)
    if n <= 1:
        return {}
        
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # Convert sequences to numpy array of ints
    # Handle potential whitespace or case issues if any, though task says ACGT
    seqs = np.array([list(map(lambda x: seq_map.get(x.upper(), 0), s)) for s in sequences.values()], dtype=np.int8)
    
    # Compute pairwise Hamming distances vectorized
    # diff[i, j] is the number of differing sites between seq i and seq j
    diff = (seqs[:, None, :] != seqs[None, :, :]).sum(axis=2)
    L = diff.shape[1]
    
    # Avoid division by zero if L is 0 (unlikely but safe)
    if L == 0:
        p = np.zeros_like(diff, dtype=float)
    else:
        p = diff / L
        
    # JC Correction
    # p >= 0.75 leads to log(0) or negative argument. Clip to 0.7499.
    p_safe = np.clip(p, 0, 0.7499)
    D = -0.75 * np.log(1.0 - 4.0 * p_safe / 3.0)
    np.fill_diagonal(D, 0)
    
    # Map back to dictionary for NJ
    dist = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[(labels[i], labels[j])] = float(D[i, j])
    return dist

def _neighbor_joining(dist, labels):
    """
    Perform Neighbor Joining. Returns a Newick string.
    """
    nodes = list(labels)
    trees = {name: name for name in labels}
    matrix = dist
    
    while len(nodes) > 3:
        if _elapsed() > _DEADLINE:
            # If we run out of time, return a star tree or whatever we have so far?
            # Better to return a valid star tree to avoid crashes, though NJ should be fast.
            break
        
        n = len(nodes)
        # Calculate row sums
        total = {}
        for a in nodes:
            total[a] = 0.0
            for b in nodes:
                if a != b:
                    total[a] += matrix.get((a, b), 0.0)
        
        best_criterion = float('inf')
        best_pair = None
        
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                # Ensure key exists
                d_ab = matrix.get((a, b), 0.0)
                criterion = (n - 2) * d_ab - total[a] - total[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_pair = (a, b)
        
        if best_pair is None:
            break
            
        a, b = best_pair
        d_ab = matrix.get((a, b), 0.0)
        
        joined = f"({trees[a]},{trees[b]})"
        new_node = f"n{len(nodes)}_{a}_{b}"
        
        rest = [x for x in nodes if x not in (a, b)]
        
        # Update distances
        for x in rest:
            d_ax = matrix.get((a, x), 0.0)
            d_bx = matrix.get((b, x), 0.0)
            d_cx = 0.5 * (d_ax + d_bx - d_ab)
            matrix[(new_node, x)] = d_cx
            matrix[(x, new_node)] = d_cx
            
        nodes = rest + [new_node]
        trees[new_node] = joined
    
    # Construct final Newick string
    # If we broke out early due to deadline, nodes might be > 3.
    # For 3 nodes: (A, B, C) is a star, which is correct for unrooted 3-taxon tree.
    # For > 3 nodes (timeout case), we need to make a star or connect them arbitrarily.
    # A star tree is (A, B, C, D, ...)
    if len(nodes) == 3:
        return f"({trees[nodes[0]]},{trees[nodes[1]]},{trees[nodes[2]]});"
    elif len(nodes) == 2:
        return f"({trees[nodes[0]]},{trees[nodes[1]]});"
    elif len(nodes) == 1:
        return f"{trees[nodes[0]]};"
    else:
        # Fallback: star tree
        return "(" + ",".join(trees[x] for x in nodes) + ");"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Build distance matrix
    dist = _build_distance_matrix(sequences, labels)
    
    # Run Neighbor Joining
    result = _neighbor_joining(dist, labels)
    
    return result
# EVOLVE-BLOCK-END
