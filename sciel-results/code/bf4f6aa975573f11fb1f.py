"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Robust Neighbor Joining on Jukes-Cantor corrected distances.'''
import math
import time
import numpy as np


def _jc_distance_matrix(aln_matrix):
    """Compute JC-corrected distance matrix from a binary alignment matrix."""
    n_taxa = aln_matrix.shape[0]
    n_sites = aln_matrix.shape[1]
    a = aln_matrix[:, None, :]
    b = aln_matrix[None, :, :]
    diff = (a != b).sum(axis=2)
    p = diff.astype(np.float64) / n_sites
    np.fill_diagonal(p, 0.0)
    p_clipped = np.clip(p, 0.0, 0.7499)
    arg = 1.0 - (4.0 / 3.0) * p_clipped
    dist = -0.75 * np.log(arg)
    np.fill_diagonal(dist, 0.0)
    return dist


def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 260.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    aln_list = []
    for lab in labels:
        seq = sequences[lab]
        aln_list.append([base_map[c] for c in seq])
    aln_matrix = np.array(aln_list, dtype=np.int8)
    
    # Initial distance matrix for original taxa
    D_init = _jc_distance_matrix(aln_matrix)
    
    # We will maintain a global distance matrix of size (2n) x (2n) to handle new internal nodes
    # Max nodes: n leaves + (n-3) internal = 2n-3. So size 2n is safe.
    max_nodes = 2 * n
    D = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    D[:n, :n] = D_init
    
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    next_id = n
    
    while len(active) > 3:
        if time.time() > deadline - 5.0:
            break
        
        m = len(active)
        # Extract submatrix for active nodes
        sub_D = D[np.ix_(active, active)]
        
        # Row sums
        row_sums = sub_D.sum(axis=1)
        
        # NJ criterion
        crit = (m - 2) * sub_D - row_sums[:, None] - row_sums[None, :]
        np.fill_diagonal(crit, np.inf)
        
        idx = np.unravel_index(crit.argmin(), crit.shape)
        i, j = idx
        
        a = active[i]
        b = active[j]
        
        # Build new internal node
        joined = f"({trees[a]},{trees[b]})"
        new_node = next_id
        next_id += 1
        
        # Update distances
        # d(new, x) = 0.5 * (d(a, x) + d(b, x) - d(a, b))
        d_ab = D[a, b]
        
        # Calculate new distances to all other active nodes
        for k, x in enumerate(active):
            if x == a or x == b:
                continue
            D[new_node, x] = 0.5 * (D[a, x] + D[b, x] - d_ab)
            D[x, new_node] = D[new_node, x]
        
        # Update active list: remove a, b, add new_node
        active = [x for x in active if x not in (a, b)]
        active.append(new_node)
        trees[new_node] = joined
    
    # Build final Newick
    if len(active) == 3:
        a, b, c = active
        # The tree for 3 leaves is ((a,b),c) or similar, but NJ usually returns a trichotomy or resolves it.
        # Since we stopped at 3, we can just connect them arbitrarily or use the last join logic.
        # Actually, standard NJ stops at 3 and the last 3 form a star or a resolved triplet.
        # For unrooted topology, any of the 3 possible triplets is a guess if not resolved.
        # However, the last join in NJ (when 4 nodes remain) resolves the triplet.
        # If we break early or n=3, we need to handle it.
        # If we reached 3 nodes via the loop, the last operation joined two nodes into one, leaving 3.
        # The 'trees' dict contains the Newick strings for these 3 nodes.
        # We need to connect them. A common way is to just make a star: (a,b,c)
        # But wait, if we had 4 nodes and joined 2, we have 3 nodes left.
        # The newick strings for the 3 nodes are ready.
        # We just need to combine them.
        return f"({trees[a]},{trees[b]},{trees[c]});"
    elif len(active) == 2:
        return f"({trees[active[0]]},{trees[active[1]]});"
    elif len(active) == 1:
        return trees[active[0]] + ";"
    else:
        # Emergency fallback: star tree
        parts = [trees[x] for x in active]
        return "(" + ",".join(parts) + ");"

# EVOLVE-BLOCK-END
