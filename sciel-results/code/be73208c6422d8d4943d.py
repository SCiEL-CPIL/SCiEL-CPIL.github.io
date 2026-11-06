"""Phylogenetic topology recovery from simulated sequences."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE_MARGIN_S = 10.0  # safety margin before the 270s wall-clock kill


def _jc_distance_matrix(sequences, labels):
    """Compute the Jukes-Cantor corrected pairwise distance matrix using NumPy."""
    n = len(labels)
    # Convert sequences to integer arrays for fast comparison
    # A=0, C=1, G=2, T=3
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    n_sites = len(sequences[labels[0]])
    # Build a 2D array of shape (n, n_sites)
    arr = np.empty((n, n_sites), dtype=np.uint8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        # Fast conversion
        arr[i] = np.frombuffer(s.encode('ascii'), dtype=np.uint8)
        # Map ASCII to 0-3: A=65, C=67, G=71, T=84
        # We'll just use direct comparison which is fine
        pass
    
    # Actually, let's do it more simply with direct numpy operations on bytes
    # Rebuild properly
    arr = np.zeros((n, n_sites), dtype=np.uint8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        # Convert string to array of ASCII values
        arr[i] = np.array([ord(c) for c in s], dtype=np.uint8)
    
    # Compute pairwise Hamming distances
    # For each pair (i, j), count positions where arr[i] != arr[j]
    # This is O(n^2 * L) but vectorized
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        # Compare row i with all other rows
        # arr[i] has shape (L,), arr[j] has shape (L,)
        # We want count of arr[i] != arr[j] for all j
        # (arr[i][None, :] != arr) has shape (n, L)
        diff = (arr[i][None, :] != arr).sum(axis=1)
        p = diff / n_sites
        # Jukes-Cantor correction: d = -3/4 * ln(1 - 4p/3)
        # Handle p >= 0.75
        valid = p < 0.75
        D[i, valid] = -0.75 * np.log(1.0 - 4.0 * p[valid] / 3.0)
        D[i, ~valid] = 3.0  # large value for saturated distances
    D = (D + D.T) * 0.5  # symmetrize
    np.fill_diagonal(D, 0.0)
    return D


def _neighbor_joining(D, labels):
    """
    Perform Neighbor Joining on distance matrix D.
    Returns a Newick string.
    """
    n = len(labels)
    if n <= 1:
        return labels[0] + ";" if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    # Work with a copy of D
    D = D.copy()
    # Track which original labels are still active
    active = list(range(n))
    # Track the tree structure for each active node
    # Each node has a label (either original or a composite)
    node_labels = {i: labels[i] for i in range(n)}
    
    while len(active) > 3:
        m = len(active)
        # Compute r_i = sum of D[i][j] for all active j
        r = np.array([D[active[i], active[j]].sum() for i in range(m)])
        
        best_criterion = float('inf')
        best_i, best_j = -1, -1
        
        for i in range(m):
            for j in range(i + 1, m):
                a, b = active[i], active[j]
                criterion = (m - 2) * D[a, b] - r[i] - r[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i, best_j = i, j
        
        a, b = active[best_i], active[best_j]
        
        # Compute branch lengths
        # delta = D[a,b]/2
        delta = D[a, b] / 2.0
        # For each other active node x, compute:
        # len_a = delta + (D[a,x] - D[b,x]) / 2
        # len_b = delta - (D[a,x] - D[b,x]) / 2
        # Use the average over all x for stability, or just pick one
        # Standard NJ uses the formula with the specific x, but we need a single value
        # Actually, the standard formula is:
        # len(a) = D[a,b]/2 + (1/(2*(m-2))) * (sum_x D[a,x] - sum_x D[b,x])
        # Let's use that
        sum_a = r[best_i]  # sum of D[a, x] for all active x
        sum_b = r[best_j]  # sum of D[b, x] for all active x
        len_a = D[a, b] / 2.0 + (sum_a - sum_b) / (2.0 * (m - 2))
        len_b = D[a, b] / 2.0 - (sum_a - sum_b) / (2.0 * (m - 2))
        
        # Create new node
        new_node = n + len([k for k in node_labels if k >= n])  # unique id
        # Build the subtree string for the new node
        # We'll store the subtree structure
        subtree_a = node_labels[a]
        subtree_b = node_labels[b]
        new_label = f"({subtree_a}:{len_a:.6f},{subtree_b}:{len_b:.6f})"
        
        # Update distance matrix for the new node
        # D[new, x] = (D[a,x] + D[b,x] - D[a,b]) / 2
        for x in active:
            if x != a and x != b:
                D[new_node, x] = (D[a, x] + D[b, x] - D[a, b]) / 2.0
                D[x, new_node] = D[new_node, x]
        
        node_labels[new_node] = new_label
        
        # Remove a and b from active, add new_node
        active = [x for x in active if x != a and x != b] + [new_node]
    
    # Now we have 3 nodes left
    # Form a trifurcation or a resolved tree
    # For unrooted topology, we just need the topology
    # Take the three remaining nodes and form a tree
    remaining = active
    if len(remaining) == 3:
        # Create a tree with one internal node connecting all three
        # For Newick, we can do (A,(B,C)) or similar
        # The topology is determined by how we group them
        # In NJ, the last three are joined as a star or resolved
        # Let's resolve as (A,(B,C)) for a fixed ordering
        a, b, c = remaining
        # Compute approximate branch lengths
        # This is a bit tricky for the last step
        # For topology recovery, we just need the correct grouping
        # The standard NJ resolves the last three by joining two first
        # Let's join b and c, then join a with that
        # D between the three
        d_ab = D[a, b]
        d_ac = D[a, c]
        d_bc = D[b, c]
        
        # Join the pair with smallest distance first? No, NJ already did that.
        # The last three in NJ form a resolved tree.
        # Let's just create a balanced representation
        new_label = f"({node_labels[a]},{node_labels[b]},{node_labels[c]})"
        return new_label + ";"
    
    return "(" + ",".join(node_labels[x] for x in active) + ");"


def _nj_topology_only(D, labels):
    """
    Perform Neighbor Joining but only track topology (no branch lengths in output).
    This is faster and cleaner for topology recovery.
    """
    n = len(labels)
    if n <= 1:
        return labels[0] + ";" if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    D = D.copy()
    active = list(range(n))
    node_labels = {i: labels[i] for i in range(n)}
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        r = np.array([D[active[i], active[j]].sum() for i in range(m)])
        
        best_criterion = float('inf')
        best_i, best_j = -1, -1
        
        for i in range(m):
            for j in range(i + 1, m):
                a, b = active[i], active[j]
                criterion = (m - 2) * D[a, b] - r[i] - r[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i, best_j = i, j
        
        a, b = active[best_i], active[best_j]
        
        new_id_val = next_id
        next_id += 1
        
        # New node label (topology only, no branch lengths)
        new_label = f"({node_labels[a]},{node_labels[b]})"
        
        # Update distances
        for x in active:
            if x != a and x != b:
                D[new_id_val, x] = (D[a, x] + D[b, x] - D[a, b]) / 2.0
                D[x, new_id_val] = D[new_id_val, x]
        
        node_labels[new_id_val] = new_label
        active = [x for x in active if x != a and x != b] + [new_id_val]
    
    # Last three nodes
    remaining = active
    if len(remaining) == 3:
        a, b, c = remaining
        # Resolve as a resolved tree: join two, then join with third
        # To determine which two to join, use distances
        d_ab = D[a, b]
        d_ac = D[a, c]
        d_bc = D[b, c]
        
        # Join the pair with smallest distance
        if d_ab <= d_ac and d_ab <= d_bc:
            pair = (a, b)
            other = c
        elif d_ac <= d_ab and d_ac <= d_bc:
            pair = (a, c)
            other = b
        else:
            pair = (b, c)
            other = a
        
        p1, p2 = pair
        new_label = f"({node_labels[p1]},{node_labels[p2]})"
        final_label = f"({node_labels[other]},{new_label})"
        return final_label + ";"
    
    return "(" + ",".join(node_labels[x] for x in active) + ");"


def _fastmedian_junction(D, labels):
    """
    Fastmedian-style approach: use NJ but with a different objective.
    For now, this is a placeholder that just calls NJ.
    """
    return _nj_topology_only(D, labels)


def _bootstrap_support(D, labels, n_bootstraps=50, rng=None):
    """
    Perform bootstrap on the distance matrix to assess topology support.
    Returns the most common topology.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    n = len(labels)
    n_sites = D.shape[0]  # This is wrong, D is n x n, not n_sites x n
    
    # We need the original sequences to bootstrap
    # This function is not complete without sequences
    # Let's skip bootstrap for now and just use NJ
    return None


def reconstruct(sequences):
    """
    Reconstruct the unrooted tree topology from sequences.
    Uses Neighbor Joining on Jukes-Cantor corrected distances.
    """
    start_time = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return (labels[0] + ";") if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Compute distance matrix
    D = _jc_distance_matrix(sequences, labels)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > _DEADLINE_MARGIN_S:
        # Fallback: return a star tree
        return "(" + ",".join(labels) + ");"
    
    # Perform NJ
    newick = _nj_topology_only(D, labels)
    
    return newick
# EVOLVE-BLOCK-END
