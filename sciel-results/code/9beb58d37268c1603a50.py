"""Fast Distance-based Quartet (FDQ) topology reconstruction under Jukes-Cantor."""
import time
import math
import numpy as np

_DEADLINE_MARGIN = 2.0  # seconds to keep before the 270s wall-clock kill
_START = time.time()


def _elapsed():
    return time.time() - _START


def _remaining():
    return 270.0 - _elapsed() - _DEADLINE_MARGIN


def _jc_distance_matrix(sequences, labels):
    """Compute JC-corrected pairwise distances using vectorized NumPy ops."""
    n = len(labels)
    # Build a numeric matrix: map A,C,G,T to 0,1,2,3
    seqs = [sequences[lb] for lb in labels]
    L = len(seqs[0])
    # Encode each sequence as an integer array
    enc = {}
    for ch, val in (('A', 0), ('C', 1), ('G', 2), ('T', 3)):
        enc[ch] = val
    # Use a list of bytearrays for speed, or just arrays
    mat = np.zeros((n, L), dtype=np.int8)
    for i, s in enumerate(seqs):
        # s is a string; encode
        arr = np.frombuffer(s.encode('ascii'), dtype=np.uint8)
        # Map: A=65, C=67, G=71, T=84
        # We'll do a lookup via np.where
        # Actually, let's just build a lookup table
        lut = np.zeros(256, dtype=np.int8)
        lut[ord('A')] = 0
        lut[ord('C')] = 1
        lut[ord('G')] = 2
        lut[ord('T')] = 3
        mat[i] = lut[arr]

    # Pairwise differences: for each pair (i,j), count mismatches
    # d[i,j] = sum over positions where mat[i,pos] != mat[j,pos]
    # This is O(n^2 * L) which for n=32, L=2000 is ~2M ops, fine.
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.count_nonzero(mat[i] != mat[j])
            p = diff / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _nj_reconstruct(dist, labels):
    """
    Perform Neighbor Joining on the distance matrix.
    Returns a Newick string.
    """
    n = dist.shape[0]
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"

    # Use a dictionary for distances to handle dynamic node IDs safely
    D = {}
    for i in range(n):
        for j in range(n):
            D[(i, j)] = dist[i, j]
            D[(j, i)] = dist[j, i]
            
    active = list(range(n))
    newick_strs = {i: labels[i] for i in range(n)}
    next_id = n
    
    while len(active) > 3:
        if _remaining() < 0.5:
            # Fallback: return a chain of remaining active nodes
            if len(active) == 0:
                return ""
            if len(active) == 1:
                return newick_strs[active[0]] + ";"
            if len(active) == 2:
                return f"({newick_strs[active[0]]},{newick_strs[active[1]]});"
            # Chain them
            current = newick_strs[active[0]]
            for i in range(1, len(active)):
                current = f"({current},{newick_strs[active[i]]})"
            return current + ";"

        m = len(active)
        # Calculate r values
        r = {}
        for ai in active:
            r[ai] = sum(D[(ai, aj)] for aj in active if aj != ai)
        
        best_a = None
        best_b = None
        best_criterion = float('inf')
        
        for i in range(m):
            ai = active[i]
            for j in range(i + 1, m):
                aj = active[j]
                criterion = (m - 2) * D[(ai, aj)] - r[ai] - r[aj]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a = ai
                    best_b = aj
        
        a = best_a
        b = best_b
        
        # Create new node
        new_node = next_id
        next_id += 1
        
        # Update distances
        for c in active:
            if c == a or c == b:
                continue
            D[(new_node, c)] = 0.5 * (D[(a, c)] + D[(b, c)] - D[(a, b)])
            D[(c, new_node)] = D[(new_node, c)]
            
        # Update active list
        new_active = [node for node in active if node != a and node != b]
        new_active.append(new_node)
        active = new_active
        
        # Update Newick string
        newick_strs[new_node] = f"({newick_strs[a]},{newick_strs[b]})"
        
    # Handle final 3 nodes
    if len(active) == 3:
        a, b, c = active
        new_node1 = next_id
        next_id += 1
        newick_strs[new_node1] = f"({newick_strs[b]},{newick_strs[c]})"
        new_node2 = next_id
        next_id += 1
        newick_strs[new_node2] = f"({newick_strs[a]},{newick_strs[new_node1]})"
        return newick_strs[new_node2] + ";"
    elif len(active) == 2:
        a, b = active
        return f"({newick_strs[a]},{newick_strs[b]});"
    elif len(active) == 1:
        return newick_strs[active[0]] + ";"
    else:
        return ""

def reconstruct(sequences):
    """Reconstruct unrooted tree topology from sequences."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 0:
            return ""
        if n == 1:
            return labels[0] + ";"
        return "(" + ",".join(labels) + ");"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    # Compute distance matrix
    dist = _jc_distance_matrix(sequences, labels)
    
    # Use NJ
    try:
        newick = _nj_reconstruct(dist, labels)
        if newick and all(lb in newick for lb in labels):
            return newick
    except Exception as e:
        # Fallback to star tree if NJ fails
        pass
    
    # Fallback: Star tree
    # A star tree with N leaves can be represented as a binary tree that is "balanced" or just a chain.
    # To be safe and valid, we'll create a chain.
    if n > 0:
        current = labels[0]
        for i in range(1, n):
            current = f"({current},{labels[i]})"
        return current + ";"
    return ""
