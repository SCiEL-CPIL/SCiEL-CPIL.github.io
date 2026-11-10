"""Phylogenetic topology recovery from Jukes-Cantor simulated sequences."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 250.0  # seconds wall-clock; evaluator kills at 270s


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distance matrix using vectorized numpy."""
    n = len(labels)
    seq_len = len(sequences[labels[0]])
    
    # Encode sequences: A=0, C=1, G=2, T=3
    # ASCII: A=65, C=67, G=71, T=84
    enc = np.zeros((n, seq_len), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        arr = np.frombuffer(s.encode('ascii'), dtype=np.int8)
        enc[i] = arr
    
    # Vectorized pairwise difference counting
    diff_matrix = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        diffs = np.count_nonzero(enc[i] != enc, axis=1)
        diff_matrix[i] = diffs
    
    # Convert to proportions and apply JC correction
    p = diff_matrix.astype(np.float64) / seq_len
    p_clipped = np.minimum(p, 0.749999)
    D = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    D[p >= 0.75] = 3.0
    D = (D + D.T) * 0.5
    np.fill_diagonal(D, 0.0)
    return D


def _neighbor_joining_soft(D, labels, deadline=None, temperature=0.5):
    """
    Perform neighbor joining with soft-min selection on the Q matrix.
    Uses a temperature parameter to create a smooth surrogate for the min operation.
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    D = D.astype(np.float64).copy()
    node_names = list(labels)
    
    while len(node_names) > 3:
        if deadline is not None and time.time() > deadline:
            return "(" + ",".join(node_names) + ")"
        
        m = len(node_names)
        r = D.sum(axis=1)
        
        # Q matrix: Q[i,j] = (m-2)*D[i,j] - r[i] - r[j]
        Q = (m - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        # Soft-min selection: use temperature to create a smooth selection
        # Lower temperature = more confident in the minimum
        # We use a log-sum-exp based soft min
        Q_sym = np.minimum(Q, Q.T)  # Ensure symmetry
        Q_lower = np.tril(Q_sym, -1)  # Only lower triangle
        
        # For soft min, we want to weight pairs by exp(-Q/temperature)
        # But to avoid numerical issues, subtract the min first
        Q_min = np.min(Q_lower)
        Q_shifted = Q_lower - Q_min
        weights = np.exp(-Q_shifted / temperature)
        
        # Normalize to get probabilities
        total_weight = np.sum(weights)
        if total_weight > 0:
            probs = weights / total_weight
        else:
            probs = weights
        
        # Find the pair with highest probability (soft minimum of Q)
        flat_idx = np.argmax(probs)
        i, j = np.unravel_index(flat_idx, probs.shape)
        if i > j:
            i, j = j, i
        
        # Create new node
        new_name = f"({node_names[i]},{node_names[j]})"

        # Compute new distances
        new_dist = 0.5 * (D[i, :] + D[j, :] - D[i, j])

        # Build new distance matrix
        keep = [k for k in range(m) if k != i and k != j]
        keep_arr = np.array(keep, dtype=np.intp)
        new_m = m - 1
        
        new_D = np.empty((new_m, new_m), dtype=np.float64)
        new_D[:-1, :-1] = D[np.ix_(keep_arr, keep_arr)]
        
        new_node_idx = m - 2
        new_D[:-1, new_node_idx] = new_dist[keep_arr]
        new_D[new_node_idx, :-1] = new_dist[keep_arr]
        new_D[new_node_idx, new_node_idx] = 0.0

        new_names = [node_names[k] for k in keep] + [new_name]
        D = new_D
        node_names = new_names

    return f"({node_names[0]},{node_names[1]},{node_names[2]})"


def _neighbor_joining_standard(D, labels, deadline=None):
    """
    Standard neighbor joining (greedy min of Q matrix).
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    D = D.astype(np.float64).copy()
    node_names = list(labels)
    
    while len(node_names) > 3:
        if deadline is not None and time.time() > deadline:
            return "(" + ",".join(node_names) + ")"
        
        m = len(node_names)
        r = D.sum(axis=1)
        Q = (m - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = idx
        if i > j:
            i, j = j, i

        new_name = f"({node_names[i]},{node_names[j]})"
        new_dist = 0.5 * (D[i, :] + D[j, :] - D[i, j])

        keep = [k for k in range(m) if k != i and k != j]
        keep_arr = np.array(keep, dtype=np.intp)
        new_m = m - 1
        
        new_D = np.empty((new_m, new_m), dtype=np.float64)
        new_D[:-1, :-1] = D[np.ix_(keep_arr, keep_arr)]
        
        new_node_idx = m - 2
        new_D[:-1, new_node_idx] = new_dist[keep_arr]
        new_D[new_node_idx, :-1] = new_dist[keep_arr]
        new_D[new_node_idx, new_node_idx] = 0.0

        new_names = [node_names[k] for k in keep] + [new_name]
        D = new_D
        node_names = new_names

    return f"({node_names[0]},{node_names[1]},{node_names[2]})"


def _nj_with_deadline(sequences, labels, deadline):
    """Run NJ with deadline guard. Returns Newick string or None if timeout."""
    start = time.time()
    D = _compute_distance_matrix(sequences, labels)
    elapsed = time.time() - start
    if elapsed > deadline - 5:
        return None
    
    # Use standard NJ (greedy) as it's more reliable for topology recovery
    # The soft-min approach can be too aggressive in its smoothing
    return _neighbor_joining_standard(D, labels, deadline=start + deadline - 2)


def reconstruct(sequences):
    """
    Reconstruct unrooted phylogenetic tree topology.
    Returns Newick string over labels t0..t(n-1).
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 3:
        if n == 1:
            return labels[0]
        elif n == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return f"({labels[0]},{labels[1]},{labels[2]})"

    # Try neighbor joining
    result = _nj_with_deadline(sequences, labels, deadline - start_time)
    if result is not None:
        return result + ";"

    # Fallback: star tree
    return "(" + ",".join(labels) + ");"
# EVOLVE-BLOCK-END
