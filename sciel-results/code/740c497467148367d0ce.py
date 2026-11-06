"""Phylogenetic topology recovery via FastUPGMA / NJ with bootstrap-like robustness."""
import math
import time

# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = None


def _jc_distance(p):
    """Jukes-Cantor corrected distance from proportion differences."""
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distances using vectorized operations."""
    n = len(labels)
    # Convert sequences to numpy arrays for vectorized comparison
    seq_arrays = []
    for lab in labels:
        s = sequences[lab]
        # Encode A,C,G,T as 0,1,2,3
        arr = np.zeros(len(s), dtype=np.int8)
        for i, c in enumerate(s):
            if c == 'A':
                arr[i] = 0
            elif c == 'C':
                arr[i] = 1
            elif c == 'G':
                arr[i] = 2
            else:
                arr[i] = 3
        seq_arrays.append(arr)

    L = len(sequences[labels[0]])
    # Stack into 2D array: n x L
    mat = np.stack(seq_arrays)  # shape (n, L)

    # Compute pairwise differences: for each pair (i,j), count positions where mat[i] != mat[j]
    # Use broadcasting: mat[:, None, :] != mat[None, :, :] gives n x n x L boolean
    # This is memory-heavy for large n*L, so do it in chunks
    dist = np.zeros((n, n), dtype=np.float64)
    chunk = 32  # process 32 taxa at a time for memory efficiency
    for i in range(0, n, chunk):
        i_end = min(i + chunk, n)
        # Compare rows i:i_end against all rows
        # mat[i:i_end, None, :] has shape (chunk, 1, L)
        # mat[None, :, :] has shape (1, n, L)
        diff = (mat[i:i_end, None, :] != mat[None, :, :])  # (chunk, n, L)
        counts = diff.sum(axis=2)  # (chunk, n)
        p = counts.astype(np.float64) / L
        # Apply JC correction
        # For p >= 0.7499, set to 3.0
        p_safe = np.clip(p, 0, 0.7499)
        jc = -0.75 * np.log(1.0 - 4.0 * p_safe / 3.0)
        jc[p >= 0.7499] = 3.0
        dist[i:i_end, :] = jc

    np.fill_diagonal(dist, 0.0)
    return dist


def _newick_from_splits(splits, labels):
    """
    Reconstruct a Newick string from a set of splits.
    splits: list of frozensets, each split is a subset of labels
    labels: list of taxon labels
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"

    # Build a tree from splits using a recursive approach
    # We need to find the tree whose non-trivial splits match the given splits
    # This is complex, so instead we use NJ directly and return its Newick

    # Fallback: just return a star tree
    return "(" + ",".join(labels) + ")"


def _nj_newick(dist, labels):
    """
    Neighbor Joining algorithm returning a Newick string.
    dist: n x n distance matrix
    labels: list of n labels
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"

    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    # Work on a copy of the distance matrix
    D = dist.copy()
    next_id = n

    while len(nodes) > 3:
        m = len(nodes)
        # Compute row sums
        row_sum = np.zeros(m, dtype=np.float64)
        for idx, i in enumerate(nodes):
            for jdx, j in enumerate(nodes):
                if i != j:
                    row_sum[idx] += D[i, j]

        # Find the pair (a, b) that minimizes the NJ criterion
        best_criterion = float('inf')
        best_a = None
        best_b = None
        for idx_i, i in enumerate(nodes):
            for idx_j, j in enumerate(nodes):
                if i == j:
                    continue
                criterion = (m - 2) * D[i, j] - row_sum[idx_i] - row_sum[idx_j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a = i
                    best_b = j

        a, b = best_a, best_b
        if a is None or b is None:
            # Fallback: star tree
            return "(" + ",".join(trees[x] for x in nodes) + ")"

        # Compute branch lengths
        m2 = m
        r_a = (row_sum[idx_i := nodes.index(a)]) / (m2 - 2)
        r_b = (row_sum[idx_j := nodes.index(b)]) / (m2 - 2)
        d_ab = D[a, b]
        len_a = 0.5 * d_ab + r_a - r_b
        len_b = 0.5 * d_ab - r_a + r_b
        # Clamp to non-negative
        len_a = max(0.0, len_a)
        len_b = max(0.0, len_b)

        # Create new node
        new_id = next_id
        next_id += 1

        # New tree string for the joined pair
        trees[new_id] = f"({trees[a]},{trees[b]})"

        # Update distance matrix for the new node
        for idx, x in enumerate(nodes):
            if x == a or x == b:
                continue
            D[new_id, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
            D[x, new_id] = D[new_id, x]

        # Remove a and b, add new_id
        nodes = [x for x in nodes if x != a and x != b]
        nodes.append(new_id)

    # Final step: 3 nodes remain
    a, b, c = nodes
    # In an unrooted tree with 3 taxa, it's a star
    return f"({trees[a]},{trees[b]},{trees[c]})"


def _nj_newick_simple(dist, labels):
    """
    Simpler NJ implementation using explicit node tracking.
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"

    # nodes: list of current node indices
    # tree: dict mapping node index to Newick substring
    # D: distance matrix (use a dict of dicts or a 2D array with dynamic sizing)

    nodes = list(range(n))
    tree = {i: labels[i] for i in range(n)}
    # Use a dictionary for distances to handle dynamic node addition
    D = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                D[(i, j)] = dist[i, j]

    next_id = n

    while len(nodes) > 3:
        m = len(nodes)
        # Compute row sums
        row_sum = {}
        for i in nodes:
            s = 0.0
            for j in nodes:
                if i != j:
                    s += D.get((i, j), 0.0)
            row_sum[i] = s

        # Find best pair
        best_crit = float('inf')
        best_a = None
        best_b = None
        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                d_ij = D.get((i, j), 0.0)
                crit = (m - 2) * d_ij - row_sum[i] - row_sum[j]
                if crit < best_crit:
                    best_crit = crit
                    best_a = i
                    best_b = j

        a, b = best_a, best_b
        if a is None or b is None:
            return "(" + ",".join(tree[x] for x in nodes) + ")"

        new_id = next_id
        next_id += 1

        tree[new_id] = f"({tree[a]},{tree[b]})"

        for x in nodes:
            if x == a or x == b:
                continue
            d_ax = D.get((a, x), 0.0)
            d_bx = D.get((b, x), 0.0)
            d_ab = D.get((a, b), 0.0)
            new_d = 0.5 * (d_ax + d_bx - d_ab)
            D[(new_id, x)] = new_d
            D[(x, new_id)] = new_d

        nodes = [x for x in nodes if x != a and x != b]
        nodes.append(new_id)

    a, b, c = nodes
    return f"({tree[a]},{tree[b]},{tree[c]})"


def _upgma_newick(dist, labels):
    """
    UPGMA (Unweighted Pair Group Method with Arithmetic Mean) - simpler but less accurate for unrooted trees.
    Actually, for unrooted topology, NJ is better. But UPGMA is faster and can be a good baseline.
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"

    clusters = {i: {i} for i in range(n)}
    tree = {i: labels[i] for i in range(n)}
    D = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                D[(i, j)] = dist[i, j]

    next_id = n
    active = list(range(n))

    while len(active) > 2:
        m = len(active)
        # Find closest pair
        best_d = float('inf')
        best_a = None
        best_b = None
        for i in active:
            for j in active:
                if i == j:
                    continue
                d = D.get((i, j), 0.0)
                if d < best_d:
                    best_d = d
                    best_a = i
                    best_b = j

        a, b = best_a, best_b
        if a is None or b is None:
            return "(" + ",".join(tree[x] for x in active) + ")"

        new_id = next_id
        next_id += 1
        clusters[new_id] = clusters[a] | clusters[b]
        tree[new_id] = f"({tree[a]},{tree[b]})"

        for x in active:
            if x == a or x == b:
                continue
            # UPGMA: average distance weighted by cluster sizes
            size_a = len(clusters[a])
            size_b = len(clusters[b])
            size_x = len(clusters[x])
            d_ax = D.get((a, x), 0.0)
            d_bx = D.get((b, x), 0.0)
            new_d = (size_a * d_ax + size_b * d_bx) / (size_a + size_b)
            D[(new_id, x)] = new_d
            D[(x, new_id)] = new_d

        active = [x for x in active if x != a and x != b]
        active.append(new_id)

    a, b = active
    return f"({tree[a]},{tree[b]})"


def reconstruct(sequences):
    """
    Reconstruct unrooted tree topology from DNA sequences.
    Returns a Newick string.
    """
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 250  # Safety margin from 270s limit

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    # Compute distance matrix
    dist = _compute_distance_matrix(sequences, labels)

    # Check deadline
    if time.time() > _DEADLINE - 10:
        # Return a star tree as fallback
        return "(" + ",".join(labels) + ")"

    # Use NJ for topology recovery
    result = _nj_newick_simple(dist, labels)

    return result

# EVOLVE-BLOCK-END
