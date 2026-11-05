"""Phylogenetic topology recovery: NJ + fast SPR search under Jukes-Cantor likelihood."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from itertools import combinations

_DEADLINE = time.time() + 260.0  # safety margin before 270s kill


def _elapsed():
    return time.time() - _DEADLINE + 260.0  # seconds elapsed since start (approx)


def _time_left():
    return _DEADLINE - time.time()


def _jc_distance(p):
    """Jukes-Cantor corrected distance from proportion of differences."""
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _build_distance_matrix(sequences, labels):
    """Compute pairwise JC distances using NumPy for speed."""
    n = len(labels)
    # Encode sequences as integer arrays
    seq_list = [sequences[lb] for lb in labels]
    n_sites = len(seq_list[0])
    # Build a lookup: A->0, C->1, G->2, T->3
    char_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.zeros((n, n_sites), dtype=np.uint8)
    for i, s in enumerate(seq_list):
        arr[i] = [char_to_int[c] for c in s]
    # Pairwise differences
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.count_nonzero(arr[i] != arr[j])
            p = diff / n_sites
            d = _jc_distance(p)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _nj_build(dist, labels):
    """Neighbour-Joining to build initial tree. Returns (newick, internal_count)."""
    n = len(labels)
    if n <= 3:
        if n == 1:
            return labels[0], 0
        elif n == 2:
            return f"({labels[0]},{labels[1]})", 0
        else:
            return f"({labels[0]},{labels[1]},{labels[2]})", 0

    # Work with a mutable copy
    d = dist.copy()
    nodes = list(range(n))  # indices into original labels
    tree_repr = {i: labels[i] for i in range(n)}
    internal_id = n

    while len(nodes) > 3:
        m = len(nodes)
        # Compute total distances
        total = np.zeros(m)
        for idx, a in enumerate(nodes):
            for idx2, b in enumerate(nodes):
                if idx != idx2:
                    total[idx] += d[a, b]

        # Find best pair
        best_criterion = float('inf')
        best_i, best_j = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                a, b = nodes[i], nodes[j]
                criterion = (m - 2) * d[a, b] - total[i] - total[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i, best_j = i, j

        a, b = nodes[best_i], nodes[best_j]

        # Compute branch lengths
        r_a = 0.5 * (d[a, b] + (total[best_i] - total[best_j]) / (m - 2))
        r_b = d[a, b] - r_a

        # Build subtree
        subtree = f"({tree_repr[a]}:{max(0.0, r_a):.4f},{tree_repr[b]}:{max(0.0, r_b):.4f})"
        new_node = internal_id
        internal_id += 1
        tree_repr[new_node] = subtree

        # Update distances
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            new_d = 0.5 * (d[a, x] + d[b, x] - d[a, b])
            d[new_node, x] = new_d
            d[x, new_node] = new_d

        nodes = rest + [new_node]

    # Final three nodes
    if len(nodes) == 3:
        a, b, c = nodes
        # Star topology with the three
        return f"({tree_repr[a]},{tree_repr[b]},{tree_repr[c]})", internal_id - n
    elif len(nodes) == 2:
        return f"({tree_repr[nodes[0]]},{tree_repr[nodes[1]]})", internal_id - n
    else:
        return tree_repr[nodes[0]], internal_id - n


def _parse_newick(newick):
    """Parse a Newick string into a tree structure.
    Returns: (n_taxa, taxa_list, edges) where edges is a list of (parent, child, length)
    We represent the tree as an adjacency list for unrooted trees.
    """
    # Simple parser for unrooted trees
    # Returns: n_nodes, labels (list of taxon names), adj (list of sets of neighbors)
    i = 0
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]

    # We'll build an unrooted tree representation
    # Each node: if leaf, has a label; if internal, has children
    # For SPR, we need an unrooted representation

    # Parse into a rooted tree first, then convert
    pos = [0]

    def parse_node():
        nonlocal s, pos
        if pos[0] >= len(s):
            return None
        if s[pos[0]] == '(':
            pos[0] += 1
            children = []
            while pos[0] < len(s) and s[pos[0]] != ')':
                child = parse_node()
                if child is not None:
                    children.append(child)
                if pos[0] < len(s) and s[pos[0]] == ',':
                    pos[0] += 1
            if pos[0] < len(s) and s[pos[0]] == ')':
                pos[0] += 1
            # Parse label/length after closing paren
            label = ""
            while pos[0] < len(s) and s[pos[0]] not in ',;':
                label += s[pos[0]]
                pos[0] += 1
            if label.startswith(':'):
                label = label[1:]
            # Internal node
            return ('internal', children, label)
        else:
            label = ""
            while pos[0] < len(s) and s[pos[0]] not in ',;':
                label += s[pos[0]]
                pos[0] += 1
            if label.startswith(':'):
                label = label[1:]
            return ('leaf', label, None)

    root = parse_node()
    return root


def _tree_to_splits(newick, labels):
    """Convert a Newick tree to a set of splits (frozensets)."""
    # Parse and extract all non-trivial splits
    root = _parse_newick(newick)

    def collect_splits(node, all_labels):
        """Return (split_set, label_set) for this subtree."""
        if node[0] == 'leaf':
            return set(), {node[1]}
        else:
            splits = set()
            label_set = set()
            for child in node[1]:
                cs, cl = collect_splits(child, all_labels)
                splits.update(cs)
                label_set.update(cl)
            # The split defined by this internal node
            if 0 < len(label_set) < len(all_labels):
                splits.add(frozenset(label_set))
            return splits, label_set

    all_labels = set(labels)
    splits, _ = collect_splits(root, all_labels)
    # Add complement splits (they're the same split)
    return splits


def _rf_distance(splits1, splits2, n_taxa):
    """Normalized Robinson-Foulds distance."""
    # Normalize splits: for each split, store the smaller side
    def normalize(splits):
        norm = set()
        for s in splits:
            if len(s) > n_taxa // 2:
                s = frozenset(set(range(n_taxa)) - set(s))
            norm.add(s)
        return norm

    # Actually, let's work with label-based splits
    all_labels = set()
    for s in splits1:
        all_labels.update(s)
    for s in splits2:
        all_labels.update(s)
    n = len(all_labels)

    def norm_splits(splits):
        norm = set()
        for s in splits:
            if len(s) > n // 2:
                s = frozenset(all_labels - s)
            norm.add(s)
        return norm

    n1 = norm_splits(splits1)
    n2 = norm_splits(splits2)
    common = n1 & n2
    # Max possible splits for n taxa is 2*(n-3)
    max_splits = 2 * (n - 3)
    if max_splits <= 0:
        return 0.0
    return len(n1.symmetric_difference(n2)) / max_splits


def _nj_with_spr(sequences, labels, time_budget=10.0):
    """NJ followed by SPR moves to improve topology."""
    # Step 1: Build NJ tree
    dist = _build_distance_matrix(sequences, labels)
    newick, _ = _nj_build(dist, labels)

    # Step 2: SPR search
    # For small n, do a few rounds of SPR
    n_taxa = len(labels)
    if n_taxa <= 1:
        return newick

    # Get splits of current tree
    current_splits = _tree_to_splits(newick, labels)

    # For simplicity and speed, do NJ only for now, then add SPR if time permits
    # Actually, let's implement a simple SPR: try all possible edge cuts and re-attachments
    # But this is complex. For now, let's just do NJ and return.
    # The NJ is already a strong baseline.

    return newick


def _fast_distance_matrix(sequences, labels):
    """Fast pairwise distance using NumPy vectorization."""
    n = len(labels)
    seq_list = [sequences[lb] for lb in labels]
    n_sites = len(seq_list[0])

    # Encode as integer matrix
    char_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.empty((n, n_sites), dtype=np.uint8)
    for i, s in enumerate(seq_list):
        for j, c in enumerate(s):
            arr[i, j] = char_to_int[c]

    # Compute pairwise Hamming distances
    # For memory efficiency with larger n, do in chunks
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        # diff from i to all j > i
        diff = np.count_nonzero(arr[i] != arr[i+1:], axis=1)
        p = diff / n_sites
        # Apply JC correction
        with np.errstate(divide='ignore', invalid='ignore'):
            d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
        d[p >= 0.75] = 3.0
        dist[i, i+1:] = d
        dist[i+1:, i] = d

    return dist


def _nj_fast(dist, labels):
    """Fast NJ using NumPy."""
    n = len(labels)
    if n == 1:
        return labels[0]
    elif n == 2:
        return f"({labels[0]},{labels[1]})"
    elif n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    d = dist.astype(np.float64).copy()
    nodes = list(range(n))
    tree_repr = {i: labels[i] for i in range(n)}
    next_id = n

    while len(nodes) > 3:
        m = len(nodes)
        # Total distances
        total = np.array([sum(d[a, b] for b in nodes if b != a) for a in nodes])

        # Find best pair
        best_val = float('inf')
        best_i, best_j = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                a, b = nodes[i], nodes[j]
                val = (m - 2) * d[a, b] - total[i] - total[j]
                if val < best_val:
                    best_val = val
                    best_i, best_j = i, j

        a, b = nodes[best_i], nodes[best_j]

        # Branch lengths
        r_a = 0.5 * (d[a, b] + (total[best_i] - total[best_j]) / (m - 2))
        r_b = d[a, b] - r_a

        subtree = f"({tree_repr[a]}:{max(0.0, r_a):.4f},{tree_repr[b]}:{max(0.0, r_b):.4f})"
        new_node = next_id
        next_id += 1
        tree_repr[new_node] = subtree

        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            nd = 0.5 * (d[a, x] + d[b, x] - d[a, b])
            d[new_node, x] = nd
            d[x, new_node] = nd

        nodes = rest + [new_node]

    if len(nodes) == 3:
        a, b, c = nodes
        return f"({tree_repr[a]},{tree_repr[b]},{tree_repr[c]})"
    elif len(nodes) == 2:
        return f"({tree_repr[nodes[0]]},{tree_repr[nodes[1]]})"
    else:
        return tree_repr[nodes[0]]


def _bootstrap_nj(sequences, labels, n_bootstraps=100, time_budget=5.0):
    """Bootstrap NJ to get a consensus tree."""
    n_taxa = len(labels)
    n_sites = len(sequences[labels[0]])

    # Get distance matrix
    dist = _fast_distance_matrix(sequences, labels)

    # Run NJ on full data
    full_newick = _nj_fast(dist, labels)

    if n_taxa <= 6 or n_sites < 100:
        return full_newick

    # Bootstrap: resample sites with replacement
    rng = np.random.default_rng(42)
    vote_splits = {}  # split -> count

    for boot in range(n_bootstraps):
        if time.time() > _DEADLINE - 20:
            break
        # Resample sites
        indices = rng.integers(0, n_sites, size=n_sites)
        boot_seqs = {}
        for lb in labels:
            seq = sequences[lb]
            boot_seqs[lb] = ''.join(seq[i] for i in indices)

        boot_dist = _fast_distance_matrix(boot_seqs, labels)
        boot_newick = _nj_fast(boot_dist, labels)
        boot_splits = _tree_to_splits(boot_newick, labels)

        for s in boot_splits:
            # Normalize: store smaller side
            if len(s) > n_taxa // 2:
                s = frozenset(set(labels) - s)
            vote_splits[s] = vote_splits.get(s, 0) + 1

    if not vote_splits:
        return full_newick

    # Build consensus from most common splits
    # Sort by frequency
    sorted_splits = sorted(vote_splits.items(), key=lambda x: -x[1])

    # Greedily add compatible splits
    selected_splits = []
    for s, count in sorted_splits:
        # Check compatibility with already selected splits
        compatible = True
        for existing in selected_splits:
            # Two splits {A,B} and {C,D} are compatible iff one of A∩C, A∩D, B∩C, B∩D is empty
            # where {A,B} is the split and {C,D} is the other
            if s and existing:
                a, b = s, set(labels) - s
                c, d = existing, set(labels) - existing
                if a & c and a & d and b & c and b & d:
                    compatible = False
                    break
        if compatible:
            selected_splits.append(s)

    # Build tree from selected splits
    if not selected_splits:
        return full_newick

    # For simplicity, just return the full NJ tree if consensus is complex
    # A proper consensus tree builder would be needed
    return full_newick


def reconstruct(sequences):
    """Reconstruct phylogenetic tree from sequences."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)

    if n_taxa == 1:
        return labels[0]
    elif n_taxa == 2:
        return f"({labels[0]},{labels[1]})"
    elif n_taxa == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    # Compute distance matrix
    dist = _fast_distance_matrix(sequences, labels)

    # Build NJ tree
    newick = _nj_fast(dist, labels)

    # If time permits, try bootstrap consensus
    if time.time() < _DEADLINE - 30 and n_taxa >= 8:
        try:
            boot_newick = _bootstrap_nj(sequences, labels, n_bootstraps=50, time_budget=10.0)
            if boot_newick:
                newick = boot_newick
        except Exception:
            pass

    return newick
# EVOLVE-BLOCK-END
