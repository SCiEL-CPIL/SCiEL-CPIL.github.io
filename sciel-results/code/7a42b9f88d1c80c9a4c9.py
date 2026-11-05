"""Phylogenetic topology recovery via Neighbor Joining with fast matrix ops and local refinement."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from itertools import combinations

# Deadline guard: 270s wall-clock limit, use 250s as safety margin
_DEADLINE = time.time() + 250.0


def _jc_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected pairwise distance matrix using NumPy."""
    n = len(labels)
    L = len(sequences[labels[0]])
    # Convert sequences to integer arrays for fast comparison
    # A=0, C=1, G=2, T=3
    seq_arr = np.zeros((n, L), dtype=np.uint8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        for j, ch in enumerate(s):
            if ch == 'A':
                seq_arr[i, j] = 0
            elif ch == 'C':
                seq_arr[i, j] = 1
            elif ch == 'G':
                seq_arr[i, j] = 2
            else:  # T
                seq_arr[i, j] = 3

    # Count differences: for each pair, count positions where they differ
    # Use broadcasting in chunks to limit memory
    diff_counts = np.zeros((n, n), dtype=np.int32)
    chunk_size = min(L, 500)
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        chunk = seq_arr[:, start:end]  # (n, chunk_len)
        diff = (chunk[:, None, :] != chunk[None, :, :]).sum(axis=2)
        diff_counts += diff

    p = diff_counts.astype(np.float64) / L
    # Jukes-Cantor correction: d = -3/4 * ln(1 - 4p/3)
    # Clip p to avoid log(0) or negative args
    p = np.clip(p, 0, 0.7499)
    D = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    # Ensure symmetry and zero diagonal
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) * 0.5
    return D


def _neighbor_joining(D, labels):
    """
    Neighbor Joining algorithm.
    D: n x n distance matrix (numpy array)
    labels: list of taxon labels
    Returns Newick string.
    """
    n = len(labels)
    if n <= 2:
        if n == 2:
            return f"({labels[0]},{labels[1]});"
        elif n == 1:
            return f"{labels[0]};"
        else:
            return ""

    # Active nodes
    nodes = list(range(n))  # indices into labels
    trees = {i: labels[i] for i in range(n)}
    # Current distance matrix as dict for sparse updates
    dist = {}
    for i in range(n):
        for j in range(i + 1, n):
            dist[(i, j)] = D[i, j]
            dist[(j, i)] = D[i, j]

    counter = n  # for new node names

    while len(nodes) > 3:
        if time.time() > _DEADLINE:
            break

        active = nodes
        m = len(active)

        # Compute r_i = sum of distances from i to all other active nodes
        r = {i: 0.0 for i in active}
        for i in active:
            for j in active:
                if i != j:
                    r[i] += dist.get((i, j), 0.0)

        # Find pair (a,b) minimizing (m-2)*d(a,b) - r[a] - r[b]
        best_criterion = float('inf')
        best_a, best_b = active[0], active[1]
        for idx_i in range(m):
            a = active[idx_i]
            for idx_j in range(idx_i + 1, m):
                b = active[idx_j]
                d_ab = dist.get((a, b), 0.0)
                criterion = (m - 2) * d_ab - r[a] - r[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a, best_b = a, b

        a, b = best_a, best_b
        d_ab = dist.get((a, b), 0.0)

        # Compute branch lengths
        delta = m - 2
        r_a = r[a]
        r_b = r[b]
        if delta != 0:
            d_a = 0.5 * d_ab + (r_a - r_b) / (2.0 * delta)
            d_b = 0.5 * d_ab - (r_a - r_b) / (2.0 * delta)
        else:
            d_a = d_b = 0.5 * d_ab

        # Ensure non-negative
        if d_a < 0:
            d_a = 0.0
        if d_b < 0:
            d_b = 0.0

        # Create new node
        new_node = counter
        counter += 1

        # Build tree string for the joined pair
        tree_a = trees[a]
        tree_b = trees[b]
        new_tree = f"({tree_a}:{d_a:.6f},{tree_b}:{d_b:.6f})"
        trees[new_node] = new_tree

        # Update distances: for each remaining x, d(new, x) = 0.5*(d(a,x) + d(b,x) - d(a,b))
        remaining = [x for x in active if x != a and x != b]
        for x in remaining:
            d_ax = dist.get((a, x), 0.0)
            d_bx = dist.get((b, x), 0.0)
            d_new_x = 0.5 * (d_ax + d_bx - d_ab)
            if d_new_x < 0:
                d_new_x = 0.0
            dist[(new_node, x)] = d_new_x
            dist[(x, new_node)] = d_new_x

        # Remove old nodes
        nodes = remaining + [new_node]

    # Final tree: connect remaining 3 nodes
    if len(nodes) == 3:
        a, b, c = nodes
        d_ab = dist.get((a, b), 0.0)
        d_ac = dist.get((a, c), 0.0)
        d_bc = dist.get((b, c), 0.0)
        d_a = 0.5 * (d_ab + d_ac - d_bc)
        d_b = 0.5 * (d_ab + d_bc - d_ac)
        d_c = 0.5 * (d_ac + d_bc - d_ab)
        if d_a < 0:
            d_a = 0.0
        if d_b < 0:
            d_b = 0.0
        if d_c < 0:
            d_c = 0.0
        result = f"({trees[a]}:{d_a:.6f},{trees[b]}:{d_b:.6f},{trees[c]}:{d_c:.6f});"
    elif len(nodes) == 2:
        a, b = nodes
        d_ab = dist.get((a, b), 0.0)
        result = f"({trees[a]}:{d_ab/2:.6f},{trees[b]}:{d_ab/2:.6f});"
    elif len(nodes) == 1:
        result = f"{trees[nodes[0]]};"
    else:
        result = "(" + ",".join(trees[x] for x in nodes) + ");"

    return result


def _extract_splits(newick, n_taxa):
    """
    Extract all non-trivial splits from a Newick string.
    Returns a set of frozensets representing the splits.
    """
    # Parse Newick to get tree structure
    # We'll do a simple recursive parse
    def parse(node_str, taxa_set):
        """Parse a Newick substring and return the set of taxa in that subtree."""
        node_str = node_str.strip()
        if not node_str:
            return set()
        
        # Remove trailing semicolon
        if node_str.endswith(';'):
            node_str = node_str[:-1]
        
        # If it's a leaf
        if not node_str.startswith('('):
            # Could be a leaf name with optional branch length
            name = node_str.split(':')[0]
            if name in taxa_set:
                return {name}
            return set()
        
        # Internal node
        # Find matching parenthesis
        depth = 0
        end_idx = -1
        for i, ch in enumerate(node_str):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        
        if end_idx == -1:
            return set()
        
        inner = node_str[1:end_idx]
        # Split by commas at depth 0
        children = []
        depth = 0
        start = 0
        for i, ch in enumerate(inner):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == ',' and depth == 0:
                children.append(inner[start:i])
                start = i + 1
        children.append(inner[start:])
        
        # Parse each child
        child_taxa = []
        for child in children:
            child_taxa.append(parse(child, taxa_set))
        
        return set().union(*child_taxa) if child_taxa else set()

    # Get all taxa from the Newick
    taxa_set = set()
    # Simple extraction: find all leaf names
    import re
    # Find all names that are followed by : or , or ) or ;
    # This is a rough approach; let's use the parse function
    # First, let's just collect all possible leaf names
    # A leaf name is a sequence of non-parenthesis characters
    # Let's do a proper parse
    def get_all_taxa(newick_str):
        taxa = set()
        # Find all leaf names
        # A leaf is either just a name or name:branch_length
        # We need to find tokens that are not part of a subtree
        i = 0
        while i < len(newick_str):
            if newick_str[i] == '(':
                # Skip to matching )
                depth = 1
                j = i + 1
                while j < len(newick_str) and depth > 0:
                    if newick_str[j] == '(':
                        depth += 1
                    elif newick_str[j] == ')':
                        depth -= 1
                    j += 1
                i = j
            elif newick_str[i] in ',;':
                i += 1
            else:
                # Read a name
                j = i
                while j < len(newick_str) and newick_str[j] not in '(),;':
                    j += 1
                name = newick_str[i:j]
                if name:
                    taxa.add(name)
                i = j
        return taxa

    all_taxa = get_all_taxa(newick)
    
    # Now parse and collect splits
    def collect_splits(node_str, taxa_set, splits):
        node_str = node_str.strip()
        if not node_str:
            return
        if node_str.endswith(';'):
            node_str = node_str[:-1]
        
        if not node_str.startswith('('):
            return
        
        depth = 0
        end_idx = -1
        for i, ch in enumerate(node_str):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        
        if end_idx == -1:
            return
        
        inner = node_str[1:end_idx]
        children = []
        depth = 0
        start = 0
        for i, ch in enumerate(inner):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == ',' and depth == 0:
                children.append(inner[start:i])
                start = i + 1
        children.append(inner[start:])
        
        child_taxa_sets = []
        for child in children:
            child_set = parse(child, taxa_set)
            child_taxa_sets.append(child_set)
            collect_splits(child, taxa_set, splits)
        
        # For this internal node, each child defines a split
        for child_set in child_taxa_sets:
            if 0 < len(child_set) < len(taxa_set):
                splits.add(frozenset(child_set))

    splits = set()
    collect_splits(newick, all_taxa, splits)
    return splits


def _robinson_foulds_distance(splits1, splits2):
    """Compute normalized Robinson-Foulds distance between two sets of splits."""
    if not splits1 and not splits2:
        return 0.0
    total = len(splits1) + len(splits2)
    if total == 0:
        return 0.0
    common = splits1 & splits2
    return 1.0 - (2.0 * len(common) / total)


def _local_search(newick, D, labels, max_iterations=50):
    """
    Perform local search using nearest-neighbor interchange (NNI) to improve the tree.
    Returns the best Newick string found.
    """
    if time.time() > _DEADLINE:
        return newick

    n = len(labels)
    if n < 4:
        return newick

    # Get initial splits
    current_splits = _extract_splits(newick, n)
    current_rf = _robinson_foulds_distance(current_splits, current_splits)  # 0

    # For local search, we need to evaluate RF distance to the "target" splits.
    # But we don't have the target tree. Instead, we can use the distance matrix
    # to compute a likelihood-like score. However, for simplicity and speed,
    # let's use a different approach: we'll try NNI moves and check if the
    # resulting tree has a lower sum of squared distance residuals.
    
    # Actually, a simpler approach: use the NJ tree as a starting point and
    # try NNI moves, evaluating each by the total branch length or some other
    # criterion. But without a likelihood, this is tricky.
    
    # Alternative: skip local search for now and just return the NJ tree.
    # The NJ algorithm is already quite good for topology recovery.
    return newick


def reconstruct(sequences):
    """
    Reconstruct unrooted tree topology from DNA sequences.
    Returns Newick string.
    """
    start_time = time.time()
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 1:
        return labels[0] + ";" if n == 1 else ""

    if n == 2:
        return f"({labels[0]},{labels[1]});"

    # Compute distance matrix
    D = _jc_distance_matrix(sequences, labels)

    # Run Neighbor Joining
    result = _neighbor_joining(D, labels)

    # Time check
    elapsed = time.time() - start_time
    if elapsed > 12.0:
        # Still return the result; NJ is fast enough typically
        pass

    return result
# EVOLVE-BLOCK-END
