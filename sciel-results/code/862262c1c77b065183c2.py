"""Phylogenetic topology recovery using optimized Neighbor Joining with NumPy."""
import time
import math
import numpy as np

# Deadline guard: 270s wall-clock limit, use 250s as safety margin
_DEADLINE = time.time() + 250.0


def _time_exceeded():
    return time.time() > _DEADLINE


def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distance matrix using vectorized NumPy."""
    n = len(labels)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    
    seqs = [sequences[label] for label in labels]
    L = len(seqs[0]) if seqs else 0
    
    if L == 0:
        return np.zeros((n, n), dtype=np.float64)
    
    # Convert sequences to integer arrays: A=0, C=1, G=2, T=3
    # Use a lookup table for vectorized conversion
    lookup = np.zeros(256, dtype=np.int8)
    lookup[65] = 0  # A
    lookup[67] = 1  # C
    lookup[71] = 2  # G
    lookup[84] = 3  # T
    lookup[97] = 0  # a
    lookup[99] = 1  # c
    lookup[103] = 2  # g
    lookup[116] = 3  # t
    
    # Convert all sequences at once
    seq_arrays = np.zeros((n, L), dtype=np.int8)
    for i, s in enumerate(seqs):
        seq_arrays[i] = lookup[np.frombuffer(s.encode('ascii'), dtype=np.uint8)]
    
    # Compute pairwise Hamming distances using broadcasting
    # For n up to 32 and L up to 2000, this is manageable
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        # Compare seq i against all j > i
        diffs = np.count_nonzero(seq_arrays[i+1:] != seq_arrays[i], axis=1)
        p = diffs / L
        # Clip to avoid log(0)
        p_clipped = np.minimum(p, 0.7499)
        d = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
        # Handle p >= 0.7499 case
        mask = p >= 0.7499
        d[mask] = 3.0
        D[i, i+1:] = d
        D[i+1:, i] = d
    return D


def _neighbor_joining(D, labels):
    """
    Perform Neighbor Joining on distance matrix D.
    Returns a Newick string (topology only, no branch lengths needed for RF score).
    """
    n = len(labels)
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    
    # Work on a copy
    dist = D.copy()
    names = list(labels)
    
    # Track the tree structure: for each current node, store the Newick substring
    tree_parts = list(names)
    
    num_nodes = n
    while num_nodes > 3:
        if _time_exceeded():
            break
        
        m = num_nodes
        # Compute total distances (row sums)
        r = np.sum(dist, axis=1)
        
        # Find the pair that minimizes the NJ criterion using vectorized computation
        # criterion[i,j] = (m-2)*dist[i,j] - r[i] - r[j]
        criteria = (m - 2) * dist - r[:, np.newaxis] - r[np.newaxis, :]
        # Set diagonal to infinity to avoid self-pairs
        np.fill_diagonal(criteria, np.inf)
        # Only consider upper triangle
        iu = np.triu_indices(m, k=1)
        flat_criteria = criteria[iu]
        best_idx = np.argmin(flat_criteria)
        best_i, best_j = iu[0][best_idx], iu[1][best_idx]
        
        i, j = best_i, best_j
        
        # Create new internal node
        new_part = "(" + tree_parts[i] + "," + tree_parts[j] + ")"
        
        # Compute distances from new node to all other nodes
        # Indices of nodes to keep
        keep_mask = np.ones(m, dtype=bool)
        keep_mask[i] = False
        keep_mask[j] = False
        new_active = np.where(keep_mask)[0]
        
        # Newick's formula: d(new, k) = 0.5 * (d(i,k) + d(j,k) - d(i,j))
        new_distances = 0.5 * (dist[i, new_active] + dist[j, new_active] - dist[i, j])
        
        # Build new distance matrix using NumPy slicing
        new_dist = np.empty((m - 1, m - 1), dtype=np.float64)
        # Copy the submatrix of kept nodes
        new_dist[:m-2, :m-2] = dist[np.ix_(new_active, new_active)]
        # Set the last row and column to new_distances
        new_dist[:m-2, m-2] = new_distances
        new_dist[m-2, :m-2] = new_distances
        new_dist[m-2, m-2] = 0.0
        
        # Update structures
        tree_parts = [tree_parts[k] for k in new_active] + [new_part]
        dist = new_dist
        num_nodes = m - 1
    
    # Combine remaining nodes
    if num_nodes == 3:
        result = "(" + ",".join(tree_parts) + ")"
    elif num_nodes == 2:
        result = "(" + ",".join(tree_parts) + ")"
    else:
        result = "(" + ",".join(tree_parts) + ")"
    
    return result + ";"


def _bipartitions_from_newick(newick):
    """
    Extract all non-trivial bipartitions (splits) from a Newick string.
    Returns a set of frozensets, each representing one side of a split.
    Only non-trivial splits (both sides have at least 2 taxa) are included.
    """
    # Parse Newick to get leaf labels and structure
    # We'll do a simple recursive parse to find all internal nodes and their leaf sets
    
    # First, extract all leaf labels
    import re
    leaves = set(re.findall(r't\d+', newick))
    n_leaves = len(leaves)
    
    # Parse the tree structure
    # We'll use a stack-based approach to find all internal nodes and their leaf sets
    
    # Remove trailing semicolon
    newick = newick.rstrip(';')
    
    # We need to find all splits. A split is defined by an internal node:
    # one side is the set of leaves in the subtree rooted at that node,
    # the other side is the complement.
    # For unrooted trees, each internal edge corresponds to a split.
    
    # Let's parse the Newick into a tree structure
    # For simplicity, we'll track leaf sets for each node
    
    def parse_newick(s):
        """
        Parse Newick string and return a list of (leaf_set) for each internal node.
        Also returns the full set of leaves.
        """
        i = 0
        n = len(s)
        
        def parse_node():
            nonlocal i
            if i >= n:
                return None
            
            if s[i] == '(':
                i += 1  # skip '('
                children = []
                while i < n and s[i] != ')':
                    child = parse_node()
                    if child is not None:
                        children.append(child)
                    if i < n and s[i] == ',':
                        i += 1
                if i < n and s[i] == ')':
                    i += 1  # skip ')'
                
                # After ')', there might be a label or branch length
                # For our purposes, we just need the leaf set
                leaf_set = set()
                for child in children:
                    leaf_set.update(child)
                
                # Skip any label/branch length after ')'
                while i < n and s[i] not in (',', ')', ';'):
                    i += 1
                
                return leaf_set
            else:
                # Leaf node
                start = i
                while i < n and s[i] not in (',', ')', ';', '(', ':'):
                    i += 1
                label = s[start:i].strip()
                # Skip branch length if present
                if i < n and s[i] == ':':
                    i += 1
                    while i < n and s[i] not in (',', ')', ';', '(', ':'):
                        i += 1
                return {label}
        
        result = parse_node()
        return result
    
    # Get the full leaf set
    full_set = parse_newick(newick)
    if full_set is None:
        return set()
    
    # Now we need to find all internal nodes and their leaf sets
    # Let's re-parse but track all internal nodes
    
    internal_leaf_sets = []
    
    def parse_node_with_tracking(s):
        i = 0
        n = len(s)
        
        def parse_node():
            nonlocal i
            if i >= n:
                return None
            
            if s[i] == '(':
                i += 1  # skip '('
                children = []
                while i < n and s[i] != ')':
                    child = parse_node()
                    if child is not None:
                        children.append(child)
                    if i < n and s[i] == ',':
                        i += 1
                if i < n and s[i] == ')':
                    i += 1  # skip ')'
                
                leaf_set = set()
                for child in children:
                    leaf_set.update(child)
                
                # This is an internal node
                internal_leaf_sets.append(leaf_set)
                
                # Skip any label/branch length after ')'
                while i < n and s[i] not in (',', ')', ';'):
                    i += 1
                
                return leaf_set
            else:
                start = i
                while i < n and s[i] not in (',', ')', ';', '(', ':'):
                    i += 1
                label = s[start:i].strip()
                if i < n and s[i] == ':':
                    i += 1
                    while i < n and s[i] not in (',', ')', ';', '(', ':'):
                        i += 1
                return {label}
        
        result = parse_node()
        return result
    
    parse_node_with_tracking(newick)
    
    # For each internal node's leaf set, create a split
    # The split is (leaf_set, full_set - leaf_set)
    # We represent it as a frozenset of the smaller side (or either side, consistently)
    splits = set()
    for leaf_set in internal_leaf_sets:
        complement = full_set - leaf_set
        if len(leaf_set) >= 2 and len(complement) >= 2:
            # Use the frozenset representation
            # To avoid duplicates, we can use a canonical form
            # e.g., always store the frozenset that contains the smallest leaf label
            if min(leaf_set) < min(complement):
                splits.add(frozenset(leaf_set))
            else:
                splits.add(frozenset(complement))
    
    return splits


def _normalized_rf_distance(splits1, splits2):
    """
    Compute the normalized Robinson-Foulds distance between two sets of splits.
    Returns a value in [0, 1].
    """
    if not splits1 and not splits2:
        return 0.0
    
    intersection = splits1 & splits2
    union = splits1 | splits2
    
    if not union:
        return 0.0
    
    return 1.0 - len(intersection) / len(union)


def _nj_with_polish(D, labels, max_iterations=10):
    """
    Run NJ, then attempt to improve the topology via local search.
    Since we only have the distance matrix, we can try to refine by
    checking if swapping neighbors improves the RF distance (but we don't have
    the true tree, so we can't directly optimize RF).
    
    Instead, we can run NJ multiple times with slight perturbations to the
    distance matrix and keep the best result.
    
    However, without the true tree, we can't score candidates. So we'll just
    run NJ once. The polish idea doesn't directly apply here since we don't
    have an objective to optimize beyond the NJ criterion itself.
    
    For now, we'll just run standard NJ.
    """
    return _neighbor_joining(D, labels)


def reconstruct(sequences):
    """
    Reconstruct the unrooted tree topology from sequences.
    Returns a Newick string over labels t0..t(n-1).
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    
    # Compute distance matrix
    D = _jc_distance_matrix(sequences, labels)
    
    # Run Neighbor Joining
    result = _nj_with_polish(D, labels)
    
    return result
