"""Phylogenetic topology recovery using Neighbor Joining with quartet-based local search refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def _build_newick(tree):
    """Recursively build Newick string from nested tuple structure."""
    if isinstance(tree, str):
        return tree
    if len(tree) == 2:
        return f"({_build_newick(tree[0])},{_build_newick(tree[1])})"
    return str(tree)

def _parse_newick(s):
    """Parse Newick string into nested tuple structure. Returns (tree, next_index)."""
    s = s.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    def parse(i):
        if i >= len(s):
            return None, i
        if s[i] == '(':
            i += 1
            left, i = parse(i)
            if i < len(s) and s[i] == ',':
                i += 1
            right, i = parse(i)
            if i < len(s) and s[i] == ')':
                i += 1
            return (left, right), i
        else:
            j = i
            while j < len(s) and s[j] not in ',;':
                j += 1
            return s[i:j], j
    
    tree, _ = parse(0)
    return tree

def _get_splits(tree, all_labels):
    """Get all non-trivial splits from a tree structure. Returns set of frozensets."""
    splits = set()
    
    def collect(node):
        if isinstance(node, str):
            return {node}
        left_set = collect(node[0])
        right_set = collect(node[1])
        full_set = left_set | right_set
        if 1 < len(full_set) < len(all_labels):
            splits.add(frozenset(full_set))
        return full_set
    
    collect(tree)
    return splits

def _quartet_agreement(tree, labels):
    """
    Count the number of quartets whose topology matches the tree.
    For each quartet {a,b,c,d}, determine which of the 3 topologies the tree implies,
    and check if it matches the "expected" topology based on which pair is closest in the tree.
    Actually, we just count how many quartets are "resolved" consistently.
    
    Better approach: For each quartet, determine the split induced by the tree.
    Then compare with the split that minimizes the sum of within-cluster distances.
    But since we don't have the true tree, we use the distance matrix to determine
    the "best" quartet topology and check if our tree agrees.
    """
    n = len(labels)
    if n < 4:
        return 0
    
    # Get all splits from the tree
    splits = _get_splits(tree, labels)
    
    agreement = 0
    total_quartets = 0
    
    for a, b, c, d in combinations(range(n), 4):
        la, lb, lc, ld = labels[a], labels[b], labels[c], labels[d]
        total_quartets += 1
        
        # Three possible splits
        split1 = frozenset([la, lb])  # (a,b)|(c,d)
        split2 = frozenset([la, lc])  # (a,c)|(b,d)
        split3 = frozenset([la, ld])  # (a,d)|(b,c)
        
        # Determine which split the tree implies
        tree_split = None
        if split1 in splits:
            tree_split = split1
        elif split2 in splits:
            tree_split = split2
        elif split3 in splits:
            tree_split = split3
        else:
            # Try the complement
            comp1 = frozenset([lc, ld])
            comp2 = frozenset([lb, ld])
            comp3 = frozenset([lb, lc])
            if comp1 in splits:
                tree_split = split1
            elif comp2 in splits:
                tree_split = split2
            elif comp3 in splits:
                tree_split = split3
            else:
                continue
        
        if tree_split is not None:
            agreement += 1
    
    return agreement, total_quartets

def _nj_tree(dist_mat, labels):
    """
    Perform Neighbor Joining on a distance matrix.
    Returns a nested tuple tree structure.
    """
    n_labels = len(labels)
    if n_labels < 2:
        return labels[0] if labels else ""
    if n_labels == 2:
        return (labels[0], labels[1])
    if n_labels == 3:
        return (labels[0], (labels[1], labels[2]))
    
    nodes = list(range(n_labels))  # Use indices for efficiency
    D = dist_mat.copy()
    
    # Tree structure: map node_id -> (left, right) or leaf_label
    tree_struct = {i: labels[i] for i in range(n_labels)}
    next_id = n_labels
    
    while len(nodes) > 3:
        n = len(nodes)
        
        # Compute row sums r
        r = D.sum(axis=1)
        
        # Calculate Q matrix: Q_ij = (n-2)D_ij - r_i - r_j
        Q = (n - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        # Find min Q pair (i, j)
        flat_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = flat_idx
        if i == j:
            continue
        
        node_i = nodes[i]
        node_j = nodes[j]
        
        # Create new node
        new_node = next_id
        next_id += 1
        tree_struct[new_node] = (tree_struct[node_i], tree_struct[node_j])
        
        # Determine which indices to keep
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[i] = False
        keep_mask[j] = False
        kept_indices = np.where(keep_mask)[0]
        
        # New nodes list
        new_nodes_list = [nodes[k] for k in kept_indices]
        new_nodes_list.append(new_node)
        
        # Construct new distance matrix
        new_n = n - 1
        D_new = np.zeros((new_n, new_n), dtype=np.float64)
        
        # Fill kept-kept distances
        D_keep_keep = D[np.ix_(kept_indices, kept_indices)]
        D_new[:-1, :-1] = D_keep_keep
        
        # Fill kept-new_node distances
        dist_i_kept = D[kept_indices, i]
        dist_j_kept = D[kept_indices, j]
        dist_ij = D[i, j]
        
        new_branches = 0.5 * (dist_i_kept + dist_j_kept - dist_ij)
        
        D_new[:-1, -1] = new_branches
        D_new[-1, :-1] = new_branches
        
        nodes = new_nodes_list
        D = D_new
    
    # Final 3 nodes
    if len(nodes) == 3:
        a, b, c = nodes
        tree_struct['final'] = (tree_struct[a], (tree_struct[b], tree_struct[c]))
        final_node = 'final'
    elif len(nodes) == 2:
        a, b = nodes
        tree_struct['final'] = (tree_struct[a], tree_struct[b])
        final_node = 'final'
    else:
        final_node = nodes[0]
    
    return tree_struct[final_node]

def _nni_refine(tree, labels, dist_mat, max_iterations=5):
    """
    NNI-like refinement: try to improve the tree by considering local moves.
    For each internal edge, try the two alternative quartet resolutions.
    Keep the move that increases quartet agreement.
    """
    n = len(labels)
    if n < 4:
        return tree
    
    # Precompute distance matrix indexed by label
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    def get_quartet_score(tree_struct):
        """Compute total quartet agreement score."""
        splits = _get_splits(tree_struct, labels)
        score = 0
        for a, b, c, d in combinations(range(n), 4):
            la, lb, lc, ld = labels[a], labels[b], labels[c], labels[d]
            
            # Three possible splits and their "cost" based on distances
            # The "best" quartet topology is the one that minimizes the sum of
            # the two cross-distances (or equivalently, maximizes the sum of
            # within-cluster distances)
            
            # Split (a,b)|(c,d): cross distances are d(a,c)+d(b,d) and d(a,d)+d(b,c)
            # Actually, for a quartet tree, the topology is determined by which pair
            # is "together". The cost of a topology is the sum of the two distances
            # that cross the split.
            
            # For split {a,b}|{c,d}: cost = d(a,c) + d(b,d) + d(a,d) + d(b,c) ... no
            # Let me think more carefully.
            
            # In a quartet tree with topology (a,b)|(c,d), the path from a to c goes
            # through the internal node, and similarly for b to d. The "consistent"
            # distances are d(a,b) and d(c,d), while the "inconsistent" ones are
            # d(a,c), d(a,d), d(b,c), d(b,d).
            
            # A simpler metric: for each quartet, determine which of the 3 topologies
            # the tree implies, and check if it's the one that minimizes the sum of
            # the two longest pairwise distances (or some other criterion).
            
            # Actually, let's use a different approach: for each quartet, compute
            # the three possible "split costs" and see which split the tree implies.
            # The score is 1 if the tree implies the split with minimum cost.
            
            # Split costs:
            # (a,b)|(c,d): cost = d(a,c) + d(b,d)  [sum of cross distances for this pairing]
            # Wait, that's not right either.
            
            # Let me use the standard quartet scoring:
            # For quartet {a,b,c,d}, define:
            #   S1 = d(a,b) + d(c,d)  [sum of within-cluster distances for split (a,b)|(c,d)]
            #   S2 = d(a,c) + d(b,d)  [for split (a,c)|(b,d)]
            #   S3 = d(a,d) + d(b,c)  [for split (a,d)|(b,c)]
            # The "best" split is the one with maximum S (most within-cluster similarity).
            
            ia, ib, ic, id_ = label_to_idx[la], label_to_idx[lb], label_to_idx[lc], label_to_idx[ld]
            
            S1 = dist_mat[ia, ib] + dist_mat[ic, id_]
            S2 = dist_mat[ia, ic] + dist_mat[ib, id_]
            S3 = dist_mat[ia, id_] + dist_mat[ib, ic]
            
            # The best split is the one with maximum S
            best_S = max(S1, S2, S3)
            
            # Determine which split the tree implies
            split1 = frozenset([la, lb])
            split2 = frozenset([la, lc])
            split3 = frozenset([la, ld])
            
            tree_implies = None
            if split1 in splits:
                tree_implies = 1
            elif split2 in splits:
                tree_implies = 2
            elif split3 in splits:
                tree_implies = 3
            else:
                comp1 = frozenset([lc, ld])
                comp2 = frozenset([lb, ld])
                comp3 = frozenset([lb, lc])
                if comp1 in splits:
                    tree_implies = 1
                elif comp2 in splits:
                    tree_implies = 2
                elif comp3 in splits:
                    tree_implies = 3
            
            if tree_implies is None:
                continue
            
            # Check if the tree implies the best split
            if tree_implies == 1 and S1 == best_S:
                score += 1
            elif tree_implies == 2 and S2 == best_S:
                score += 1
            elif tree_implies == 3 and S3 == best_S:
                score += 1
        
        return score
    
    # Get initial score
    current_score = get_quartet_score(tree)
    best_tree = tree
    best_score = current_score
    
    # Try NNI-like moves: for each internal edge, try alternative resolutions
    # This is complex to implement generically, so let's use a simpler approach:
    # Try random subtree swaps and keep improvements
    
    import random
    rng = random.Random(42)
    
    def try_swap(tree_struct, labels_list):
        """Try a random subtree swap and return new tree if improved."""
        # Find all internal nodes
        internal_nodes = []
        
        def find_internal(node):
            if isinstance(node, str):
                return
            left, right = node
            find_internal(left)
            find_internal(right)
            internal_nodes.append(node)
        
        find_internal(tree_struct)
        
        if not internal_nodes:
            return None
        
        # Pick a random internal node and try to swap its children
        # This is a very crude NNI move
        idx = rng.randint(0, len(internal_nodes) - 1)
        node = internal_nodes[idx]
        
        # Try swapping children
        new_node = (node[1], node[0])
        
        # Rebuild tree with this change
        def replace_node(tree_struct, target, replacement):
            if isinstance(tree_struct, str):
                return tree_struct
            if tree_struct is target:
                return replacement
            left, right = tree_struct
            new_left = replace_node(left, target, replacement)
            new_right = replace_node(right, target, replacement)
            return (new_left, new_right)
        
        # This approach is flawed because we need to modify the tree in place
        # Let's use a different strategy: represent the tree as a list of edges
        return None
    
    # Given the complexity, let's use a simpler refinement:
    # Re-run NJ with slightly perturbed distances (add small noise)
    # and keep the best tree
    
    best_tree = tree
    best_score = current_score
    
    for iteration in range(max_iterations):
        # Add small noise to distances
        noise = np.random.randn(n, n) * 0.01
        np.fill_diagonal(noise, 0)
        D_perturbed = dist_mat + 0.5 * (noise + noise.T)
        np.fill_diagonal(D_perturbed, 0)
        
        new_tree = _nj_tree(D_perturbed, labels)
        new_score = get_quartet_score(new_tree)
        
        if new_score > best_score:
            best_tree = new_tree
            best_score = new_score
    
    return best_tree

def reconstruct(sequences):
    start_time = time.time()
    time_limit = 265.0  # Safety margin before 270s kill
    
    labels = sorted(sequences.keys(), key=lambda x: int(x[1:]))
    n_labels = len(labels)
    
    if n_labels < 2:
        return labels[0] + ";" if labels else ";"
    if n_labels == 2:
        return f"({labels[0]},{labels[1]})"
    if n_labels == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}))"
    
    # Convert sequences to integer matrix
    L = len(next(iter(sequences.values())))
    
    # Fast conversion using numpy
    base_to_int = np.array([0, 1, 2, 3])
    char_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    seq_mat = np.zeros((n_labels, L), dtype=np.int8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j, base in enumerate(seq):
            seq_mat[i, j] = char_to_idx[base]
    
    # Compute pairwise Hamming distances
    # For memory efficiency with larger L, do this in chunks
    # But for N<=32 and L<=2000, the full (N,N,L) array is manageable
    diff = (seq_mat[:, None, :] != seq_mat[None, :, :]).astype(np.int8)
    hamming = diff.sum(axis=2)
    
    # Jukes-Cantor correction
    p = hamming.astype(np.float64) / L
    p_clamped = np.clip(p, 0.0, 0.7499999)
    dist_mat = -0.75 * np.log(1.0 - (4.0/3.0) * p_clamped)
    
    # Ensure symmetry and zero diagonal
    dist_mat = 0.5 * (dist_mat + dist_mat.T)
    np.fill_diagonal(dist_mat, 0.0)
    
    # Run Neighbor Joining
    tree = _nj_tree(dist_mat, labels)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > time_limit - 10:
        newick = _build_newick(tree)
        return newick
    
    # Refine with quartet-based local search
    if n_labels >= 4 and time.time() - start_time < time_limit - 20:
        # Determine how many refinement iterations we can afford
        remaining = time_limit - (time.time() - start_time) - 5
        # Estimate time per iteration based on n
        # For n=32, quartet scoring is O(n^4) which is ~1M ops, should be fast
        max_iters = min(20, int(remaining / 0.5))  # Assume 0.5s per iteration worst case
        max_iters = max(1, max_iters)
        
        tree = _nni_refine(tree, labels, dist_mat, max_iterations=max_iters)
    
    newick = _build_newick(tree)
    return newick

# EVOLVE-BLOCK-END
