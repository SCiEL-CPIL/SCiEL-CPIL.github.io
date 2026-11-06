import numpy as np
from collections import defaultdict

def _char_to_mask(c):
    if c == 'A': return 1
    if c == 'C': return 2
    if c == 'G': return 4
    if c == 'T': return 8
    return 15 # N

def _build_alignment(sequences, labels):
    n_sites = len(next(iter(sequences.values())))
    n_taxa = len(labels)
    # labels are sorted indices
    # Create array of shape (n_sites, n_taxa)
    align = np.zeros((n_sites, n_taxa), dtype=np.uint8)
    for j, label in enumerate(labels):
        seq = sequences[label]
        for i in range(n_sites):
            align[i, j] = _char_to_mask(seq[i])
    return align

def _fitch(node, align):
    """
    Returns (total_cost, state_array)
    state_array is uint8 array of shape (L,)
    """
    if isinstance(node, int):
        state = align[:, node]
        cost = np.zeros(align.shape[0], dtype=np.int32)
        return cost, state
    else:
        left_cost, left_state = _fitch(node[0], align)
        right_cost, right_state = _fitch(node[1], align)
        
        inter = left_state & right_state
        union = left_state | right_state
        
        # Cost increases where intersection is empty
        # (inter == 0) is boolean array
        inc = (inter == 0).astype(np.int32)
        
        new_cost = left_cost + right_cost + inc
        new_state = np.where(inter != 0, inter, union)
        
        return new_cost, new_state

def _get_parity_score(tree, align):
    cost, _ = _fitch(tree, align)
    return cost.sum()

def _expand_tree(node, new_idx):
    """
    Generator that yields all possible trees formed by inserting new_idx into node.
    """
    if isinstance(node, int):
        # A leaf cannot be expanded internally, but the edge above it is handled by parent.
        # If this is the root (which shouldn't be a leaf in our logic), we can't insert.
        # But in our recursive call, we only call this on internal nodes or handle leaves in parent.
        # Wait, if node is a leaf, we don't yield anything here.
        # The parent will handle inserting as sibling to this leaf.
        return
    
    left, right = node
    
    # 1. Insert as sibling to left
    yield ( (new_idx, left), right )
    
    # 2. Insert as sibling to right
    yield ( left, (new_idx, right) )
    
    # 3. Recurse into left
    for new_left in _expand_tree(left, new_idx):
        yield ( new_left, right )
        
    # 4. Recurse into right
    for new_right in _expand_tree(right, new_idx):
        yield ( left, new_right )

def _to_newick(node, labels):
    if isinstance(node, int):
        return labels[node]
    else:
        left, right = node
        return f"({_to_newick(left, labels)},{_to_newick(right, labels)})"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 0:
        return ";"
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Build alignment
    align = _build_alignment(sequences, labels)
    
    # Initialize tree with first 3 taxa
    # Tree structure: (0, (1, 2))
    current_tree = (0, (1, 2))
    
    # Add remaining taxa
    for idx in range(3, n):
        best_score = float('inf')
        best_tree = current_tree
        
        for candidate in _expand_tree(current_tree, idx):
            score = _get_parity_score(candidate, align)
            if score < best_score:
                best_score = score
                best_tree = candidate
        
        current_tree = best_tree
        
    return _to_newick(current_tree, labels) + ";"
