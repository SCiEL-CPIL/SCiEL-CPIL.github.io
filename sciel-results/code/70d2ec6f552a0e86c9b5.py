"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 265.0  # seconds; leave 5s margin before the 270s kill
_START = time.time()

def _jc_distances(sequences, labels):
    """Compute pairwise JC-corrected distances using NumPy."""
    n = len(labels)
    seqs = [sequences[lab] for lab in labels]
    length = len(seqs[0])
    
    if length == 0:
        return np.zeros((n, n))

    dna_arr = np.array([list(s) for s in seqs])  # (n, length)
    
    # Compute pairwise Hamming distances
    mismatches = np.sum(dna_arr[:, None, :] != dna_arr[None, :, :], axis=2)
    
    # Proportion of differences
    p = mismatches.astype(float) / length
    
    # JC correction: -0.75 * ln(1 - 4p/3)
    # Clip p to avoid log(0) or log(negative)
    p_clipped = np.clip(p, 0, 0.7499)
    dist = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    
    # Set diagonal to 0
    np.fill_diagonal(dist, 0.0)
    
    return dist

def _neighbor_joining(D, labels):
    """
    Correct Neighbor Joining implementation.
    Returns a Newick string.
    """
    n = len(labels)
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    # State:
    # D: (m, m) distance matrix for active nodes
    # trees: list of length m, where trees[i] is the Newick string for the subtree rooted at node i
    #        Initially, trees[i] = labels[i]
    #        When joining i and j into new node k, trees[k] = "(trees[i],trees[j])"
    
    D = D.copy()
    trees = list(labels)
    
    # Map from current index in D/trees to the "original" label is not needed for Newick,
    # but we need to be careful with indices.
    
    m = n
    # We will maintain a list of active indices? 
    # Simpler: just operate on the shrinking arrays directly.
    # However, removing rows/cols from numpy arrays is O(m^2). 
    # Since m <= 32, this is negligible.
    
    while m > 3:
        if time.time() > _DEADLINE:
            # Fallback: return a star tree of current active labels
            # The current 'trees' list contains the Newick strings for the current active nodes.
            # We need to combine them into a valid Newick string.
            # A simple star: (tree0, tree1, ..., tree(m-1));
            return "(" + ",".join(trees) + ");"
            
        # Compute row sums
        r = np.sum(D, axis=1)
        
        # Compute Q matrix
        # Q[i, j] = (m - 2) * D[i, j] - r[i] - r[j]
        Q = (m - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        # Find min Q
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        if i == j:
            # Should not happen if diagonal is inf, but safety check
            continue
        if i > j:
            i, j = j, i
            
        # Determine indices of other nodes
        others = [k for k in range(m) if k != i and k != j]
        
        # Compute new branch lengths and distance from new node to others
        d_ij = D[i, j]
        
        # Distance from new node (k) to others
        if others:
            d_i_others = D[i, others]
            d_j_others = D[j, others]
            d_new_others = 0.5 * (d_i_others + d_j_others - d_ij)
        else:
            d_new_others = np.array([])
        
        # Construct new tree string for the joined node
        new_tree_str = f"({trees[i]},{trees[j]})"
        
        # Construct new distance matrix D_new
        # Size: (len(others) + 1) x (len(others) + 1)
        # The new node will be at the last index.
        new_m = len(others) + 1
        D_new = np.zeros((new_m, new_m))
        
        if others:
            # Fill distances among others
            D_new[:len(others), :len(others)] = D[np.ix_(others, others)]
            
            # Fill distances from new node to others
            D_new[:len(others), len(others)] = d_new_others
            D_new[len(others), :len(others)] = d_new_others
            # D_new[len(others), len(others)] = 0.0 (already zero)
            
            # Update trees list: remove i, j, add new_tree_str at end
            # Order in 'others' must match the order in D_new rows/cols 0..len(others)-1
            new_trees = [trees[k] for k in others]
            new_trees.append(new_tree_str)
        else:
            # This case implies m was 4, and we joined 2, leaving 1 other? No, m>3.
            # If m=4, others has 2 elements.
            # If m=3, loop doesn't run.
            # So others is never empty if m > 3.
            new_trees = [new_tree_str]
            D_new = np.array([[0.0]])
            
        D = D_new
        trees = new_trees
        m = new_m
        
    # Handle the final 3 nodes
    # We have 3 nodes in 'trees' and a 3x3 distance matrix D.
    # We can just return a star tree of these 3 subtrees, or resolve the triplet.
    # A star tree (a,b,c) is topologically equivalent to any unrooted tree with 3 leaves.
    # So returning a star is correct for topology.
    return "(" + ",".join(trees) + ");"

def reconstruct(sequences):
    global _START, _DEADLINE
    
    _START = time.time()
    _DEADLINE = _START + 265.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    # Compute distance matrix
    D = _jc_distances(sequences, labels)
    
    # Run NJ
    newick = _neighbor_joining(D, labels)
    
    return newick
# EVOLVE-BLOCK-END
