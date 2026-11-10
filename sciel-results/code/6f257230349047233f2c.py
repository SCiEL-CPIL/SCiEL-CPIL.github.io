import math
import time
import numpy as np
from itertools import combinations

def _jc_distance_matrix(sequences, labels):
    """Compute JC-corrected distance matrix using vectorized operations."""
    n = len(labels)
    # Convert sequences to integer arrays
    # Map A,C,G,T to 0,1,2,3
    # We can use a lookup table for speed
    char_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Create a matrix of integers (n x L)
    L = len(sequences[labels[0]])
    seq_matrix = np.zeros((n, L), dtype=np.int8)
    
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j in range(L):
            seq_matrix[i, j] = char_to_int.get(seq[j], 0)
            
    # Compute pairwise Hamming distances
    # diff[i, j] = number of positions where seq_matrix[i] != seq_matrix[j]
    # We can do this in chunks or use broadcasting, but n is small.
    dist_mat = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Vectorized comparison
            diff = np.sum(seq_matrix[i] != seq_matrix[j])
            p = diff / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist_mat[i, j] = d
            dist_mat[j, i] = d
            
    return dist_mat

def _nj_tree(dist_mat, labels):
    """
    Perform Neighbour Joining on a distance matrix.
    Returns a nested tuple structure representing the tree.
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return (labels[0], labels[1])
    
    # Active nodes: start with all leaves (0 to n-1)
    active = list(range(n))
    # Tree structure: dict mapping node_id -> (child_a, child_b)
    # Leaves are just labels, internal nodes are tuples
    # We will store the tree structure in a dict where key is node id
    # Node ids: 0..n-1 are leaves, n..2n-2 are internal
    tree_nodes = {i: labels[i] for i in range(n)}
    next_id = n
    
    # Distance matrix for current active nodes
    # We maintain a dict of dicts or a 2D array. Since active nodes change, 
    # a dict of dicts is safer and n is small.
    dist_dict = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_dict[(i, j)] = dist_mat[i, j]
    
    while len(active) > 3:
        k = len(active)
        
        # Compute r[i] = sum of distances from node i to all other active nodes
        r = {}
        for node in active:
            r[node] = sum(dist_dict[(node, other)] for other in active if other != node)
        
        # Compute Q matrix and find best pair
        best_q = float('inf')
        best_a, best_b = -1, -1
        
        for idx_a in range(k):
            for idx_b in range(idx_a + 1, k):
                node_a = active[idx_a]
                node_b = active[idx_b]
                d_ab = dist_dict[(node_a, node_b)]
                q = (k - 2) * d_ab - r[node_a] - r[node_b]
                if q < best_q:
                    best_q = q
                    best_a, best_b = node_a, node_b
        
        # Create new internal node
        new_node = next_id
        next_id += 1
        tree_nodes[new_node] = (best_a, best_b)
        
        # Compute distances from new node to all other active nodes
        for other in active:
            if other == best_a or other == best_b:
                continue
            d_a_other = dist_dict[(best_a, other)]
            d_b_other = dist_dict[(best_b, other)]
            d_ab = dist_dict[(best_a, best_b)]
            d_new_other = 0.5 * (d_a_other + d_b_other - d_ab)
            if d_new_other < 0:
                d_new_other = 0.0
            dist_dict[(new_node, other)] = d_new_other
            dist_dict[(other, new_node)] = d_new_other
        
        # Remove old nodes, add new node
        active = [x for x in active if x != best_a and x != best_b]
        active.append(new_node)
        
    # Handle final 3 nodes
    u, v, w = active
    # The final three nodes form a star. We need to pick a root for the star.
    # Usually, we group two and then add the third.
    # Topology: ((u, v), w)
    node_1 = next_id
    next_id += 1
    tree_nodes[node_1] = (u, v)
    
    node_2 = next_id
    next_id += 1
    tree_nodes[node_2] = (node_1, w)
    
    return tree_nodes, node_2

def _tree_to_splits(tree_nodes, root, labels):
    """
    Extract all non-trivial splits from the tree.
    Returns a set of frozensets.
    """
    splits = set()
    
    def get_leaves(node):
        """Recursively get the set of leaf labels under a node."""
        if node < len(labels):
            return frozenset([labels[node]])
        else:
            left, right = tree_nodes[node]
            return get_leaves(left) | get_leaves(right)
    
    def collect_splits(node, all_labels_set):
        if node < len(labels):
            return
        left, right = tree_nodes[node]
        left_leaves = get_leaves(left)
        right_leaves = get_leaves(right)
        
        # A split is defined by the set of leaves in one subtree.
        # We only add it if it's non-trivial (size > 0 and < n)
        if 0 < len(left_leaves) < len(all_labels_set):
            splits.add(left_leaves)
        
        collect_splits(left, all_labels_set)
        collect_splits(right, all_labels_set)
        
    all_labels_set = frozenset(labels)
    collect_splits(root, all_labels_set)
    
    return splits

def _nni_refine(sequences, labels, tree_nodes, root, time_limit):
    """
    Perform NNI refinement based on quartet support.
    Returns the improved tree_nodes and root.
    """
    n = len(labels)
    if n < 4:
        return tree_nodes, root
        
    start_time = time.time()
    
    # We need to be able to identify internal edges.
    # An internal edge connects two internal nodes.
    # In our tree structure, each internal node has 2 children.
    # An NNI move involves an internal edge (u, v) where u is parent of v (or vice versa)
    # and both u and v are internal nodes.
    
    # To simplify, we will iterate over all pairs of internal nodes that are connected.
    # We need to build a parent-child map.
    
    def build_parent_map(tree_nodes, root):
        parent_map = {root: None}
        def dfs(node):
            if node < len(labels):
                return
            left, right = tree_nodes[node]
            parent_map[left] = node
            parent_map[right] = node
            dfs(left)
            dfs(right)
        dfs(root)
        return parent_map
    
    parent_map = build_parent_map(tree_nodes, root)
    
    # Identify internal edges: (child, parent) where both are internal
    internal_edges = []
    for child, parent in parent_map.items():
        if parent is None:
            continue
        if child >= len(labels) and parent >= len(labels):
            internal_edges.append((child, parent))
            
    # Helper to get the 4 subtrees attached to an internal edge (u, v)
    # u is child, v is parent.
    # u has children u1, u2.
    # v has children u, v2.
    # The 4 subtrees are: u1, u2, v2, and the rest of the tree attached to v (which is not u).
    # Wait, the standard NNI is on the 4 subtrees attached to the two internal nodes.
    # Let u be the node closer to leaves, v be the node closer to root.
    # Children of u: a, b
    # Children of v: u, c
    # The 4 subtrees are a, b, c, and the subtree attached to v (let's call it d, which is the rest of the tree).
    # Actually, v is an internal node, so it has 2 children: u and c.
    # The "rest of the tree" is not a subtree in the same sense, it's the parent of v.
    # Let's rephrase:
    # Edge (u, v). u is internal, v is internal.
    # u has children a, b.
    # v has children u, c.
    # The 4 groups are:
    # 1. Leaves under a
    # 2. Leaves under b
    # 3. Leaves under c
    # 4. Leaves under the parent of v (excluding the path to u) -- this is complicated.
    
    # Easier: The 4 subtrees are the leaves under the 4 neighbors of the edge (u, v).
    # Neighbors of u: a, b, v
    # Neighbors of v: u, c, parent(v)
    # The 4 subtrees are:
    # S1: leaves(a)
    # S2: leaves(b)
    # S3: leaves(c)
    # S4: leaves(parent(v)) - leaves(v)  ... No, this is getting messy.
    
    # Let's use the standard definition:
    # Consider the edge between internal nodes u and v.
    # Let u be the child, v be the parent.
    # u has children u1, u2.
    # v has children u, v1.
    # The 4 subtrees are:
    # A = leaves(u1)
    # B = leaves(u2)
    # C = leaves(v1)
    # D = leaves(parent(v)) \ leaves(v)  -- i.e., the rest of the tree.
    
    # Actually, it's easier to just define the 4 sets of taxa:
    # Set 1: Taxa in u1
    # Set 2: Taxa in u2
    # Set 3: Taxa in v1
    # Set 4: All other taxa
    
    def get_taxa_set(node):
        if node < len(labels):
            return frozenset([labels[node]])
        left, right = tree_nodes[node]
        return get_taxa_set(left) | get_taxa_set(right)
    
    all_taxa = frozenset(labels)
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit - 10:
            break
        improved = False
        parent_map = build_parent_map(tree_nodes, root)
        internal_edges = [(c, p) for c, p in parent_map.items() if p is not None and c >= len(labels) and p >= len(labels)]
        
        for u, v in internal_edges:
            # u is child, v is parent
            u1, u2 = tree_nodes[u]
            # v's children are u and v1
            v_children = tree_nodes[v]
            if v_children[0] == u:
                v1 = v_children[1]
            else:
                v1 = v_children[0]
                
            # Get the 4 sets of taxa
            set_u1 = get_taxa_set(u1)
            set_u2 = get_taxa_set(u2)
            set_v1 = get_taxa_set(v1)
            set_rest = all_taxa - (set_u1 | set_u2 | set_v1)
            
            if len(set_rest) == 0:
                continue # Should not happen for n>=4
                
            # The 3 possible quartet topologies for these 4 sets:
            # 1. (u1, u2) | (v1, rest)  -> Current topology
            # 2. (u1, v1) | (u2, rest)
            # 3. (u1, rest) | (u2, v1)
            
            # We need to count the support for each.
            # Support for split (A, B) | (C, D) is the number of sites where
            # the pattern is consistent with this split.
            # In JC model, a site is informative if it has 2 distinct characters, each appearing twice.
            # Or more generally, we can use the likelihood or a simple distance-based score.
            # Let's use a simple heuristic: count the number of sites that are "compatible" with the split.
            # A site is compatible with split (A,B)|(C,D) if the characters in A and B are the same, 
            # and the characters in C and D are the same, and the two groups are different.
            
            def count_support(set_a, set_b, set_c, set_d):
                # Map labels to indices
                idx_a = [labels.index(x) for x in set_a]
                idx_b = [labels.index(x) for x in set_b]
                idx_c = [labels.index(x) for x in set_c]
                idx_d = [labels.index(x) for x in set_d]
                
                # We need to access sequences by index.
                # Let's precompute the sequence matrix again? No, we can pass it.
                # For efficiency, we should pass the sequence matrix.
                pass
            
            # This approach is too slow in Python if we do it for every NNI move.
            # Let's stick to the distance-based NNI or just return the NJ tree.
            # Given the time constraints and the complexity of implementing a fast quartet counter in Python,
            # I will rely on the robustness of the NJ tree and a simple check if we have time.
            # For now, I will just return the NJ tree to ensure correctness and avoid bugs in the refinement.
            
            break # Exit refinement loop for now
            
    return tree_nodes, root

def _tree_to_newick(tree_nodes, root, labels):
    """Convert tree structure to Newick string."""
    def get_newick(node):
        if node < len(labels):
            return labels[node]
        left, right = tree_nodes[node]
        return f"({get_newick(left)},{get_newick(right)})"
    return get_newick(root) + ";"

def reconstruct(sequences):
    start_time = time.time()
    time_limit = 260.0  # Safety margin
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 1:
            return f"{labels[0]};"
        else:
            return f"({labels[0]},{labels[1]})"
            
    # Compute distance matrix
    dist_mat = _jc_distance_matrix(sequences, labels)
    
    # Use NJ for the initial tree
    tree_nodes, root = _nj_tree(dist_mat, labels)
    
    elapsed = time.time() - start_time
    
    # If we have time, we could try refinement, but for now, we return the NJ tree.
    # The NJ tree is a strong baseline.
    
    return _tree_to_newick(tree_nodes, root, labels)
