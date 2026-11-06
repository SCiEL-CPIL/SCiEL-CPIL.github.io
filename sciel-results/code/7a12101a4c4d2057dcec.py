"""FastUPGMA + NJ hybrid for phylogenetic topology recovery."""
# EVOLVE-BLOCK-START
import math
import numpy as np

def _jc_distances(sequences, labels):
    """Compute Jukes-Cantor corrected distance matrix using numpy."""
    n = len(labels)
    seqs = [sequences[lab] for lab in labels]
    L = len(seqs[0])
    
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arrs = []
    for s in seqs:
        arr = np.array([char_map[c] for c in s], dtype=np.uint8)
        arrs.append(arr)
    
    stacked = np.stack(arrs, axis=0)
    
    # Vectorized pairwise difference
    # stacked shape (n, L)
    # We want sum of stacked[i] != stacked[j]
    # Can use broadcasting: stacked[:, None, :] != stacked[None, :, :]
    # This creates a (n, n, L) boolean array.
    # For n=32, L=2000, size is 32*32*2000 = 2M booleans = 2MB. Fine.
    diffs = (stacked[:, None, :] != stacked[None, :, :]).sum(axis=2)
    
    p = diffs / L
    # Handle saturation
    p = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    
    # Set diagonal to 0
    np.fill_diagonal(d, 0.0)
    
    return d

def _build_nj_tree(dist, labels):
    """
    Build NJ tree and return the root TreeNode.
    """
    n = len(labels)
    if n <= 1:
        return None
    if n == 2:
        # Simple case
        class T:
            def __init__(self, name): self.name = name; self.left=None; self.right=None; self.length=0
        a = T(labels[0]); b = T(labels[1])
        root = T(None)
        a.parent = root; b.parent = root
        root.left = a; root.right = b
        a.length = 0.5*dist[0,1]; b.length = 0.5*dist[0,1]
        return root

    class TreeNode:
        __slots__ = ['name', 'left', 'right', 'length', 'parent']
        def __init__(self, name=None):
            self.name = name
            self.left = None
            self.right = None
            self.length = 0.0
            self.parent = None

        def is_leaf(self):
            return self.name is not None

        def get_leaves(self):
            if self.is_leaf():
                return [self.name]
            leaves = []
            if self.left: leaves.extend(self.left.get_leaves())
            if self.right: leaves.extend(self.right.get_leaves())
            return leaves

    current_nodes = [TreeNode(name=lab) for lab in labels]
    current_dist = dist.copy()
    current_n = n
    current_labels = list(labels)

    while current_n > 3:
        # Compute Q
        r = np.sum(current_dist, axis=1)
        Q = (current_n - 2) * current_dist - np.outer(r, r)
        
        # Find min Q
        min_val = float('inf')
        min_i, min_j = 0, 1
        for i in range(current_n):
            for j in range(i + 1, current_n):
                if Q[i, j] < min_val:
                    min_val = Q[i, j]
                    min_i, min_j = i, j
        
        a, b = min_i, min_j
        
        new_node = TreeNode()
        
        d_ab = current_dist[a, b]
        r_a = r[a]
        r_b = r[b]
        
        len_a = 0.5 * (d_ab + r_a - r_b)
        len_b = 0.5 * (d_ab - r_a + r_b)
        
        node_a = current_nodes[a]
        node_b = current_nodes[b]
        
        node_a.length = len_a
        node_b.length = len_b
        node_a.parent = new_node
        node_b.parent = new_node
        new_node.left = node_a
        new_node.right = node_b
        
        # Update distance matrix
        rest_indices = [k for k in range(current_n) if k not in (a, b)]
        new_nodes = [current_nodes[k] for k in rest_indices] + [new_node]
        new_labels = [current_labels[k] for k in rest_indices] + [f"n{current_n}"]
        new_n = len(new_nodes)
        
        new_dist = np.zeros((new_n, new_n), dtype=np.float64)
        
        for i_idx in range(new_n):
            for j_idx in range(new_n):
                if i_idx == j_idx:
                    continue
                orig_i = rest_indices[i_idx] if i_idx < len(rest_indices) else -1
                orig_j = rest_indices[j_idx] if j_idx < len(rest_indices) else -1
                
                if orig_i == -1 and orig_j == -1:
                    continue
                elif orig_i == -1:
                    new_dist[i_idx, j_idx] = 0.5 * (current_dist[a, orig_j] + current_dist[b, orig_j] - d_ab)
                elif orig_j == -1:
                    new_dist[i_idx, j_idx] = 0.5 * (current_dist[a, orig_i] + current_dist[b, orig_i] - d_ab)
                else:
                    new_dist[i_idx, j_idx] = current_dist[orig_i, orig_j]
        
        current_nodes = new_nodes
        current_labels = new_labels
        current_dist = new_dist
        current_n = new_n
        
    # Final 3 nodes
    # Pick pair with min distance
    min_d = float('inf')
    min_i, min_j = 0, 1
    for i in range(3):
        for j in range(i+1, 3):
            if current_dist[i, j] < min_d:
                min_d = current_dist[i, j]
                min_i, min_j = i, j
                
    a, b = min_i, min_j
    k = 3 - a - b
    
    node_a = current_nodes[a]
    node_b = current_nodes[b]
    node_k = current_nodes[k]
    
    new_node = TreeNode()
    
    d_ab = current_dist[a, b]
    r_a = np.sum(current_dist[a])
    r_b = np.sum(current_dist[b])
    
    len_a = 0.5 * (d_ab + r_a - r_b)
    len_b = 0.5 * (d_ab - r_a + r_b)
    
    node_a.length = len_a
    node_b.length = len_b
    node_a.parent = new_node
    node_b.parent = new_node
    new_node.left = node_a
    new_node.right = node_b
    
    # Connect new_node and node_k
    d_new_k = 0.5 * (current_dist[a, k] + current_dist[b, k] - d_ab)
    
    # Make node_k the root
    node_k.parent = None
    node_k.left = new_node
    node_k.right = None
    new_node.parent = node_k
    new_node.length = d_new_k
    node_k.length = 0.0
    
    return node_k

def _tree_to_newick(node):
    if node is None:
        return ""
    if node.is_leaf():
        return node.name
    left = _tree_to_newick(node.left) if node.left else ""
    right = _tree_to_newick(node.right) if node.right else ""
    return f"({left},{right})"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";" if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"
    
    dist = _jc_distances(sequences, labels)
    root = _build_nj_tree(dist, labels)
    
    if root is None:
        return f"({','.join(labels)});"
        
    newick = _tree_to_newick(root)
    return newick + ";"
# EVOLVE-BLOCK-END
