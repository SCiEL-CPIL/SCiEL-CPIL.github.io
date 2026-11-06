"""Phylogenetic topology recovery using Neighbor Joining with NNI refinement."""
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

def _get_leaves(tree):
    """Get all leaf labels from a tree structure."""
    leaves = []
    def collect(node):
        if isinstance(node, str):
            leaves.append(node)
        else:
            collect(node[0])
            collect(node[1])
    collect(tree)
    return leaves

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
    
    nodes = list(range(n_labels))
    D = dist_mat.copy()
    
    tree_struct = {i: labels[i] for i in range(n_labels)}
    next_id = n_labels
    
    while len(nodes) > 3:
        n = len(nodes)
        
        r = D.sum(axis=1)
        Q = (n - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        flat_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = flat_idx
        if i == j:
            continue
        
        node_i = nodes[i]
        node_j = nodes[j]
        
        new_node = next_id
        next_id += 1
        tree_struct[new_node] = (tree_struct[node_i], tree_struct[node_j])
        
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[i] = False
        keep_mask[j] = False
        kept_indices = np.where(keep_mask)[0]
        
        new_nodes_list = [nodes[k] for k in kept_indices]
        new_nodes_list.append(new_node)
        
        new_n = n - 1
        D_new = np.zeros((new_n, new_n), dtype=np.float64)
        
        D_keep_keep = D[np.ix_(kept_indices, kept_indices)]
        D_new[:-1, :-1] = D_keep_keep
        
        dist_i_kept = D[kept_indices, i]
        dist_j_kept = D[kept_indices, j]
        dist_ij = D[i, j]
        
        new_branches = 0.5 * (dist_i_kept + dist_j_kept - dist_ij)
        
        D_new[:-1, -1] = new_branches
        D_new[-1, :-1] = new_branches
        
        nodes = new_nodes_list
        D = D_new
    
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

def _mean_dist_between_subtrees(tree1, tree2, dist_mat, labels):
    """
    Calculate the mean distance between all pairs of leaves in tree1 and tree2.
    """
    leaves1 = _get_leaves(tree1)
    leaves2 = _get_leaves(tree2)
    
    if not leaves1 or not leaves2:
        return 0.0
        
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    total = 0.0
    count = 0
    for l1 in leaves1:
        i1 = label_to_idx[l1]
        for l2 in leaves2:
            i2 = label_to_idx[l2]
            total += dist_mat[i1, i2]
            count += 1
            
    return total / count if count > 0 else 0.0

def _nni_refine(tree, labels, dist_mat, max_rounds=50):
    """
    NNI refinement using quartet cost minimization.
    """
    n = len(labels)
    if n < 4:
        return tree
    
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    # We work on a copy of the tree
    current_tree = tree
    
    # Helper to find internal-internal edges and their subtrees
    # Since we use nested tuples, we need to traverse.
    # This is tricky with tuples. 
    # Alternative: Convert to a Node-based structure for NNI, then back to tuples?
    # Or, simpler: Iterate over all possible quartets of leaves? 
    # For n=32, C(32,4) = 35960. 
    # For each quartet, check if current tree implies the best topology.
    # If not, try to fix it? 
    # But "fixing" a quartet in a tuple tree is hard.
    
    # Let's stick to the Edge-based NNI but implement it on a Node class.
    
    class Node:
        __slots__ = ['label', 'left', 'right', 'parent']
        def __init__(self, label=None, left=None, right=None):
            self.label = label
            self.left = left
            self.right = right
            self.parent = None

    def tuple_to_node(tree):
        if isinstance(tree, str):
            return Node(label=tree)
        left, right = tree
        node = Node()
        node.left = tuple_to_node(left)
        node.right = tuple_to_node(right)
        node.left.parent = node
        node.right.parent = node
        return node

    def node_to_tuple(node):
        if node.label is not None:
            return node.label
        return (node_to_tuple(node.left), node_to_tuple(node.right))

    # Convert to Node structure
    root = tuple_to_node(current_tree)
    
    # Root at an arbitrary leaf to establish parent pointers
    # Find a leaf
    def find_leaf(node):
        if node.label is not None:
            return node
        return find_leaf(node.left)
        
    root_leaf = find_leaf(root)
    # The tree is already rooted by the tuple structure. 
    # The "root" of the tuple is an internal node.
    # We need to ensure parent pointers are set correctly for the whole tree.
    # tuple_to_node sets parent for children. 
    # The root of the tuple has no parent.
    # This is fine. We treat the tuple root as the "root" of the rooted tree.
    # But for unrooted NNI, we need to consider the "virtual" parent of the root?
    # No, the root of the tuple represents an unrooted tree.
    # The top-most internal node in the tuple has 2 children.
    # In the unrooted tree, it has 3 neighbors: child1, child2, and "the rest".
    # But in the tuple, it only has 2 neighbors.
    # So NNI on the edge below the root is not directly accessible?
    # Actually, the root of the tuple IS an internal node.
    # Its neighbors are left and right.
    # The "third" neighbor is the parent, which is None.
    # So we can only do NNI on edges where both nodes have parents?
    # No, we can do NNI on the edge between the root and its child?
    # If we do NNI on edge (root, left_child):
    #   Subtrees: left_child.left, left_child.right, root.right, and "parent of root" (which is None/empty).
    #   This is invalid.
    
    # To handle the root properly, we should "unroot" the tree or add a dummy parent.
    # Standard trick: Add a dummy leaf connected to the root?
    # Or, simply, iterate over all internal nodes $u$ where $u.parent$ is not None.
    # This misses the edges incident to the root.
    # However, NJ trees are usually balanced enough that the root edges are less critical,
    # or we can rotate the tree to move the root to a leaf, making all internal nodes have parents.
    
    # Let's rotate the tree so the root is a leaf.
    # This makes the structure a "rooted" tree where the root is a leaf.
    # Then every internal node has a parent (except the one adjacent to the root leaf, which has the root leaf as parent).
    # Wait, if root is a leaf, the "top" internal node has the root leaf as parent.
    # So ALL internal nodes have parents.
    # And the edge between two internal nodes is always between a node and its parent.
    
    # How to rotate?
    # Pick a leaf L. Make L the root.
    # This requires changing the structure.
    # Given the complexity, let's just use the tuple structure and accept that we might miss the top 1-2 edges.
    # Or, better: Use the Quartet Score on all quartets to guide a random restart or hill climbing?
    
    # Let's try a different approach: **Fast Quartet Picking (FQP) logic**.
    # 1. Start with NJ tree.
    # 2. Compute all quartet scores for the current tree.
    # 3. Find the quartet with the worst score (disagreement with distance data).
    # 4. Perform an NNI move that fixes this quartet.
    # 5. Repeat.
    
    # To perform NNI, we still need the tree structure.
    # Let's implement a robust NNI on the Node structure, handling the root by adding a dummy node.
    
    # Add dummy parent to root
    dummy = Node(label="DUMMY")
    dummy.left = root
    root.parent = dummy
    # dummy.right = None? No, dummy is just a helper.
    # We will iterate over all internal nodes u where u.parent is internal.
    
    internal_nodes = []
    def collect_internal(node):
        if node.label is not None:
            return
        if node.parent is not None and node.parent.label is not None:
             # node is child of a leaf? No, we rooted at a leaf?
             # We didn't root at a leaf. The tuple root is internal.
             # Its parent is dummy (internal).
             # So root is a candidate.
             pass
        internal_nodes.append(node)
        if node.left: collect_internal(node.left)
        if node.right: collect_internal(node.right)
        
    collect_internal(root)
    
    # Filter: u must be internal, u.parent must be internal.
    candidates = [u for u in internal_nodes if u.parent is not None and u.parent.label is None]
    
    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        
        for u in candidates:
            p = u.parent
            # u and p are internal.
            # u's children: u.left, u.right
            # p's children: p.left, p.right. One is u. The other is sib.
            sib = p.right if p.left == u else p.left
            
            # The 4 subtrees
            T1 = u.left
            T2 = u.right
            T3 = sib
            # T4 is the "rest" of the tree connected to p's parent.
            # If p.parent is dummy, T4 is empty?
            # If p.parent is a leaf, T4 is that leaf.
            # If p.parent is internal, T4 is that internal node's subtree.
            # Wait, if p.parent is internal, then the edge (p, p.parent) is also a candidate.
            # For the edge (u, p), the 4th subtree is the component containing p.parent.
            # This component is rooted at p.parent.
            T4 = p.parent
            
            # Check if T4 is valid (not None, not dummy)
            if T4 is None or T4 is dummy:
                continue
                
            # Calculate costs
            # D13, D14, D23, D24 for topology ((T1,T2),(T3,T4)) -> Cost = D13+D14+D23+D24
            # D12, D14, D32, D34 for topology ((T1,T3),(T2,T4)) -> Cost = D12+D14+D32+D34
            # D12, D13, D42, D43 for topology ((T1,T4),(T2,T3)) -> Cost = D12+D13+D42+D43
            
            D12 = _mean_dist_between_subtrees(T1, T2, dist_mat, labels)
            D13 = _mean_dist_between_subtrees(T1, T3, dist_mat, labels)
            D14 = _mean_dist_between_subtrees(T1, T4, dist_mat, labels)
            D23 = _mean_dist_between_subtrees(T2, T3, dist_mat, labels)
            D24 = _mean_dist_between_subtrees(T2, T4, dist_mat, labels)
            D34 = _mean_dist_between_subtrees(T3, T4, dist_mat, labels)
            
            cost_current = D13 + D14 + D23 + D24
            cost_alt1 = D12 + D14 + D23 + D34 # Wait, ((T1,T3),(T2,T4)) -> Cross terms are D12, D14, D32, D34?
            # Let's re-verify.
            # Topology ((A,B),(C,D)). Cost = dist(A,C)+dist(A,D)+dist(B,C)+dist(B,D).
            # Here A=T1, B=T2, C=T3, D=T4.
            # Current: ((T1,T2),(T3,T4)). Cost = D13 + D14 + D23 + D24.
            # Alt 1: ((T1,T3),(T2,T4)). Cost = D12 + D14 + D32 + D34. (D32=D23).
            # Alt 2: ((T1,T4),(T2,T3)). Cost = D12 + D13 + D42 + D43. (D42=D24, D43=D34).
            
            cost_alt1 = D12 + D14 + D23 + D34
            cost_alt2 = D12 + D13 + D24 + D34
            
            best_cost = min(cost_current, cost_alt1, cost_alt2)
            
            if best_cost < cost_current - 1e-10:
                # Perform NNI
                if best_cost == cost_alt1:
                    # New topology: ((T1, T3), (T2, T4))
                    # We need to restructure the tree.
                    # p currently has children u and sib(T3).
                    # u has children T1 and T2.
                    # New structure:
                    # p_new has children N13 and N24.
                    # N13 has children T1 and T3.
                    # N24 has children T2 and T4.
                    
                    # But T4 is p.parent. This is the tricky part.
                    # We are changing the topology around edge (u,p).
                    # The "root" of this local change is p.
                    # p's parent is T4.
                    # So T4 remains the parent of the new root of this cluster.
                    
                    N13 = Node()
                    N13.left = T1
                    N13.right = T3
                    T1.parent = N13
                    T3.parent = N13
                    
                    N24 = Node()
                    N24.left = T2
                    N24.right = T4 # Wait, T4 is the parent of p.
                    # This means we are moving T4?
                    # No, the unrooted tree topology is changing.
                    # The edge (u,p) is removed.
                    # New edges: (T1,T3), (T2,T4), and the new internal node connects them.
                    
                    # Let's create a new node N_root for the cluster.
                    N_root = Node()
                    N_root.left = N13
                    N_root.right = N24
                    N13.parent = N_root
                    N24.parent = N_root
                    
                    # Connect N_root to the rest of the tree.
                    # The rest of the tree was connected to p via T4 (p.parent).
                    # So N_root should replace p in the tree.
                    # p.parent is T4.
                    # So N_root.parent = T4.
                    # And T4's child (which was p) should now be N_root.
                    
                    # We need to update T4's child pointer.
                    if T4.left == p:
                        T4.left = N_root
                    elif T4.right == p:
                        T4.right = N_root
                        
                    N_root.parent = T4
                    
                    # Update candidates list?
                    # The tree structure has changed.
                    # We should restart the scan or update the list.
                    # For simplicity, break and restart the while loop.
                    improved = True
                    break
                elif best_cost == cost_alt2:
                    # New topology: ((T1, T4), (T2, T3))
                    N14 = Node()
                    N14.left = T1
                    N14.right = T4
                    T1.parent = N14
                    T4.parent = N14 # Wait, T4 is p.parent.
                    
                    N23 = Node()
                    N23.left = T2
                    N23.right = T3
                    T2.parent = N23
                    T3.parent = N23
                    
                    N_root = Node()
                    N_root.left = N14
                    N_root.right = N23
                    N14.parent = N_root
                    N23.parent = N_root
                    
                    if T4.left == p:
                        T4.left = N_root
                    elif T4.right == p:
                        T4.right = N_root
                        
                    N_root.parent = T4
                    improved = True
                    break
                    
        if improved:
            # Re-collect internal nodes
            internal_nodes = []
            def collect_internal2(node):
                if node.label is not None:
                    return
                internal_nodes.append(node)
                if node.left: collect_internal2(node.left)
                if node.right: collect_internal2(node.right)
            collect_internal2(root)
            candidates = [u for u in internal_nodes if u.parent is not None and u.parent.label is None]

    # Convert back to tuple
    return node_to_tuple(root)

def reconstruct(sequences):
    start_time = time.time()
    time_limit = 265.0
    
    labels = sorted(sequences.keys(), key=lambda x: int(x[1:]))
    n_labels = len(labels)
    
    if n_labels < 2:
        return labels[0] + ";" if labels else ";"
    if n_labels == 2:
        return f"({labels[0]},{labels[1]})"
    if n_labels == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}))"
    
    L = len(next(iter(sequences.values())))
    char_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    seq_mat = np.zeros((n_labels, L), dtype=np.int8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j, base in enumerate(seq):
            seq_mat[i, j] = char_to_idx[base]
    
    diff = (seq_mat[:, None, :] != seq_mat[None, :, :]).astype(np.int8)
    hamming = diff.sum(axis=2)
    
    p = hamming.astype(np.float64) / L
    p_clamped = np.clip(p, 0.0, 0.7499999)
    dist_mat = -0.75 * np.log(1.0 - (4.0/3.0) * p_clamped)
    
    dist_mat = 0.5 * (dist_mat + dist_mat.T)
    np.fill_diagonal(dist_mat, 0.0)
    
    tree = _nj_tree(dist_mat, labels)
    
    elapsed = time.time() - start_time
    if elapsed < time_limit - 10:
        tree = _nni_refine(tree, labels, dist_mat, max_rounds=20)
        
    newick = _build_newick(tree)
    return newick
