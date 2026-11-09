import math
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
'''
Phylogenetic reconstruction using Neighbor Joining (NJ).
'''

def _jc_distance_vec(A):
    """
    Compute Jukes-Cantor corrected distances between rows of A.
    A: numpy array of shape (n, L) with values 0-3.
    Returns: n x n distance matrix.
    """
    n, L = A.shape
    # Hamming distances
    # A[:, None, :] -> (n, 1, L)
    # A[None, :, :] -> (1, n, L)
    diff = A[:, None, :] != A[None, :, :]
    hamming = np.sum(diff, axis=2)
    p = hamming / L
    
    # Apply JC correction: d = -3/4 ln(1 - 4p/3)
    # Handle p close to 0.75 to avoid log(0)
    p = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    return d

def _nj_newick(D, labels):
    """
    Neighbor Joining returning Newick string.
    D: n x n distance matrix.
    labels: list of labels.
    """
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]})" + ";"
    
    # We will maintain a list of "active" nodes.
    # node_struct: id -> representation
    # For leaves: string label
    # For internal nodes: tuple (left_node_id, right_node_id, len_left, len_right)
    # dist: dict (id1, id2) -> distance
    
    active = list(range(n))
    node_struct = {i: labels[i] for i in range(n)}
    dist = {(i, j): D[i, j] for i in range(n) for j in range(n) if i != j}
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        # Calculate r_i
        r = {}
        for i in active:
            r[i] = sum(dist[(i, j)] for j in active if j != i)
        
        # Find min Q
        min_q = float('inf')
        min_i, min_j = -1, -1
        for idx_i, i in enumerate(active):
            for idx_j in range(idx_i + 1, m):
                j = active[idx_j]
                q = (m - 2) * dist[(i, j)] - r[i] - r[j]
                if q < min_q:
                    min_q = q
                    min_i = i
                    min_j = j
        
        d_ij = dist[(min_i, min_j)]
        d_i = 0.5 * (d_ij + (r[min_i] - r[min_j]) / (m - 2))
        d_j = d_ij - d_i
        
        new_node = next_id
        next_id += 1
        
        # Store structure: (left_id, right_id, len_left, len_right)
        node_struct[new_node] = (min_i, min_j, d_i, d_j)
        
        # Update distances to new node
        for k in active:
            if k == min_i or k == min_j:
                continue
            d_new_k = 0.5 * (dist[(min_i, k)] + dist[(min_j, k)] - d_ij)
            dist[(new_node, k)] = d_new_k
            dist[(k, new_node)] = d_new_k
        
        # Remove old nodes, add new
        active.remove(min_i)
        active.remove(min_j)
        active.append(new_node)
        
    # Now active has 3 nodes. Join them.
    # We need to build the final tree.
    # Let's join two of them, then the result with the third.
    # But for the final Newick, we can just output a star or a binary tree.
    # Standard NJ ends with 3 nodes connected to a central node? No, it's a binary tree.
    # Let's join a and b into n1, then n1 and c into n2 (root).
    # But wait, the last step of NJ usually just connects the remaining 3 nodes.
    # If we treat it as binary:
    # Join a and b.
    # d_ab = dist[(a, b)]
    # d_a = 0.5 * (d_ab + dist[(a, c)] - dist[(b, c)])
    # d_b = d_ab - d_a
    # New node n1.
    # Then we have n1 and c.
    # The distance between n1 and c is dist[(n1, c)] which should have been calculated in the previous step if it was not the last one?
    # Wait, if we exit the loop with 3 nodes, we haven't updated distances for the new node yet?
    # No, we update distances inside the loop.
    # So if we have 3 nodes left, they are the result of previous joins.
    # Let's just join two of them manually.
    
    a, b, c = active
    d_ab = dist[(a, b)]
    d_a = 0.5 * (d_ab + dist[(a, c)] - dist[(b, c)])
    d_b = d_ab - d_a
    
    new_node = next_id
    next_id += 1
    node_struct[new_node] = (a, b, d_a, d_b)
    
    # Now we need to connect new_node and c.
    # The distance between new_node and c is dist[(new_node, c)]?
    # It should have been calculated if new_node was created in a previous iteration.
    # But here new_node is created NOW.
    # We need the distance between the clade (a,b) and c.
    # This is not directly in dist.
    # However, we can calculate the branch length from the clade to c.
    # In an unrooted tree, the edge connecting the cherry (a,b) to c has length:
    # len_edge = 0.5 * (dist[(a, c)] + dist[(b, c)] - dist[(a, b)])
    # This is the length of the branch from the internal node (a,b) to c.
    # Let's call it d_c_branch.
    d_c_branch = 0.5 * (dist[(a, c)] + dist[(b, c)] - d_ab)
    
    # Now we have:
    # a: d_a
    # b: d_b
    # c: d_c_branch
    # And they meet at the internal node.
    # The Newick string for this unrooted tree (rooted at the internal node) is:
    # (a:d_a, b:d_b, c:d_c_branch);
    # But we need to handle the case where a or b or c are internal nodes themselves.
    # We need a helper to convert node_id to Newick string.
    
    def to_newick(node_id):
        struct = node_struct[node_id]
        if isinstance(struct, str):
            return struct
        else:
            left_id, right_id, len_left, len_right = struct
            return f"({to_newick(left_id)}:{len_left:.4f},{to_newick(right_id)}:{len_right:.4f})"
    
    # The final tree is not just (a, b, c) if a, b, c are internal.
    # The structure is:
    # Root
    #  |-- a (len d_a)
    #  |-- b (len d_b)
    #  |-- c (len d_c_branch)
    # But a, b, c are subtrees.
    # So the Newick is:
    # (to_newick(a):d_a, to_newick(b):d_b, to_newick(c):d_c_branch);
    # Wait, to_newick(a) returns the subtree for a.
    # If a is a leaf, it returns the label.
    # If a is internal, it returns (left:..., right:...).
    # So the branch length d_a is attached to the subtree a.
    # This is correct.
    
    return f"({to_newick(a)}:{d_a:.4f},{to_newick(b)}:{d_b:.4f},{to_newick(c)}:{d_c_branch:.4f});"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]})" + ";"
        
    # Convert to array
    # Map A,C,G,T to 0,1,2,3
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # Use vectorized conversion for speed
    seqs = [sequences[l] for l in labels]
    # Create a large array of characters
    # Convert to numpy array of chars
    # Then map to integers
    # For efficiency, we can do:
    # arr = np.array(seqs, dtype='S1')
    # Then use lookup table
    
    # Simple loop is fine for n<=32, L<=2000
    A = np.array([[char_map[c] for c in seq] for seq in seqs], dtype=np.uint8)
    
    # Calculate distances
    D = _jc_distance_vec(A)
    
    # Get initial tree from NJ
    newick = _nj_newick(D, labels)
    
    return newick

# EVOLVE-BLOCK-END
