import time
import math
import numpy as np
from scipy.optimize import minimize

def _jc_distance(a, b):
    """Calculate Jukes-Cantor distance between two sequences."""
    diffs = np.sum(a != b)
    p = diffs / len(a)
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _build_distance_matrix(labels, sequences):
    """Build the distance matrix."""
    n = len(labels)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            matrix[i, j] = d
            matrix[j, i] = d
    return matrix

def _nj_tree(matrix, labels):
    """
    Neighbour Joining algorithm.
    Returns a tree structure as a list of tuples (left, right, length) and a list of leaf labels.
    Actually, we need a structure that can be used for ML.
    Let's return a list of nodes.
    Each node is a dict: {'id': int, 'left': int, 'right': int, 'length': float}
    Leaves are represented by their label.
    """
    n = len(labels)
    if n == 2:
        # Just a single branch
        return [(labels[0], labels[1], matrix[0, 1])]
    
    # nodes: list of active nodes (labels or internal node ids)
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    # matrix is n x n, but we will update it
    # We need to map active nodes to their indices in the current matrix
    # This is tricky. Let's use the standard NJ algorithm with a list of active nodes.
    
    # Standard NJ implementation
    # We will keep a dictionary of current trees
    # And a dictionary of distances between active nodes
    
    # Initialize
    # active: list of node ids
    # dist: dict of dict (a, b) -> distance
    dist = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[(i, j)] = matrix[i, j]
                dist[(j, i)] = matrix[i, j]
    
    # total: dict of node -> sum of distances to all other active nodes
    total = {i: sum(dist[(i, j)] for j in active if j != i) for i in active}
    
    trees = {i: labels[i] for i in range(n)}
    next_id = n
    
    while len(active) > 2:
        n_active = len(active)
        best_criterion = float('inf')
        best_pair = None
        
        for i in range(n_active):
            a = active[i]
            for j in range(i+1, n_active):
                b = active[j]
                criterion = (n_active - 2) * dist[(a, b)] - total[a] - total[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_pair = (a, b)
        
        a, b = best_pair
        # Join a and b
        # Calculate branch lengths
        delta = (total[a] - total[b]) / (n_active - 2)
        len_a = 0.5 * dist[(a, b)] + 0.5 * delta
        len_b = 0.5 * dist[(a, b)] - 0.5 * delta
        
        # Ensure non-negative
        if len_a < 0: len_a = 0
        if len_b < 0: len_b = 0
        
        new_id = next_id
        next_id += 1
        
        # New tree structure
        # We store the tree as a list of tuples: (left, right, len_left, len_right)
        # But we need to handle the fact that 'left' and 'right' can be leaves or internal nodes.
        # Let's represent the tree as a list of nodes.
        # We will build a list of internal nodes.
        
        # For now, let's just record the merge.
        # We need to update the distance matrix for the new node.
        for x in active:
            if x not in (a, b):
                dist[(new_id, x)] = 0.5 * (dist[(a, x)] + dist[(b, x)] - dist[(a, b)])
                dist[(x, new_id)] = dist[(new_id, x)]
        
        total[new_id] = sum(dist[(new_id, x)] for x in active if x not in (a, b))
        
        # Update active list
        active = [x for x in active if x not in (a, b)] + [new_id]
        
        # Record the tree structure
        # We need to store the topology.
        # Let's use a list of tuples: (left_id, right_id, len_left, len_right)
        # But we also need to know which ids are leaves and which are internal.
        # We will maintain a dict: node_id -> (left, right, len_left, len_right)
        # Leaves are just labels.
        
        # This is getting complicated to manage in a single function.
        # Let's return the matrix and the merge history, and reconstruct the tree later?
        # Or just return the Newick string?
        # But we need the tree for ML.
        
        # Let's store the tree in a list of nodes.
        # nodes[new_id] = (a, b, len_a, len_b)
        pass
        
    # This approach is messy. Let's use a standard library or a cleaner implementation.
    # Given the constraints, I will implement a simpler version that returns the Newick string
    # and then I will parse it back into a structure for ML? No, that's inefficient.
    
    # Let's try to return a list of internal nodes.
    # Each internal node is a tuple (left, right, len_left, len_right).
    # Leaves are strings.
    
    # Re-implementing NJ to return a tree structure.
    
    pass

# I will write the complete code in the final block.
