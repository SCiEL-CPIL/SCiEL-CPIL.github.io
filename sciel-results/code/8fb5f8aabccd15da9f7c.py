"""Program for constructing small k-regular graphs of given girth."""
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g using a tree-based construction.
    This builds a tree of depth D = g//2 (if g even) or (g-1)//2 (if g odd),
    and connects leaves from different main branches to satisfy regularity.
    This ensures no cycle is shorter than g.
    """
    if k == 2:
        # Cycle graph
        n = max(g, 3)
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges
    
    if k == 1:
        # Single edge
        return 2, [(0, 1)]
    
    # Determine depth of the tree
    # For even g, we want min cycle >= g. Tree depth D = g/2 gives min cycle g+1.
    # For odd g, we want min cycle >= g. Tree depth D = (g-1)/2 gives min cycle g.
    if g % 2 == 0:
        D = g // 2
    else:
        D = (g - 1) // 2
        
    if D < 1:
        D = 1

    # Build the tree
    # Node 0 is the root.
    # Root has k children.
    # Other nodes have k-1 children.
    # We store edges and parent pointers to find branches.
    
    n_nodes = 0
    # We will build the tree level by level.
    # levels[i] is a list of node indices at depth i.
    levels = [[] for _ in range(D + 1)]
    
    # Assign IDs
    # Root is 0
    next_id = 1
    levels[0] = [0]
    
    for depth in range(D):
        current_level = levels[depth]
        next_level = []
        for node in current_level:
            # Determine number of children
            if depth == 0:
                num_children = k
            else:
                num_children = k - 1
            
            for _ in range(num_children):
                new_node = next_id
                next_id += 1
                levels[depth + 1].append(new_node)
                # Store edge (node, new_node) later, or now
                # We'll collect edges at the end or store in a list
        levels[depth + 1] = next_level

    n = next_id
    edges = []
    
    # Add tree edges
    for depth in range(D):
        current_level = levels[depth]
        next_level = levels[depth + 1]
        # We need to map parents to children correctly.
        # Since we generated children sequentially for each parent,
        # we can just iterate.
        
        idx_in_next = 0
        for node in current_level:
            if depth == 0:
                num_children = k
            else:
                num_children = k - 1
                
            for _ in range(num_children):
                child = next_level[idx_in_next]
                edges.append((node, child))
                idx_in_next += 1

    # Now connect the leaves
    # Leaves are in levels[D]
    leaves = levels[D]
    
    if D == 0:
        # Only root, k=1 handled, so this shouldn't happen for k>=2
        return n, edges

    # Group leaves by their branch (child of root)
    # The children of the root are levels[1]
    root_children = levels[1]
    
    # Map each leaf to its root-child ancestor
    # Since we built the tree sequentially, we can track this.
    # Alternatively, we can just use the fact that the leaves are generated in blocks.
    # Block i corresponds to root_children[i].
    
    num_leaves_per_branch = (k - 1) ** (D - 1) if D > 1 else 1
    
    branches = []
    start_idx = 0
    for i in range(k):
        end_idx = start_idx + num_leaves_per_branch
        branches.append(leaves[start_idx:end_idx])
        start_idx = end_idx
        
    # Connect leaves from different branches
    # We need to ensure each leaf gets (k-1) new edges.
    # We connect branch i to branch j for all i != j.
    # Since we have k branches, and we want to connect i to j,
    # and each leaf needs k-1 connections, we can simply connect
    # branch i to branch j with a perfect matching for all i < j?
    # No, if we connect i to j, each leaf in i gets 1 edge to j.
    # If we do this for all j != i, each leaf in i gets k-1 edges.
    
    # So we need to add edges between branch i and branch j for all i != j.
    # To avoid duplicate edges (i,j) and (j,i), we iterate i < j.
    # But we need to ensure the matching covers all leaves.
    # If we zip branches[i] and branches[j], we create num_leaves_per_branch edges.
    # This gives degree 1 to each leaf in branch i towards branch j.
    # Summing over all j != i, each leaf gets k-1 edges.
    
    for i in range(k):
        for j in range(i + 1, k):
            # Zip the two lists
            # They are of the same length
            for u, v in zip(branches[i], branches[j]):
                edges.append((u, v))

    return n, edges
# EVOLVE-BLOCK-END
