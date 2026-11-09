import math
import numpy as np

def _to_mask(char):
    if char == 'A': return 1
    if char == 'C': return 2
    if char == 'G': return 4
    if char == 'T': return 8
    return 0

def _fitch_score(tree, masks):
    """
    Calculate parsimony score using Fitch's algorithm with bitmasks.
    tree: adjacency list (list of lists)
    masks: list of lists of ints (n_taxa x n_sites)
    """
    n = len(tree)
    if n <= 1:
        return 0
    
    # Root at 0
    # Get post-order traversal
    visited = [False] * n
    post_order = []
    
    # Iterative DFS to avoid recursion limits
    stack = [(0, -1, 0)] # node, parent, state (0=enter, 1=exit)
    while stack:
        node, parent, state = stack.pop()
        if state == 0:
            if visited[node]:
                continue
            visited[node] = True
            stack.append((node, parent, 1))
            for neighbor in tree[node]:
                if neighbor != parent:
                    stack.append((neighbor, node, 0))
        else:
            post_order.append(node)
            
    # Calculate scores
    # node_mask[node] = bitmask of possible states at this node
    # node_cost[node] = cost accumulated below this node
    node_mask = [0] * n
    node_cost = [0] * n
    
    # Initialize leaves
    for i in range(n):
        # If it's a leaf, mask is the state at this site
        # We process site by site? No, Fitch works per site.
        # We need to sum costs over all sites.
        # So we run Fitch for each site.
        # This is O(L * n).
        # L=2000, n=32. 64000 ops.
        pass

    total_cost = 0
    for site in range(len(masks[0])):
        # Reset masks for this site
        # We need to compute masks for all nodes for this specific site
        # Post-order is fixed for the topology
        
        # node_state[node] = bitmask for this site
        node_state = [0] * n
        site_cost = 0
        
        # Process in post-order
        for node in post_order:
            # Get children
            children = [nei for nei in tree[node] if nei != 0 or node == 0] 
            # Wait, parent tracking in post_order is hard without storing parent.
            # Let's store parent in the traversal.
            
        # Re-do traversal to store parent
        # Actually, we can just store the parent of each node during the first traversal.
        # Let's modify the traversal to store parent.
        
    # Let's refactor to store parent in a list during traversal
    # This is getting complex. Let's use a simpler recursive approach if n is small, 
    # but n=32 is safe for recursion? 
    # Python recursion limit is usually 1000. Depth 32 is fine.
    
    def calc_fitch(node, parent, site_idx):
        # Returns (mask, cost)
        if len(tree[node]) == 1 and node != 0: # Leaf (and not root if root has 1 neighbor)
             # Actually, if n=2, root 0 has neighbor 1. 1 has neighbor 0.
             # If node is leaf, its mask is the state.
             pass
             
    # Better: Pre-compute post_order and parents.
    parents = [-1] * n
    post_order = []
    visited = [False] * n
    stack = [0]
    visited[0] = True
    # This simple stack doesn't give post-order easily.
    
    # Let's use the explicit stack with state again, and store parent.
    parent_arr = [-1] * n
    post_order = []
    visited = [False] * n
    stack = [(0, -1, 0)]
    while stack:
        node, par, state = stack.pop()
        if state == 0:
            if visited[node]:
                continue
            visited[node] = True
            parent_arr[node] = par
            stack.append((node, par, 1))
            for nei in tree[node]:
                if nei != par:
                    stack.append((nei, node, 0))
        else:
            post_order.append(node)
            
    total_cost = 0
    for site in range(len(masks[0])):
        node_state = [0] * n
        site_cost = 0
        for node in post_order:
            # If leaf
            if node < n and (len(tree[node]) == 1 or (n == 2 and node != 0)):
                 # Wait, leaves are 0..n-1.
                 # If it's a leaf, state is masks[node][site]
                 # But what if the root is a leaf? (n=1)
                 # For n>=2, root 0 is not a leaf unless n=2 and we treat it specially?
                 # In a binary tree, leaves have degree 1.
                 # If node is a leaf:
                 if len(tree[node]) == 1:
                     node_state[node] = masks[node][site]
                 else:
                     # Internal node
                     # Intersect children states
                     inter = 15 # 1111
                     union = 0
                     has_child = False
                     for nei in tree[node]:
                         if nei == parent_arr[node]:
                             continue
                         has_child = True
                         inter &= node_state[nei]
                         union |= node_state[nei]
                     if has_child:
                         if inter == 0:
                             node_state[node] = union
                             site_cost += 1
                         else:
                             node_state[node] = inter
                     else:
                         # This is the root and it has no children? (n=1)
                         node_state[node] = masks[node][site]
        total_cost += site_cost
        
    return total_cost

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Convert to masks
    masks = []
    for label in labels:
        seq = sequences[label]
        mask_list = [_to_mask(c) for c in seq]
        masks.append(mask_list)
        
    # Stepwise Addition
    # Tree: adjacency list
    # Start with 0, 1
    tree = [[1], [0]]
    
    for k in range(2, n):
        best_score = float('inf')
        best_tree = None
        best_edge_idx = -1
        best_subdiv_node = -1
        
        # Try inserting k into each edge
        for u in range(len(tree)):
            for v in tree[u]:
                if v <= u: # Avoid duplicate edges
                    continue
                    
                # Create new tree
                new_tree = [list(nei_list) for nei_list in tree]
                # Subdivide edge (u, v)
                # Remove v from u, u from v
                new_tree[u].remove(v)
                new_tree[v].remove(u)
                
                # Add new node w
                w = len(new_tree)
                new_tree.append([])
                
                # Connect u-w, w-v, w-k
                new_tree[u].append(w)
                new_tree[w].append(u)
                
                new_tree[v].append(w)
                new_tree[w].append(v)
                
                # Add leaf k
                new_tree.append([])
                new_tree[k].append(w)
                new_tree[w].append(k)
                
                # Calculate score
                score = _fitch_score(new_tree, masks)
                
                if score < best_score:
                    best_score = score
                    best_tree = new_tree
                    best_edge_idx = u
                    best_subdiv_node = w
                    
        if best_tree is not None:
            tree = best_tree
            
    # NNI Optimization
    improved = True
    while improved:
        improved = False
        # Find all internal edges
        # An edge (u, v) is internal if both u and v are not leaves (degree > 1)
        # Or if n=3, the edges to the center are internal?
        # In a binary tree with n>3, internal nodes have degree 3.
        # Leaves have degree 1.
        
        for u in range(len(tree)):
            for v in tree[u]:
                if v <= u:
                    continue
                # Check if internal
                if len(tree[u]) <= 1 or len(tree[v]) <= 1:
                    continue
                    
                # Get neighbors
                # u's neighbors excluding v
                u_neis = [x for x in tree[u] if x != v]
                v_neis = [x for x in tree[v] if x != u]
                
                # In a binary tree, len(u_neis) should be 2, len(v_neis) should be 2
                if len(u_neis) != 2 or len(v_neis) != 2:
                    continue
                    
                a, b = u_neis
                c, d = v_neis
                
                # Original score
                orig_score = _fitch_score(tree, masks)
                
                # NNI 1: Swap b and c
                # u connects to a, c, v
                # v connects to b, d, u
                new_tree_1 = [list(nei_list) for nei_list in tree]
                new_tree_1[u].remove(b)
                new_tree_1[b].remove(u)
                new_tree_1[u].append(c)
                new_tree_1[c].append(u)
                
                new_tree_1[v].remove(c)
                new_tree_1[c].remove(v)
                new_tree_1[v].append(b)
                new_tree_1[b].append(v)
                
                score_1 = _fitch_score(new_tree_1, masks)
                
                # NNI 2: Swap b and d
                # u connects to a, d, v
                # v connects to b, c, u
                new_tree_2 = [list(nei_list) for nei_list in tree]
                new_tree_2[u].remove(b)
                new_tree_2[b].remove(u)
                new_tree_2[u].append(d)
                new_tree_2[d].append(u)
                
                new_tree_2[v].remove(d)
                new_tree_2[d].remove(v)
                new_tree_2[v].append(b)
                new_tree_2[b].append(v)
                
                score_2 = _fitch_score(new_tree_2, masks)
                
                if score_1 < orig_score:
                    tree = new_tree_1
                    improved = True
                    break
                elif score_2 < orig_score:
                    tree = new_tree_2
                    improved = True
                    break
            if improved:
                break
                
    # Convert to Newick
    # Root at 0
    # Post-order traversal
    visited = [False] * len(tree)
    
    def to_newick(node, parent):
        visited[node] = True
        children = [nei for nei in tree[node] if nei != parent]
        
        if not children:
            return labels[node]
            
        child_strs = []
        for c in children:
            if not visited[c]:
                child_strs.append(to_newick(c, node))
                
        if len(child_strs) == 1:
            return f"({child_strs[0]})"
        else:
            return f"({','.join(child_strs)})"
            
    result = to_newick(0, -1)
    return result
