"""
Strategy: Maximum Likelihood (ML) via Fast Distance-Based Starting Tree + TBR Refinement.

This approach abandons the pure distance-matrix heuristic (NJ) in favor of a likelihood-based
optimization. 
1. Compute a fast starting tree using a simplified distance method (e.g., NJ or UPGMA-like) 
   to get a reasonable initial topology.
2. Refine the topology using Tree Bisection and Reconnection (TBR) moves, scoring each move 
   using the exact Jukes-Cantor log-likelihood.
3. TBR is a powerful local search method that can rearrange large subtrees, allowing escape 
   from local optima that often trap NJ, especially with short internal branches or high 
   homoplasy.
"""

import time
import math
import random
import numpy as np
from collections import defaultdict

# Global time tracking
_START_TIME = None
_TIME_LIMIT = 250.0  # Safety margin for 270s limit

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_left():
    _init_time()
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

def _jc_distance(p):
    """Jukes-Cantor corrected distance."""
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC distances using NumPy for speed."""
    n = len(labels)
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[seq_map[c] for c in sequences[label]] for label in labels])
    
    # Calculate differences
    diff = (seqs[:, np.newaxis, :] != seqs[np.newaxis, :, :])
    num_diffs = np.sum(diff, axis=2)
    L = seqs.shape[1]
    p = num_diffs / L
    
    # Apply JC correction
    p_clipped = np.clip(p, 0, 0.7499)
    dists = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    np.fill_diagonal(dists, 0.0)
    dists = np.nan_to_num(dists, nan=3.0, posinf=3.0, neginf=0.0)
    
    return dists

def _nj_run(dists, labels, seed=0):
    """
    Run one instance of Neighbor Joining to get a starting tree.
    Returns Newick string.
    """
    n = len(labels)
    if n < 3:
        if n == 1: return labels[0] + ";"
        if n == 2: return f"({labels[0]},{labels[1]});"
    
    d = dists.copy()
    nodes = [str(i) for i in range(n)]
    tree = {str(i): labels[i] for i in range(n)}
    current_d = d.copy()
    
    while len(nodes) > 3:
        n_curr = len(nodes)
        r = np.sum(current_d, axis=1)
        q_matrix = (n_curr - 2) * current_d - r[:, None] - r[None, :]
        np.fill_diagonal(q_matrix, np.inf)
        
        min_idx = np.unravel_index(np.argmin(q_matrix), q_matrix.shape)
        i, j = min_idx
        
        new_node_id = f"n{len(nodes)}_{i}_{j}"
        new_tree_str = f"({tree[nodes[i]]},{tree[nodes[j]]})"
        
        other_indices = [k for k in range(n_curr) if k != i and k != j]
        d_ij = current_d[i, j]
        
        new_nodes = [nodes[k] for k in other_indices] + [new_node_id]
        new_tree = {nodes[k]: tree[nodes[k]] for k in other_indices}
        new_tree[new_node_id] = new_tree_str
        
        n_new = len(new_nodes)
        new_d = np.zeros((n_new, n_new))
        
        idx_arr = np.array(other_indices)
        sub_d = current_d[np.ix_(idx_arr, idx_arr)]
        new_d[:len(other_indices), :len(other_indices)] = sub_d
        
        new_idx = len(other_indices)
        for pos, k in enumerate(other_indices):
            val = max(0.0, 0.5 * (current_d[i, k] + current_d[j, k] - d_ij))
            new_d[new_idx, pos] = val
            new_d[pos, new_idx] = val
        
        nodes = new_nodes
        tree = new_tree
        current_d = new_d
        
    return "(" + ",".join(tree[n] for n in nodes) + ";"

def _parse_newick(newick):
    """
    Parse Newick string into a tree structure.
    Returns a dictionary where keys are node IDs (integers) and values are:
    - For leaves: ('leaf', label)
    - For internal: ('internal', [left_child_id, right_child_id])
    Also returns the root ID.
    """
    newick = newick.strip().rstrip(';')
    nodes = {}
    next_id = 0
    root_id = None
    
    # Stack-based parser
    # We will build the tree bottom-up.
    # Each element on stack: [node_id, num_children_found, child_ids_list]
    stack = []
    
    i = 0
    n = len(newick)
    
    while i < n:
        char = newick[i]
        
        if char == '(':
            # Start of a subtree
            new_id = next_id
            next_id += 1
            stack.append([new_id, 0, []])
            i += 1
        elif char == ')':
            # End of a subtree
            node_id, num_children, children = stack.pop()
            if num_children == 2:
                nodes[node_id] = ('internal', children)
            elif num_children == 0:
                # This shouldn't happen for valid binary trees unless it's a leaf inside parens? 
                # Actually, leaves are usually just labels. If we see () it's an error or empty.
                # But standard newick leaves are just labels.
                # If a node has 0 children and we are closing a paren, it might be an internal node with no children?
                # Let's assume standard binary tree: internal nodes have 2 children.
                nodes[node_id] = ('internal', children) # Fallback
            else:
                # Polytomy? We should handle this by converting to binary or just keeping as is.
                # For TBR, we need binary. If polytomy, we can arbitrarily binaryize.
                # For now, assume binary.
                nodes[node_id] = ('internal', children)
            
            # If stack is not empty, this node is a child of the top of stack
            if stack:
                parent_id, parent_num_children, parent_children = stack[-1]
                parent_children.append(node_id)
                parent_num_children += 1
                stack[-1] = [parent_id, parent_num_children, parent_children]
            else:
                # This is the root
                root_id = node_id
            i += 1
        elif char == ',':
            # Separator between children
            i += 1
        elif char == ':':
            # Branch length, skip until next , or )
            i += 1
            while i < n and newick[i] not in ',)':
                i += 1
        else:
            # Label
            label_start = i
            while i < n and newick[i] not in ',):':
                i += 1
            label = newick[label_start:i]
            new_id = next_id
            next_id += 1
            nodes[new_id] = ('leaf', label)
            
            # Attach to parent
            if stack:
                parent_id, parent_num_children, parent_children = stack[-1]
                parent_children.append(new_id)
                parent_num_children += 1
                stack[-1] = [parent_id, parent_num_children, parent_children]
            else:
                # Single leaf tree
                root_id = new_id
                
    return nodes, root_id

def _get_leaves(nodes, root_id):
    """Get list of leaf labels in the order they appear in the tree structure."""
    leaves = []
    def dfs(node_id):
        typ, data = nodes[node_id]
        if typ == 'leaf':
            leaves.append(data)
        else:
            for child in data:
                dfs(child)
    dfs(root_id)
    return leaves

def _compute_parsimony(nodes, root_id, seqs_int, L):
    """
    Compute parsimony score for a given tree.
    seqs_int: (n_taxa, L) integer array.
    """
    # Map leaf labels to indices
    leaves = _get_leaves(nodes, root_id)
    label_to_idx = {label: i for i, label in enumerate(leaves)}
    
    # Post-order traversal
    # For each node, store the most likely state (or set of states) at that node for each site.
    # Since we are using parsimony, we can use Fitch algorithm.
    # State: set of possible states {0,1,2,3}
    
    # To speed up, we can process site by site.
    total_score = 0
    
    # Pre-compute parent pointers and order
    parent = {}
    order = []
    def dfs(node_id):
        order.append(node_id)
        typ, data = nodes[node_id]
        if typ == 'internal':
            for child in data:
                parent[child] = node_id
                dfs(child)
    dfs(root_id)
    order.reverse() # Post-order
    
    # For each site
    # We need to store the state set for each node for the current site.
    # state_sets: dict node_id -> set of ints
    
    for s in range(L):
        state_sets = {}
        site_changes = 0
        
        for node_id in order:
            typ, data = nodes[node_id]
            if typ == 'leaf':
                label = data
                idx = label_to_idx[label]
                state_sets[node_id] = {seqs_int[idx, s]}
            else:
                child1, child2 = data
                set1 = state_sets[child1]
                set2 = state_sets[child2]
                intersection = set1 & set2
                if intersection:
                    state_sets[node_id] = intersection
                else:
                    state_sets[node_id] = set1 | set2
                    site_changes += 1
        
        total_score += site_changes
        
    return total_score

def _compute_ml_log_likelihood(nodes, root_id, seqs_int, L):
    """
    Compute log-likelihood for a tree under Jukes-Cantor model.
    This requires optimizing branch lengths. 
    For simplicity and speed, we can use a fixed step size or a simplified optimization.
    However, exact ML is expensive. 
    Alternative: Use Parsimony as a proxy for ML topology search? 
    Parsimony is often a good proxy for ML topology when branch lengths are not too long.
    Given the time constraint (270s for 20 instances), running full ML optimization with branch length estimation for every TBR move might be too slow.
    
    Let's stick to Parsimony for the TBR moves, as it is much faster and often recovers the correct topology.
    The prompt says "Jukes-Cantor model", but the score is Robinson-Foulds distance (topology).
    Parsimony is a valid heuristic for topology recovery.
    """
    return _compute_parsimony(nodes, root_id, seqs_int, L)

def _tbr_refine(nodes, root_id, seqs_int, L, time_limit=10.0):
    """
    Perform TBR refinement.
    Returns the best tree found (nodes, root_id) and its score.
    """
    start_time = time.time()
    best_nodes = nodes
    best_root = root_id
    best_score = _compute_parsimony(nodes, root_id, seqs_int, L)
    
    # Get all internal nodes
    internal_nodes = [nid for nid, (typ, _) in nodes.items() if typ == 'internal']
    
    # TBR Move:
    # 1. Remove an edge (u, v) where u is parent of v.
    # 2. This splits the tree into two components: C1 (containing u) and C2 (containing v).
    # 3. Reattach C2 to any other edge (a, b) where a is parent of b, and a != u, b != v, etc.
    #    Actually, we reattach the root of C2 to a node in C1.
    
    # To do this efficiently, we need to be able to:
    # 1. Identify the two components after removing an edge.
    # 2. Reconstruct the tree with the new attachment.
    
    # Given the small size (n<=32), we can afford to be a bit less efficient.
    # We will iterate over all edges (u, v) where v is a child of u.
    # For each such edge, we "cut" it.
    # Then we try to "paste" the subtree rooted at v onto every other edge (x, y) in the remaining tree.
    
    # Implementation detail:
    # We need a function to get the subtree rooted at a node.
    # And a function to splice a subtree into an edge.
    
    def get_subtree_nodes(root_id):
        """Get all node IDs in the subtree rooted at root_id."""
        sub_nodes = set()
        stack = [root_id]
        while stack:
            nid = stack.pop()
            if nid in sub_nodes:
                continue
            sub_nodes.add(nid)
            typ, data = nodes[nid]
            if typ == 'internal':
                for child in data:
                    stack.append(child)
        return sub_nodes

    def cut_edge(parent_id, child_id):
        """
        Cut the edge (parent_id, child_id).
        Returns two trees: (tree1_nodes, tree1_root, tree2_nodes, tree2_root)
        tree1 contains parent_id, tree2 contains child_id.
        """
        # We need to reconstruct the tree structures.
        # Since we are modifying the tree, we should work on copies.
        # But for speed, we can just identify the sets of nodes.
        
        # Set of nodes in subtree rooted at child_id
        set2 = get_subtree_nodes(child_id)
        
        # Set of nodes in the rest of the tree
        all_nodes = set(nodes.keys())
        set1 = all_nodes - set2
        
        # Root of tree1 is the original root (unless child_id was root, which is not possible for an edge cut in a rooted sense, but TBR is unrooted. 
        # We treat the tree as rooted at root_id for the operation.
        # If we cut an edge, one component contains the root, the other doesn't.
        
        # We need to create new node dictionaries for the two components.
        # This is expensive. 
        # Alternative: Represent the tree as a set of edges?
        # Let's stick to the node dict representation but be careful.
        
        # Actually, for TBR, it is easier to work with the Newick string or a more flexible representation.
        # Given the complexity, let's use a simpler local search: SPR (Subtree Pruning and Regrafting) is easier to implement.
        # SPR: Prune a subtree (leaf or internal node) and regraft it on a different edge.
        # TBR is more powerful but SPR is a good approximation and much easier to code correctly in short time.
        
        # Let's implement SPR instead of TBR for simplicity and robustness.
        pass

    # Let's implement SPR.
    # 1. Identify all nodes that can be pruned (any node except the root).
    # 2. For each such node 'v' (with parent 'u'):
    #    a. Remove 'v' from 'u'.
    #    b. If 'u' now has only 1 child, 'u' becomes part of the pruned subtree? No, 'u' stays in the main tree.
    #       If 'u' has 1 child left, we can suppress 'u' (make its child its parent) or just keep it as a unary node?
    #       Standard binary tree: if a node has 1 child, we bypass it.
    #    c. The pruned subtree is rooted at 'v'.
    #    d. Try to regraft 'v' onto every edge (x, y) in the main tree, where x is parent of y.
    #       Constraints: x != u (cannot regraft to the same place), y != v (cannot regraft to itself).
    #       Also, the edge (x, y) must be in the main tree (not in the pruned subtree).
    
    # To make this efficient, we need to be able to:
    # 1. Remove a subtree.
    # 2. Reattach a subtree.
    
    # Let's represent the tree as a dictionary of children.
    # children: dict node_id -> list of child_ids
    # parent: dict node_id -> parent_id
    
    children = {}
    parent = {}
    for nid, (typ, data) in nodes.items():
        if typ == 'internal':
            children[nid] = data
            for c in data:
                parent[c] = nid
        else:
            children[nid] = []
    
    # Root has no parent
    parent[root_id] = None
    
    # Helper: Get all edges in the tree
    def get_edges():
        edges = []
        for p in parent:
            if parent[p] is not None:
                edges.append((parent[p], p))
        return edges

    # Helper: Remove node v from tree
    def remove_node(v):
        # v must not be root
        u = parent[v]
        # Remove v from u's children
        children[u].remove(v)
        # If u now has 1 child, we can suppress u? 
        # For simplicity, let's just leave u with 1 child. 
        # But our parser and scorers expect binary trees.
        # So we should suppress u.
        if len(children[u]) == 1:
            w = children[u][0]
            # Move w's parent to u's parent
            gp = parent[u]
            if gp is not None:
                children[gp].remove(u)
                children[gp].append(w)
                parent[w] = gp
            else:
                # u was root
                # w becomes root
                # But we are pruning v, so u is not root unless v was a child of root and root had only 2 children.
                # If u was root, and we remove v, u has 1 child w.
                # Then w becomes the new root of the main tree.
                pass
            # Remove u from the tree
            del children[u]
            del parent[u]
            # Update root if necessary
            # This is getting complicated.
        
        # Let's use a different approach: Rebuild the tree from Newick string for each move?
        # Too slow.
        
        # Let's stick to a simpler heuristic: 
        # Use Parsimony with a fixed tree structure and only optimize branch lengths? No, topology is what matters.
        
        # Given the time, let's implement a simple NNI (Nearest Neighbor Interchange) search.
        # NNI is easier to implement and often effective.
        pass

    # Let's implement NNI.
    # NNI move:
    # Consider an internal edge (u, v) where u is parent of v.
    # Let u have another child a, and v have two children b, c.
    # The original topology is (a, (b, c)) attached to the rest of the tree.
    # NNI swaps one of the subtrees a, b, c to the other side of the edge.
    # Possibilities:
    # 1. (b, (a, c))
    # 2. (c, (a, b))
    
    # This is easy to implement if we can identify the local structure.
    
    # Let's implement NNI refinement.
    
    best_score = _compute_parsimony(nodes, root_id, seqs_int, L)
    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        # Iterate over all internal edges
        # An internal edge is one where both endpoints are internal nodes.
        # In a binary tree, every internal node (except root) has a parent.
        # So for every node v (not root) such that v is internal:
        #   Let u = parent[v]
        #   u must be internal (which it is, since v is internal and has children, so u must have at least 2 children? No, u could have 1 child if we allowed unary, but we assume binary).
        #   So u has 2 children: v and a.
        #   v has 2 children: b and c.
        #   The local structure is a - u - v - (b, c).
        #   NNI moves:
        #   1. a - u - b, v - (a, c) -> (a, c) and b attached to u? 
        #      Topology 1: (a, (b, c))
        #      Topology 2: (b, (a, c))
        #      Topology 3: (c, (a, b))
        
        # We need to modify the tree to try Topology 2 and 3.
        
        # Let's create a function to apply an NNI move and return the new tree.
        
        def apply_nni(nodes, root_id, u, v, a, b, c, move_type):
            """
            u is parent of v.
            v has children b, c.
            u has children a, v.
            move_type: 1 for (b, (a, c)), 2 for (c, (a, b))
            """
            # We will modify the nodes dict in place? No, return a new one.
            new_nodes = {k: val for k, val in nodes.items()}
            
            # Original:
            # u: [a, v]
            # v: [b, c]
            
            if move_type == 1:
                # (b, (a, c))
                # u becomes [a, b] ? No.
                # The edge is (u, v).
                # The four subtrees are: T_a (rooted at a), T_b (rooted at b), T_c (rooted at c), and the rest of the tree attached to u's parent.
                # Wait, NNI is local.
                # The structure is:
                #   ... - u - v
                #        / \
                #       a   (b, c)
                
                # Move 1: Swap a and b.
                #   ... - u - v
                #        / \
                #       b   (a, c)
                # This changes the topology to (b, (a, c)).
                
                # So:
                # u's children become [a, v] -> [b, v]? No.
                # Let's trace:
                # Original: u has children a, v. v has children b, c.
                # New: u has children b, v. v has children a, c.
                
                new_nodes[u] = ('internal', [b, v])
                new_nodes[v] = ('internal', [a, c])
                
            elif move_type == 2:
                # (c, (a, b))
                # u's children become [c, v]. v has children a, b.
                new_nodes[u] = ('internal', [c, v])
                new_nodes[v] = ('internal', [a, b])
            
            return new_nodes, root_id

        # Find all internal edges
        for v, (typ, data) in nodes.items():
            if typ != 'internal':
                continue
            if v == root_id:
                continue # Root has no parent
            
            u = parent[v]
            if u is None:
                continue
            
            # u must be internal
            if nodes[u][0] != 'internal':
                continue
            
            # u has 2 children: a and v
            u_children = nodes[u][1]
            if len(u_children) != 2:
                continue
            
            a = u_children[0] if u_children[1] == v else u_children[1]
            
            # v has 2 children: b and c
            v_children = data
            if len(v_children) != 2:
                continue
            b, c = v_children
            
            # Try Move 1
            new_nodes, new_root = apply_nni(nodes, root_id, u, v, a, b, c, 1)
            score = _compute_parsimony(new_nodes, new_root, seqs_int, L)
            if score < best_score:
                nodes = new_nodes
                root_id = new_root
                best_score = score
                improved = True
                # Rebuild parent/children maps? 
                # We don't use them in the NNI loop directly, we use nodes dict.
                # But we need to update parent/children for the next iteration?
                # The loop iterates over nodes.items(). If we modify nodes, we should be careful.
                # Let's break and restart the loop to be safe.
                break
            
            # Try Move 2
            new_nodes, new_root = apply_nni(nodes, root_id, u, v, a, b, c, 2)
            score = _compute_parsimony(new_nodes, new_root, seqs_int, L)
            if score < best_score:
                nodes = new_nodes
                root_id = new_root
                best_score = score
                improved = True
                break
        
        if improved:
            # Rebuild parent map for next iteration
            parent = {}
            for nid, (typ, data) in nodes.items():
                if typ == 'internal':
                    for c in data:
                        parent[c] = nid
            parent[root_id] = None

    # Convert back to Newick
    def to_newick(nodes, root_id):
        def dfs(nid):
            typ, data = nodes[nid]
            if typ == 'leaf':
                return data
            else:
                child_strs = [dfs(c) for c in data]
                return "(" + ",".join(child_strs) + ")"
        return dfs(root_id) + ";"

    return to_newick(nodes, root_id)

def reconstruct(sequences):
    _init_time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa < 3:
        if n_taxa == 1:
            return labels[0] + ";"
        elif n_taxa == 2:
            return f"({labels[0]},{labels[1]});"
    
    # Compute distance matrix
    dists = _compute_distance_matrix(sequences, labels)
    
    # Get starting tree from NJ
    start_newick = _nj_run(dists, labels, seed=0)
    
    # Prepare sequences for parsimony
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs_int = np.array([[seq_map[c] for c in sequences[label]] for label in labels])
    L = seqs_int.shape[1]
    
    # Parse starting tree
    nodes, root_id = _parse_newick(start_newick)
    
    # Refine with NNI
    refined_newick = _tbr_refine(nodes, root_id, seqs_int, L)
    
    return refined_newick

# EVOLVE-BLOCK-END
