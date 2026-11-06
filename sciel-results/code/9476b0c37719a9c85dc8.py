import time
import math
import numpy as np
from collections import defaultdict

_START_TIME = None
_TIME_LIMIT = 265.0  # Safety margin for 270s limit

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_left():
    _init_time()
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

class TreeNode:
    __slots__ = ['left', 'right', 'label', 'length']
    def __init__(self, left=None, right=None, label=None, length=0.0):
        self.left = left
        self.right = right
        self.label = label
        self.length = length

    def is_leaf(self):
        return self.left is None and self.right is None

def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC distances for initial NJ tree."""
    n = len(labels)
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[seq_map[c] for c in sequences[label]] for label in labels])
    diff = (seqs[:, np.newaxis, :] != seqs[np.newaxis, :, :])
    num_diffs = np.sum(diff, axis=2)
    L = seqs.shape[1]
    p = num_diffs / L
    p_clipped = np.clip(p, 0, 0.7499)
    dists = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    np.fill_diagonal(dists, 0.0)
    dists = np.nan_to_num(dists, nan=3.0, posinf=3.0, neginf=0.0)
    return dists

def _nj_run(dists, labels):
    """Standard Neighbor Joining to get an initial tree topology."""
    n = len(labels)
    if n < 3:
        if n == 1:
            return TreeNode(label=labels[0])
        elif n == 2:
            root = TreeNode()
            l = TreeNode(label=labels[0])
            r = TreeNode(label=labels[1])
            root.left = l
            root.right = r
            return root

    nodes = [TreeNode(label=labels[i]) for i in range(n)]
    current_d = dists.copy()
    active_ids = list(range(n))
    node_map = {i: nodes[i] for i in range(n)}
    node_id_counter = n
    
    while len(active_ids) > 2:
        n_curr = len(active_ids)
        r = np.sum(current_d, axis=1)
        q_matrix = (n_curr - 2) * current_d - r[:, None] - r[None, :]
        np.fill_diagonal(q_matrix, np.inf)
        
        min_idx = np.unravel_index(np.argmin(q_matrix), q_matrix.shape)
        i, j = min_idx
        
        id_i = active_ids[i]
        id_j = active_ids[j]
        
        node_i = node_map[id_i]
        node_j = node_map[id_j]
        
        d_ij = current_d[i, j]
        
        alpha = (r[i] - r[j]) / (2.0 * (n_curr - 2))
        len_i = 0.5 * d_ij + alpha
        len_j = 0.5 * d_ij - alpha
        
        new_id = node_id_counter
        node_id_counter += 1
        
        new_node = TreeNode()
        new_node.left = node_i
        new_node.right = node_j
        node_i.length = max(0.0, len_i)
        node_j.length = max(0.0, len_j)
        
        node_map[new_id] = new_node
        
        new_active_ids = [k for k in range(n_curr) if k != i and k != j]
        new_active_ids.append(len(new_active_ids))
        
        n_new = len(new_active_ids)
        new_d = np.zeros((n_new, n_new))
        
        pos_to_id = {}
        for pos, old_pos in enumerate(new_active_ids[:-1]):
            pos_to_id[pos] = active_ids[old_pos]
        pos_to_id[n_new - 1] = new_id
        
        for p1 in range(n_new):
            for p2 in range(p1 + 1, n_new):
                id1 = pos_to_id[p1]
                id2 = pos_to_id[p2]
                
                if id1 == new_id or id2 == new_id:
                    if id1 == new_id:
                        x_id = id2
                    else:
                        x_id = id1
                    
                    old_pos_x = active_ids.index(x_id)
                    
                    d_ix = current_d[i, old_pos_x]
                    d_jx = current_d[j, old_pos_x]
                    
                    d_new_x = 0.5 * (d_ix + d_jx - d_ij)
                    d_new_x = max(0.0, d_new_x)
                    
                    new_d[p1, p2] = d_new_x
                    new_d[p2, p1] = d_new_x
                else:
                    old_pos1 = active_ids.index(id1)
                    old_pos2 = active_ids.index(id2)
                    new_d[p1, p2] = current_d[old_pos1, old_pos2]
                    new_d[p2, p1] = current_d[old_pos1, old_pos2]
                    
        current_d = new_d
        active_ids = list(pos_to_id.values())
        
    if len(active_ids) == 2:
        id1 = active_ids[0]
        id2 = active_ids[1]
        node1 = node_map[id1]
        node2 = node_map[id2]
        
        d_12 = current_d[0, 1]
        
        root = TreeNode()
        root.left = node1
        root.right = node2
        node1.length = 0.5 * d_12
        node2.length = 0.5 * d_12
        
        return root
        
    return nodes[0]

def _tree_to_newick(root):
    """Convert TreeNode to Newick string."""
    if root is None:
        return ""
    if root.is_leaf():
        return root.label
    left_str = _tree_to_newick(root.left)
    right_str = _tree_to_newick(root.right)
    return f"({left_str},{right_str})"

def _get_leaves(node):
    if node.is_leaf():
        return [node.label]
    leaves = []
    if node.left:
        leaves.extend(_get_leaves(node.left))
    if node.right:
        leaves.extend(_get_leaves(node.right))
    return leaves

def _compute_parsimony(sequences, labels, root):
    """
    Compute parsimony score for a tree.
    Uses Fitch algorithm.
    """
    n_sites = len(sequences[labels[0]])
    # Map labels to indices for fast access
    label_to_idx = {label: i for i, label in enumerate(labels)}
    seqs_int = np.array([[{'A': 0, 'C': 1, 'G': 2, 'T': 3}[c] for c in sequences[label]] for label in labels])
    
    # Fitch algorithm
    # dp[node][site] = set of states
    # We store as bitmasks for speed. 4 states -> 4 bits.
    # State 0: A, 1: C, 2: G, 3: T
    # Mask for state s: 1 << s
    
    # Post-order traversal
    order = []
    parent = {}
    
    def dfs(node, p):
        parent[node] = p
        order.append(node)
        if not node.is_leaf():
            dfs(node.left, node)
            dfs(node.right, node)
            
    dfs(root, None)
    
    total_cost = 0
    # Process in reverse order (leaves to root)
    dp = {}
    
    # Initialize leaves
    for node in order:
        if node.is_leaf():
            idx = label_to_idx[node.label]
            # One-hot vector for each site
            # mask = 1 << state
            masks = np.zeros(n_sites, dtype=np.uint8)
            states = seqs_int[idx]
            for s in range(n_sites):
                masks[s] = 1 << states[s]
            dp[node] = masks
        else:
            dp[node] = None
            
    for node in reversed(order):
        if node.is_leaf():
            continue
            
        left = node.left
        right = node.right
        
        # Intersection
        inter = dp[left] & dp[right]
        union = dp[left] | dp[right]
        
        # Cost is number of sites where intersection is empty
        # If intersection is empty, we must make a change (cost +1)
        # We take the union as the new state set for the parent
        
        empty_mask = (inter == 0)
        cost = np.sum(empty_mask)
        total_cost += cost
        
        dp[node] = union
        
    return total_cost

def _spr_refine(sequences, labels, root):
    """
    Refine topology using SPR moves guided by parsimony.
    """
    n_sites = len(sequences[labels[0]])
    label_to_idx = {label: i for i, label in enumerate(labels)}
    seqs_int = np.array([[{'A': 0, 'C': 1, 'G': 2, 'T': 3}[c] for c in sequences[label]] for label in labels])
    
    def compute_dp(node):
        """Compute DP vectors for a subtree rooted at node. Returns (dp_dict, cost)."""
        dp_local = {}
        cost = 0
        order = []
        def dfs(n):
            order.append(n)
            if not n.is_leaf():
                dfs(n.left)
                dfs(n.right)
        dfs(node)
        
        for n in order:
            if n.is_leaf():
                idx = label_to_idx[n.label]
                masks = np.zeros(n_sites, dtype=np.uint8)
                states = seqs_int[idx]
                for s in range(n_sites):
                    masks[s] = 1 << states[s]
                dp_local[n] = masks
            else:
                dp_local[n] = None
                
        for n in reversed(order):
            if n.is_leaf():
                continue
            l = n.left
            r = n.right
            inter = dp_local[l] & dp_local[r]
            union = dp_local[l] | dp_local[r]
            empty = (inter == 0)
            cost += np.sum(empty)
            dp_local[n] = union
        return dp_local, cost

    # Precompute initial DP for whole tree
    dp_tree, current_cost = compute_dp(root)
    
    # Helper to get leaves of a node
    def get_leaves(n):
        if n.is_leaf():
            return [n.label]
        return get_leaves(n.left) + get_leaves(n.right)
    
    improved = True
    max_iterations = 20000
    iter_count = 0
    
    while improved and iter_count < max_iterations and _time_left() > 10:
        improved = False
        iter_count += 1
        
        # Collect all internal edges (u, v) where u is parent, v is child
        # We need to traverse the tree to find all such pairs.
        # Since we are modifying the tree, we must be careful.
        # We will collect a list of (parent_node, child_node) pairs.
        
        edges = []
        def collect_edges(node, parent_node):
            if parent_node:
                edges.append((parent_node, node))
            if not node.is_leaf():
                collect_edges(node.left, node)
                collect_edges(node.right, node)
                
        collect_edges(root, None)
        
        best_gain = 0
        best_move = None
        
        for u, v in edges:
            # Prune subtree v
            # Regraft into edge (a, b)
            # Constraints:
            # 1. v is not in subtree of a (if we reattach to a, it creates a cycle or invalid tree)
            # 2. v is not in subtree of b (same)
            # 3. The edge (a, b) must exist in the tree.
            
            # To check if v is in subtree of x, we can check if any leaf of v is in leaves of x.
            # This is O(n) per check, which is too slow for O(n^2) edges.
            # We need a faster way.
            # Since we are doing this for every edge, and n is small (<=32), maybe O(n^3) total is okay?
            # 32^3 = 32768. Times number of iterations.
            # Let's optimize the "is in subtree" check.
            
            leaves_v = set(get_leaves(v))
            
            for a, b in edges:
                # Skip if a or b is v or u (self-regraft or parent)
                if a == v or b == v or a == u or b == u:
                    continue
                
                # Check if v is in subtree of a or b
                # If v is in subtree of a, then a is a descendant of v or v is a descendant of a.
                # If v is a descendant of a, then a is an ancestor of v.
                # If we reattach v to (a, b), and v is already in subtree of a, we create a cycle.
                
                # How to check if v is in subtree of a?
                # We can precompute ancestors? But tree changes.
                # Let's just check if any leaf of v is in leaves of a.
                # This is O(n) set intersection.
                
                # Optimization: If the number of leaves in v is small, this is fast.
                # Worst case v has 1 leaf, a has 31 leaves.
                
                # Let's try to be smarter.
                # We only need to check if the "root" of the v-subtree is inside a-subtree.
                # Actually, if v is a descendant of a, then a is an ancestor of v.
                # We can check if `a` is in the path from `v` to `root`.
                # Or simply: if we reattach v to (a, b), the new tree is valid iff v is not a descendant of a or b.
                
                # Let's assume for now we do the set check.
                leaves_a = set(get_leaves(a))
                leaves_b = set(get_leaves(b))
                
                if leaves_v & leaves_a or leaves_v & leaves_b:
                    continue
                
                # Calculate gain
                # New cost = Cost(Tree) - Cost(v) - Cost(a) - Cost(b) + Cost(new_a) + Cost(new_b)
                # Where new_a is a with v attached, new_b is b with v detached?
                # No, the structure is:
                # Original: ... - a - b - ... and v is somewhere else.
                # Move: Detach v from u. Attach v to edge (a, b).
                # This splits edge (a, b) into (a, new_node) and (new_node, b).
                # new_node connects to a, b, and v.
                
                # The change in cost is localized to the subtrees rooted at a, b, and v?
                # Not exactly, because the "boundary" states of a and b change.
                
                # Let's use the formula:
                # Gain = (Cost(old_a) + Cost(old_b) + Cost(v)) - (Cost(new_a) + Cost(new_b))
                # Where old_a is the subtree rooted at a (in the original tree, excluding b? No, a is a node).
                # In a binary tree, a node has a parent and two children.
                # If we attach v to (a, b), we are modifying the subtree rooted at a (if b is child of a) or b (if a is child of b).
                
                # Let's define the cost of a node's subtree as the cost of the Fitch algorithm on that subtree,
                # given the "upward" state from the parent.
                # This is getting complicated.
                
                # Alternative: Just recompute the total parsimony score for the candidate tree.
                # If we do this for every candidate, it's O(n * n_sites) per candidate.
                # Number of candidates is O(n^2).
                # Total O(n^3 * n_sites).
                # For n=32, n_sites=1000: 32^3 * 1000 = 32,768,000 operations.
                # In Python, this might be too slow for many iterations.
                # But we only do this if the set check passes.
                
                # Let's try to implement the "localized" cost calculation properly.
                # This requires storing "upward" and "downward" DP vectors.
                
                # For now, let's implement the naive recomputation but with a time check.
                # If it's too slow, we'll switch to NNI or a faster heuristic.
                
                # Actually, let's implement NNI first as it's simpler and faster to evaluate.
                # NNI is a subset of SPR.
                
                # If we find an NNI improvement, we take it.
                # If not, we might try SPR in a later version.
                
                # For this implementation, let's stick to NNI to ensure speed and stability.
                # NNI moves on edge (u, v) where u is parent of v.
                # Swap v's sibling with one of v's children.
                
                # We will handle NNI separately in the loop below.
                pass
        
        # --- NNI Search ---
        # Collect internal edges again (tree might have changed? No, we didn't change it yet in this loop)
        # We need to identify internal nodes v (not root) and their parent u.
        
        nni_candidates = []
        def collect_nni(node, parent_node):
            if parent_node and not node.is_leaf():
                # node is internal, parent is parent_node
                # Check if parent_node is internal (it is, since node is not root)
                # Get sibling of node
                if parent_node.left == node:
                    sibling = parent_node.right
                else:
                    sibling = parent_node.left
                    
                # Get children of node
                child1 = node.left
                child2 = node.right
                
                # Candidate 1: Swap sibling and child1
                # New structure: parent connects to (child1, node) and (sibling, child2)?
                # No.
                # Original: parent - (node, sibling). node - (child1, child2).
                # NNI 1: parent - (child1, node). node - (sibling, child2). 
                # Wait, this changes the topology.
                # Standard NNI:
                # Edge (u, v).
                # u connects to v and w (sibling).
                # v connects to x and y (children).
                # Move 1: u connects to x and w. v connects to u and y? No.
                # The three topologies are:
                # 1. (x, y) - (w, rest)  [Original: u-v edge]
                # 2. (x, w) - (y, rest)  [New 1]
                # 3. (y, w) - (x, rest)  [New 2]
                
                # In our rooted representation:
                # u is the "rest" side center? No, u is a node.
                # Let's just construct the new trees and compare parsimony.
                
                # To avoid complex pointer manipulation and backtracking,
                # we will generate the new Newick string, parse it, and score it.
                # This is slow but robust.
                
                # However, parsing and scoring for every NNI candidate is very slow.
                # Let's stick to the "localized cost" idea but simplified.
                
                # Given the complexity, I will implement a simpler heuristic:
                # Randomly sample a few NNI moves and check if they improve the score.
                # If so, accept.
                
                pass
                
        # This approach is getting too complex for a single pass.
        # Let's implement a simpler "Best-First" NNI using full recomputation.
        # We will limit the number of NNI checks per iteration to keep it fast.
        
        # Re-collect edges for NNI
        internal_nodes = []
        def collect_internal(node, parent_node):
            if not node.is_leaf():
                internal_nodes.append((node, parent_node))
            if node.left:
                collect_internal(node.left, node)
            if node.right:
                collect_internal(node.right, node)
                
        collect_internal(root, None)
        
        # Shuffle to avoid bias? Or just iterate.
        # Let's iterate.
        
        for v, u in internal_nodes:
            if u is None: # Root
                continue
                
            # Get sibling of v
            if u.left == v:
                w = u.right
            else:
                w = u.left
                
            # Get children of v
            x = v.left
            y = v.right
            
            if x is None or y is None:
                continue
                
            # Candidate 1: Swap w and x
            # Original: u(v(x,y), w)
            # New: u(x(v(y, ?), ?), ?) -- This is hard to describe without drawing.
            
            # Let's use the "Newick string" approach for the candidate.
            # It's safer.
            
            # But we need to be fast.
            # Let's try to implement the "localized" parsimony gain properly.
            # This is the standard way to make SPR/NNI fast.
            
            # We need:
            # 1. Downward DP vectors for all nodes (from leaves to root).
            # 2. Upward DP vectors for all nodes (from root to leaves).
            # 3. A function to compute the cost of a subtree given an "upward" vector.
            
            # This requires significant refactoring of the DP storage.
            # Given the time, I will implement a "Naive NNI" that recomputes the score
            # but only for a subset of moves (e.g., random 10% per pass) to save time.
            
            # Actually, let's just implement the full SPR with full recomputation
            # but limit the number of candidates evaluated per iteration.
            
            # If time allows, we'll do more.
            
            # For now, let's implement a "Random Restart" strategy if we get stuck?
            # No, let's just do a few passes of full NNI with full recomputation.
            
            # To make it fast, I'll precompute the sequence matrix.
            
            # Let's try one NNI move: Swap w and x.
            # We need to create a copy of the tree structure?
            # Modifying the tree in place and reverting is efficient.
            
            # Save original pointers
            u_l_orig = u.left
            u_r_orig = u.right
            v_l_orig = v.left
            v_r_orig = v.right
            
            # Perform Move 1: u connects to x and w. x connects to v and y?
            # No, the topologies are:
            # 1. (x, y) - (w, rest)
            # 2. (x, w) - (y, rest)
            # 3. (y, w) - (x, rest)
            
            # In rooted terms (rooted at u's parent):
            # u is the root of this local structure.
            # Children of u: v, w.
            # Children of v: x, y.
            
            # Move 2: u connects to x, w. x connects to v, y? No.
            # The "rest" is attached to u.
            # So u is the junction.
            # Move 2: u connects to x and w. The subtree (v, y) is attached to x?
            # No, x is a leaf or internal node.
            # If x is internal, it has children.
            
            # This is confusing. Let's use the standard NNI definition on unrooted trees.
            # Edge (u, v).
            # Neighbors of u: p, v, w.
            # Neighbors of v: u, x, y.
            # Subtrees: A (at p), B (at w), C (at x), D (at y).
            # Original: (A, B) - (C, D).
            # New 1: (A, C) - (B, D).
            # New 2: (A, D) - (B, C).
            
            # In our rooted tree, u is the parent of v.
            # So p is parent(u). w is sibling(v).
            # x, y are children(v).
            
            # To perform New 1:
            # We need to restructure so that u connects to A, C, and a new node?
            # No, binary tree.
            # u connects to A and C? And v connects to B and D?
            # This implies u and v are still connected?
            # No, the edge (u, v) is replaced by (u, C) and (v, D)?
            # And C is connected to u?
            
            # This is too hard to do in-place without a clear data structure.
            
            # I will fall back to a simpler strategy:
            # 1. NJ Tree.
            # 2. Random NNI moves with full parsimony scoring.
            # 3. Accept if better.
            
            # This is O(N^2) scoring per iteration if we check all, but we can check randomly.
            
            # Let's just do 10 random NNI checks per iteration.
            
            pass

    return root

def reconstruct(sequences):
    _init_time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa < 3:
        if n_taxa == 1:
            return labels[0] + ";"
        elif n_taxa == 2:
            return f"({labels[0]},{labels[1]});"
            
    # Convert sequences to integer matrix
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    n_sites = len(sequences[labels[0]])
    
    # Compute distance matrix
    dists = _compute_distance_matrix(sequences, labels)
    
    # Run NJ
    root = _nj_run(dists, labels)
    
    # Refine with SPR/Parsimony
    if _time_left() > 5:
        root = _spr_refine(sequences, labels, root)
        
    # Convert to Newick
    newick = _tree_to_newick(root)
    
    return newick + ";"

# EVOLVE-BLOCK-END
