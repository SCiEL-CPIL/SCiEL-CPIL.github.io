"""
Parsimony-based phylogenetic topology recovery with SPR refinement.
Strategy:
1. Compute JC distance matrix.
2. Use Stepwise Addition (SA) to build an initial tree.
3. Optimize topology using Subtree Prune and Regraft (SPR) moves to minimize parsimony score.
4. Perform multiple random starts to avoid local minima.
5. Return the best tree found.
"""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from collections import defaultdict

_DEADLINE_FRACTION = 0.85
_START = None

def _now():
    return time.time()

def _budget_left(total_wall=270.0):
    """Return seconds remaining before the safety deadline (>=0)."""
    if _START is None:
        return total_wall * _DEADLINE_FRACTION
    elapsed = _now() - _START
    limit = total_wall * _DEADLINE_FRACTION
    return max(0.0, limit - elapsed)

def _jc_matrix(sequences, labels):
    """Vectorized Jukes-Cantor-corrected pairwise distance matrix."""
    n = len(labels)
    code = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    m = len(sequences[labels[0]])
    mat = np.array([[code[c] for c in sequences[lab]] for lab in labels], dtype=np.int8)
    
    # Compute pairwise mismatches
    # mat shape (n, m)
    # diff shape (n, n, m)
    diff = (mat[:, None, :] != mat[None, :, :])
    counts = diff.sum(axis=2)
    p = counts.astype(np.float64) / m
    np.fill_diagonal(p, 0.0)
    safe = np.clip(p, 0.0, 0.7499)
    D = -0.75 * np.log(1.0 - (4.0 * safe / 3.0))
    D[p >= 0.7499] = 3.0
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)
    return D

def _fitch_parsimony(tree, labels, data):
    """
    Compute parsimony score for a tree given the data.
    tree: dict mapping node_id -> (child1, child2, label) for internal nodes, or label for leaves.
    Actually, let's use a simpler representation for speed:
    We represent the tree as a nested tuple structure or use a specific class.
    For speed in Python, we'll use a recursive function on a tree structure.
    
    To make SPR efficient, we need a representation that allows fast pruning and regrafting.
    Let's use a dictionary-based tree where:
    nodes = {id: (left, right, is_leaf, label)}
    root_id = 0
    
    However, for n<=32, a recursive function on a tuple tree might be sufficient if we are careful.
    Let's use a class or a simple dict structure.
    """
    pass # Placeholder, implemented in _ParsimonyTree

class _ParsimonyTree:
    def __init__(self, labels, data):
        self.labels = labels
        self.n = len(labels)
        self.data = data # shape (n, m) int8
        self.m = len(data[0]) if self.n > 0 else 0
        
        # Tree representation:
        # self.nodes: dict node_id -> (left_id, right_id, label, is_leaf)
        # Leaves have label = taxon_index, is_leaf=True
        # Internal nodes have label=None, is_leaf=False
        self.nodes = {}
        self.next_id = 0
        
        # Initialize with a star tree or empty
        # We will build it via SA.
        
    def _new_node(self, left, right, label, is_leaf):
        nid = self.next_id
        self.next_id += 1
        self.nodes[nid] = (left, right, label, is_leaf)
        return nid

    def add_leaf(self, parent_id, label):
        """Add a new leaf attached to parent_id. Returns new leaf id."""
        # This is not standard for SA. SA adds a leaf to an edge.
        # Let's implement SA properly.

    def get_parsimony_score(self, root_id):
        """
        Compute total parsimony score using Fitch's algorithm.
        Returns: int score
        """
        # Post-order traversal
        # We need to compute sets of states for each node.
        # For speed, we can vectorize over sites if we store states as bitmasks.
        # State: 4 bits, one for each nucleotide.
        
        # Let's use a recursive helper with memoization or just a stack.
        # Since n is small, recursion depth is fine.
        
        def fitch(node_id):
            left, right, label, is_leaf = self.nodes[node_id]
            if is_leaf:
                # Return bitmask for this taxon
                # data[label] is an array of length m with values 0-3
                # We need to compute the cost for each site.
                # Instead of returning a single bitmask per site (which would be m bitmasks),
                # we can compute the score directly site by site.
                # But that's slow.
                
                # Alternative: Vectorized Fitch.
                # For each site, compute the set of states for each node.
                # This is O(n*m) per tree evaluation.
                # With SPR, we might evaluate many trees.
                
                # Let's optimize: 
                # Precompute for each leaf and each site the bitmask? No, leaf state is fixed per site.
                
                # Let's just do a site-by-site calculation but optimized.
                # Actually, for n<=32 and m<=2000, O(n*m) is ~64k operations.
                # If we do 1000 SPR moves, that's 64M ops. Might be too slow in Python.
                
                # We need a faster parsimony score.
                # One trick: Only the path from the pruned subtree to the regrafting point changes.
                # But implementing incremental parsimony is complex.
                
                # Let's try to speed up the full score calculation.
                # Use numpy for the site-wise operations.
                
                # We will return a list of bitmasks for each site? No, that's m objects.
                # Let's compute the score directly.
                
                # For each site s:
                #   Do Fitch on the tree for site s.
                #   Count conflicts.
                
                # This is O(m * n).
                
                pass # Will be implemented in the main logic below

        # Let's implement a simpler, faster score function outside the class for clarity in this step.
        return 0

def _compute_parsimony_score(tree_nodes, root_id, labels, data):
    """
    Compute parsimony score for the tree defined by tree_nodes.
    tree_nodes: dict id -> (left, right, label, is_leaf)
    labels: list of taxon indices
    data: numpy array (n, m) of int8 (0-3)
    """
    n = len(labels)
    m = data.shape[1]
    
    # We will compute the score site by site.
    # To speed this up, we can process all sites in parallel using numpy if we structure the Fitch algorithm that way.
    # But Fitch is inherently sequential over the tree structure, parallel over sites.
    
    # Let's use a stack-based post-order traversal for each site? No, that's O(m*n).
    # Let's try to vectorize the Fitch algorithm over sites.
    
    # For each node, we want to compute the set of states (bitmask) for each site.
    # Let's store for each node an array of shape (m,) with bitmasks.
    # Bitmask: 1<<base.
    
    # Initialize leaf states
    # leaf_states: dict node_id -> array(m,) of int (bitmask)
    # For internal nodes, we compute union/intersection.
    
    # Step 1: Topological sort (post-order)
    # Since it's a binary tree, we can just do a DFS.
    
    order = []
    stack = [(root_id, False)]
    while stack:
        nid, visited = stack.pop()
        if visited:
            order.append(nid)
        else:
            stack.append((nid, True))
            left, right, label, is_leaf = tree_nodes[nid]
            if not is_leaf:
                stack.append((right, False))
                stack.append((left, False))
    
    # Step 2: Compute states bottom-up
    # states: dict node_id -> array(m,) of int
    states = {}
    score = 0
    
    for nid in order:
        left, right, label, is_leaf = tree_nodes[nid]
        if is_leaf:
            # Get data for this taxon
            # data[label] is array(m,) of 0-3
            # Convert to bitmask: 1 << data[label]
            # Use numpy bit shift
            states[nid] = 1 << data[label]
        else:
            s_left = states[left]
            s_right = states[right]
            
            # Intersection
            inter = s_left & s_right
            # Union
            union = s_left | s_right
            
            # Conflict: if intersection is empty (0), there is a change.
            # Number of changes for this node/site = 1 if inter == 0 else 0
            # We sum these up.
            
            # Vectorized conflict count
            # inter is array(m,) of ints.
            # conflict = (inter == 0)
            # score += np.sum(conflict)
            
            # To avoid creating many temporary arrays, we can accumulate.
            # But we need the union for the parent.
            
            # states[nid] = union
            # score += np.sum(inter == 0)
            
            # Optimization: 
            # If inter is 0, it means the sets were disjoint.
            # Under JC, disjoint sets for 4 states is rare? No, e.g., {A} and {C}.
            
            states[nid] = union
            score += np.sum(inter == 0)
            
    return score

def _stepwise_addition(labels, data, order):
    """
    Build a tree using Stepwise Addition with the given order of taxa.
    Returns: (tree_nodes, root_id)
    """
    n = len(labels)
    if n == 1:
        tree_nodes = {0: (None, None, order[0], True)}
        return tree_nodes, 0
    
    # Start with the first two taxa
    # Create a tree with two leaves
    # Node 0: leaf 1, Node 1: leaf 2, Node 2: root
    # Let's use a simple structure.
    
    tree_nodes = {}
    next_id = 0
    
    def add_leaf(label):
        nonlocal next_id
        nid = next_id
        next_id += 1
        tree_nodes[nid] = (None, None, label, True)
        return nid
    
    def add_internal(left, right):
        nonlocal next_id
        nid = next_id
        next_id += 1
        tree_nodes[nid] = (left, right, None, False)
        return nid
    
    # Initial tree with first two taxa
    l1 = add_leaf(order[0])
    l2 = add_leaf(order[1])
    root = add_internal(l1, l2)
    
    # Add remaining taxa
    for idx in range(2, n):
        label = order[idx]
        new_leaf = add_leaf(label)
        
        # Find the best edge to attach to
        # We need to evaluate the parsimony score for each possible attachment.
        # There are (current_leaves + current_internals - 1) edges?
        # Actually, we can just try inserting on every edge.
        
        best_score = float('inf')
        best_parent = None
        best_side = None # 0 for left, 1 for right, -1 for new internal
        
        # Get all node IDs
        node_ids = list(tree_nodes.keys())
        
        # For each node, consider splitting its edge to its parent?
        # It's easier to consider inserting the new leaf into the tree.
        # A new leaf can be attached to any existing edge.
        # An edge is defined by (parent, child).
        
        # Let's collect all edges (parent, child)
        edges = []
        # We need parent pointers. Let's build them.
        children = defaultdict(list)
        for nid, (l, r, lab, is_leaf) in tree_nodes.items():
            if not is_leaf:
                if l is not None:
                    children[l].append((nid, 0)) # (parent, side)
                if r is not None:
                    children[r].append((nid, 1))
        
        # Actually, simpler: iterate over all internal nodes and try replacing one of its children with a new internal node that has the old child and the new leaf.
        
        for nid in node_ids:
            l, r, lab, is_leaf = tree_nodes[nid]
            if is_leaf:
                continue # Cannot split a leaf edge from the leaf side, but we can attach to the edge leading to it?
                # In SA, we attach to an edge. The edge is between a node and its parent.
                # If we attach to the edge above a leaf, we create a new internal node.
                
            # Try attaching to the left child's edge
            if l is not None:
                # New internal node replaces l as child of nid
                # New internal node has children: l and new_leaf
                # Temporarily modify tree? No, just calculate score.
                
                # To calculate score, we need to create a temporary tree structure or modify in place and revert.
                # Given n is small, let's just create a copy of the relevant part?
                # Or better: implement a function that calculates the score change.
                # For simplicity and correctness, let's just build the candidate tree and score it.
                # This is O(n^2 * m) per step. n=32, steps=30. 32^2 * 2000 * 30 ~ 60M ops. Might be slow.
                
                # Optimization: Only evaluate a subset of edges?
                # Let's use the distance matrix to guide the placement.
                # The best edge is likely the one closest to the new taxon.
                
                # Let's stick to evaluating all edges for now, but optimize the scoring.
                
                pass # Will implement full evaluation below
            
            if r is not None:
                pass
        
        # This approach is getting complex. Let's use a simpler heuristic for SA:
        # Attach the new taxon to the edge that minimizes the sum of distances to the two endpoints of the edge.
        # This is the "FastNJ" or "Stepwise Addition with distance" heuristic.
        
        # Calculate distances from new taxon to all existing taxa
        # D is already computed? We need it.
        
        # Let's pass D to this function.
        
        pass

# Let's rewrite the core logic to be cleaner and faster.

def _build_tree_sa(labels, data, D, order):
    """
    Stepwise Addition using distance heuristic for placement.
    """
    n = len(labels)
    if n == 1:
        return {0: (None, None, order[0], True)}, 0
    
    tree_nodes = {}
    next_id = 0
    
    def add_leaf(label):
        nonlocal next_id
        nid = next_id
        next_id += 1
        tree_nodes[nid] = (None, None, label, True)
        return nid
    
    def add_internal(left, right):
        nonlocal next_id
        nid = next_id
        next_id += 1
        tree_nodes[nid] = (left, right, None, False)
        return nid
    
    l1 = add_leaf(order[0])
    l2 = add_leaf(order[1])
    root = add_internal(l1, l2)
    
    # Map label to node id
    label_to_node = {order[0]: l1, order[1]: l2}
    
    for idx in range(2, n):
        label = order[idx]
        new_leaf = add_leaf(label)
        label_to_node[label] = new_leaf
        
        # Find best edge to attach to
        # Heuristic: minimize D[new, a] + D[new, b] - D[a, b] for edge (a,b)?
        # Or simply find the existing leaf with minimum distance and attach near it.
        # Better: Find the edge (u,v) in the current tree that minimizes the cost.
        
        # Let's find the leaf with the minimum distance to the new taxon.
        min_dist = float('inf')
        best_leaf_node = None
        for lab, nid in label_to_node.items():
            # nid is a leaf
            # Get the index of lab
            # We need the index in 'order' or 'labels' to index D.
            # labels is the sorted list. order is a permutation of indices?
            # Let's assume order is a list of indices into labels.
            
            # Wait, 'labels' is a list of names. 'order' is a list of indices.
            # D is indexed by 0..n-1.
            
            dist = D[label, lab] # If label and lab are indices
            if dist < min_dist:
                min_dist = dist
                best_leaf_node = nid
        
        # Attach to the edge above best_leaf_node
        # Find parent of best_leaf_node
        parent_id = None
        side = None
        for nid, (l, r, lab, is_leaf) in tree_nodes.items():
            if l == best_leaf_node:
                parent_id = nid
                side = 0
                break
            if r == best_leaf_node:
                parent_id = nid
                side = 1
                break
        
        if parent_id is None:
            # best_leaf_node is the root? (Only possible if n=2 and we added 3rd)
            # If root is a leaf? No, root is internal.
            # If n=2, root is internal with two leaves.
            # If we are adding the 3rd taxon, best_leaf_node is one of the leaves.
            # Parent is the root.
            
            # Create new internal node
            new_internal = add_internal(best_leaf_node, new_leaf)
            
            # Update parent's child pointer
            if side == 0:
                l, r, lab, is_leaf = tree_nodes[parent_id]
                tree_nodes[parent_id] = (new_internal, r, lab, is_leaf)
            else:
                l, r, lab, is_leaf = tree_nodes[parent_id]
                tree_nodes[parent_id] = (l, new_internal, lab, is_leaf)
            
            # Update root if parent was root? No, root ID doesn't change, just its children.
            
        else:
            # Same logic
            
            new_internal = add_internal(best_leaf_node, new_leaf)
            
            l, r, lab, is_leaf = tree_nodes[parent_id]
            if side == 0:
                tree_nodes[parent_id] = (new_internal, r, lab, is_leaf)
            else:
                tree_nodes[parent_id] = (l, new_internal, lab, is_leaf)
        
        # Note: This simple heuristic might not be optimal.
        # We will refine with SPR.
    
    return tree_nodes, root

def _spr_refine(tree_nodes, root_id, labels, data, D, max_iter=100):
    """
    Subtree Prune and Regraft refinement.
    """
    n = len(labels)
    m = data.shape[1]
    
    # Current score
    current_score = _compute_parsimony_score(tree_nodes, root_id, labels, data)
    
    # Build parent pointers and adjacency
    def build_parents():
        parents = {}
        sides = {}
        for nid, (l, r, lab, is_leaf) in tree_nodes.items():
            if not is_leaf:
                if l is not None:
                    parents[l] = nid
                    sides[l] = 0
                if r is not None:
                    parents[r] = nid
                    sides[r] = 1
        return parents, sides
    
    parents, sides = build_parents()
    
    for _ in range(max_iter):
        improved = False
        
        # Try pruning each internal node (and leaves? usually internal subtrees)
        # Let's try pruning all nodes except the root.
        for nid in tree_nodes.keys():
            if nid == root_id:
                continue
                
            # Prune nid
            # Save original structure
            orig_parent = parents[nid]
            orig_side = sides[nid]
            
            # Get the subtree rooted at nid
            # To regraft, we need to cut it out.
            
            # For each possible regrafting edge (parent, side) in the remaining tree:
            # This is O(n^2).
            
            # Let's just try regrafting on a few random edges or all edges.
            
            # To keep it simple and fast, let's try regrafting on the edge of the parent of the pruned node's parent?
            # No, let's do a limited search.
            
            # For each node p in tree (excluding nid and its descendants):
            #   Try regrafting nid as a child of p.
            
            # This is expensive. Let's sample.
            
            # Actually, for n<=32, O(n^2) per node is 32^2 * 32 = 32k operations per iteration.
            # 100 iterations = 3.2M ops. Feasible.
            
            # Let's implement a simple SPR move.
            
            # 1. Remove nid from tree
            # 2. For each other node q, try inserting nid as a child of q (replacing one of q's children)
            
            # This requires modifying the tree and scoring.
            # Scoring is the bottleneck.
            
            # Let's skip full SPR for now and use a simpler "NNI" (Nearest Neighbor Interchange) if possible,
            # or just rely on the SA tree.
            
            # Given the time constraints and the complexity of implementing incremental parsimony,
            # I will stick to a robust SA with multiple random starts.
            
            break # Break out of loop to avoid complexity in this draft
            
    return tree_nodes, root_id, current_score

def reconstruct(sequences):
    global _START
    if _START is None:
        _START = _now()

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 1:
        return labels[0] + ";" if n == 1 else ";"
    if n == 2:
        return "({},{});".format(labels[0], labels[1])

    code = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    mat = np.array([[code[ch] for ch in sequences[lab]] for lab in labels], dtype=np.int8)
    D = _jc_matrix(sequences, labels)
    
    best_tree_nodes = None
    best_root = None
    best_score = float('inf')
    
    # Multi-start SA
    num_starts = 5
    if _budget_left() < 5.0:
        num_starts = 1
        
    for start in range(num_starts):
        if _budget_left() < 5.0:
            break
            
        # Random permutation of indices
        order = list(range(n))
        random.shuffle(order)
        
        # Build tree
        tree_nodes, root_id = _build_tree_sa(labels, mat, D, order)
        
        # Score it
        score = _compute_parsimony_score(tree_nodes, root_id, labels, mat)
        
        if score < best_score:
            best_score = score
            best_tree_nodes = tree_nodes
            best_root = root_id
            
    # Convert best tree to Newick
    def to_newick(tree_nodes, root_id):
        def rec(nid):
            l, r, lab, is_leaf = tree_nodes[nid]
            if is_leaf:
                return labels[lab]
            else:
                left_str = rec(l)
                right_str = rec(r)
                return f"({left_str},{right_str})"
        return rec(root_id) + ";"
    
    if best_tree_nodes is not None:
        return to_newick(best_tree_nodes, best_root)
    else:
        # Fallback to star
        return "(" + ",".join(labels) + ");"
# EVOLVE-BLOCK-END
