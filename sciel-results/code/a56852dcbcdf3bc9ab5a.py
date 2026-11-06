import time
import math
import numpy as np
from collections import defaultdict

_START_TIME = None
_TIME_LIMIT = 260.0  # Safety margin for 270s limit

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_left():
    _init_time()
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC distances."""
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

class Node:
    __slots__ = ['label', 'left', 'right', 'length', 'leaves']
    def __init__(self, label=None, left=None, right=None, length=0.0):
        self.label = label
        self.left = left
        self.right = right
        self.length = length
        self.leaves = None # Cache for frozenset of leaf labels

    def is_leaf(self):
        return self.label is not None

    def get_leaves(self):
        if self.leaves is not None:
            return self.leaves
        if self.is_leaf():
            self.leaves = frozenset([self.label])
        else:
            l_leaves = self.left.get_leaves()
            r_leaves = self.right.get_leaves()
            self.leaves = l_leaves | r_leaves
        return self.leaves

    def __repr__(self):
        if self.is_leaf():
            return self.label
        return f"({self.left},{self.right})"

def _nj_build(dists, labels):
    """Standard Neighbor Joining. Returns root Node."""
    n = len(labels)
    if n < 3:
        if n == 1:
            return Node(label=labels[0])
        else:
            root = Node()
            root.left = Node(label=labels[0])
            root.right = Node(label=labels[1])
            return root

    nodes = [Node(label=labels[i]) for i in range(n)]
    current_d = dists.copy()
    active_nodes = nodes[:]
    node_id_map = {id(node): node for node in nodes} # Not strictly needed if we use objects directly
    # We will maintain a list of active nodes and a distance matrix corresponding to their order.
    
    # For efficiency in Python, we'll keep a list of active node objects and a numpy matrix of distances between them.
    # When we join i and j, we create a new node, remove i and j from active list, add new node, and update matrix.
    
    active = active_nodes[:]
    d = current_d
    
    while len(active) > 2:
        n_curr = len(active)
        r = d.sum(axis=1)
        q = (n_curr - 2) * d - r[:, None] - r[None, :]
        np.fill_diagonal(q, np.inf)
        
        min_idx = np.unravel_index(np.argmin(q), q.shape)
        i, j = min_idx
        
        node_i = active[i]
        node_j = active[j]
        
        d_ij = d[i, j]
        alpha = (r[i] - r[j]) / (2.0 * (n_curr - 2))
        len_i = max(0.0, 0.5 * d_ij + alpha)
        len_j = max(0.0, 0.5 * d_ij - alpha)
        
        new_node = Node()
        new_node.left = node_i
        new_node.right = node_j
        node_i.length = len_i
        node_j.length = len_j
        
        # Remove i and j from active, add new_node
        # We need to update the distance matrix.
        # New active list: all except i, j, plus new_node at the end.
        new_active = []
        for k in range(n_curr):
            if k != i and k != j:
                new_active.append(active[k])
        new_active.append(new_node)
        
        n_new = len(new_active)
        new_d = np.zeros((n_new, n_new))
        
        # Map new indices to old indices for copying
        # The new node is at index n_new - 1
        # For existing nodes, we need to find their old index.
        # This is O(N^2) per step if done naively, but N is small (<=32).
        
        # Optimization: Pre-compute mapping from new index to old index
        old_indices = {}
        for new_idx, node in enumerate(new_active):
            if node is new_node:
                continue
            # Find old index of this node in active
            # Since we removed i and j, we can just iterate
            for old_idx, old_node in enumerate(active):
                if old_node is node:
                    old_indices[new_idx] = old_idx
                    break
        
        for p1 in range(n_new):
            for p2 in range(p1 + 1, n_new):
                if p1 == n_new - 1 or p2 == n_new - 1:
                    # One is the new node
                    other_p = p2 if p1 == n_new - 1 else p1
                    other_node = new_active[other_p]
                    old_other_idx = old_indices[other_p]
                    
                    # Distance from new node to other node
                    # d(new, x) = 0.5 * (d(i, x) + d(j, x) - d(i, j))
                    d_ix = d[i, old_other_idx]
                    d_jx = d[j, old_other_idx]
                    d_new = 0.5 * (d_ix + d_jx - d_ij)
                    d_new = max(0.0, d_new)
                    
                    new_d[p1, p2] = d_new
                    new_d[p2, p1] = d_new
                else:
                    old_p1 = old_indices[p1]
                    old_p2 = old_indices[p2]
                    new_d[p1, p2] = d[old_p1, old_p2]
                    new_d[p2, p1] = d[old_p1, old_p2]
        
        active = new_active
        d = new_d
        
    # Connect the last two
    if len(active) == 2:
        node1 = active[0]
        node2 = active[1]
        root = Node()
        root.left = node1
        root.right = node2
        d_12 = d[0, 1]
        node1.length = 0.5 * d_12
        node2.length = 0.5 * d_12
        return root
        
    return active[0]

def _get_splits(root):
    """Get canonical splits from a tree."""
    splits = set()
    total_leaves = root.get_leaves()
    n_total = len(total_leaves)
    
    def traverse(node):
        if node.is_leaf():
            return
        leaves = node.get_leaves()
        if 1 < len(leaves) < n_total:
            # Canonicalize: use the smaller side, or if equal, the one with lexicographically smaller min label?
            # Simpler: if size > n/2, use complement.
            if len(leaves) > n_total // 2:
                # Need complement. 
                # To avoid expensive set subtraction in tight loops, 
                # we can just store the frozenset and handle matching in consensus.
                # But for RF distance, we need consistent representation.
                # Let's store the frozenset of the smaller side.
                # If sizes are equal, we need a tie-breaker. 
                # Let's just store the frozenset and rely on the fact that 
                # for a given tree, the split is unique. 
                # Actually, for consensus, we need to count identical splits.
                # A split {A, B} is the same as {B, A}.
                # So we should canonicalize.
                # If len(leaves) <= n_total // 2:
                #    splits.add(leaves)
                # else:
                #    splits.add(total_leaves - leaves)
                # 
                # But total_leaves - leaves is O(N). 
                # Given N<=32, it's fine.
                comp = total_leaves - leaves
                splits.add(comp)
            else:
                splits.add(leaves)
                
        traverse(node.left)
        traverse(node.right)
        
    traverse(root)
    return splits

def _consensus_tree(trees, labels):
    """
    Build a consensus tree from a list of trees (root Nodes).
    Uses majority vote on splits.
    Returns a root Node or None if no consensus.
    """
    if not trees:
        return None
        
    # Collect all splits with counts
    split_counts = defaultdict(int)
    for tree in trees:
        splits = _get_splits(tree)
        for s in splits:
            split_counts[s] += 1
            
    # Filter splits that appear in more than 50% of trees
    threshold = len(trees) / 2.0
    valid_splits = [s for s, c in split_counts.items() if c > threshold]
    
    if not valid_splits:
        # Fallback to first tree? Or star?
        # If no consensus, return the first tree (or a star).
        # Returning first tree is safer than a star which is likely wrong.
        return trees[0]
        
    # Reconstruct tree from splits
    # This is a known NP-hard problem in general, but for small N and consistent splits from NJ,
    # we can use a simple recursive partitioning.
    # 1. Find the split that partitions the leaves most evenly or is a "root" split.
    # Actually, any valid set of splits from a tree forms a laminar family (if rooted) or can be rooted.
    # Since these are unrooted splits, we can pick any split as the "root" edge.
    # Let's pick the split with size closest to N/2 to balance the tree.
    
    # Sort splits by size to find a good root
    # A split is a frozenset of leaves.
    # We need to build the tree.
    
    # Algorithm:
    # 1. Start with the set of all leaves.
    # 2. Find a split S that is a subset of current leaves.
    # 3. The edge corresponding to S divides current leaves into S and Current-S.
    # 4. Recursively build subtrees for S and Current-S.
    
    all_leaves = frozenset(labels)
    
    def build_subtree(leaves_set):
        if len(leaves_set) <= 1:
            if len(leaves_set) == 1:
                label = list(leaves_set)[0]
                return Node(label=label)
            else:
                return None
        
        # Find a valid split that is a proper subset of leaves_set
        # and is not the whole set.
        # We want to find a split S such that S is a subset of leaves_set.
        # Note: The splits stored are canonical (smaller side).
        # If S is in valid_splits, then either S is subset of leaves_set or (leaves_set - S) is subset of leaves_set?
        # Actually, if the tree is consistent, for any node, the set of leaves in its subtree is a union of splits.
        
        # Let's iterate through valid_splits and find one that is a proper subset of leaves_set.
        # To ensure we don't pick the "whole" set as a subset, we check len(S) < len(leaves_set).
        
        candidate_splits = []
        for s in valid_splits:
            if s <= leaves_set and len(s) < len(leaves_set) and len(s) > 1:
                # Also check if the complement is the other part?
                # If s is a split in the original tree, and we are looking at a subtree,
                # s must be a subset of the subtree leaves.
                # Is it possible that s is not a subset, but its complement is?
                # If s is the "smaller" side of the split, and we are in a subtree that contains the "larger" side...
                # Wait, the splits are from the *unrooted* tree.
                # If we root the tree, every edge defines a split.
                # When we recurse, we are effectively rooting at that edge.
                # So if we pick split S to be the left child, the leaves in the left child are S.
                # The leaves in the right child are leaves_set - S.
                # We need S to be a valid split.
                # The split is {S, Leaves-S}.
                # Our stored splits are canonicalized to the smaller side.
                # So if len(S) <= N/2, S is stored. If len(S) > N/2, Leaves-S is stored.
                
                # Case 1: s is stored. s <= leaves_set.
                # This implies s is one side of the partition.
                # The other side is leaves_set - s.
                # We need to check if {s, leaves_set - s} is a valid split in the consensus.
                # Since s is in valid_splits, it was frequent.
                # We assume consistency.
                
                candidate_splits.append(s)
                
            # Check if the complement of s (relative to all leaves) is a subset?
            # No, we are working within leaves_set.
            # If s is not a subset of leaves_set, could leaves_set - s be a subset?
            # That would mean s contains some leaves outside leaves_set.
            # If s is a split in the global tree, and we are in a subtree,
            # the split must be "inside" the subtree or "outside".
            # If it's inside, one side is a subset of leaves_set.
            # So checking s <= leaves_set is sufficient if we consider both s and its complement?
            # No, s is the *canonical* representation.
            # If the actual side in the subtree is the *larger* side, then s (the smaller side) is the *complement* of the side in the subtree.
            # So if s is not a subset of leaves_set, but (all_leaves - s) is a subset of leaves_set?
            # Let's check:
            # Let A be the side of the split that is inside leaves_set.
            # If A is the smaller side globally, then s = A. Then s <= leaves_set.
            # If A is the larger side globally, then s = all_leaves - A.
            # Then A = all_leaves - s.
            # We need to check if A <= leaves_set.
            # This is equivalent to: leaves_set - A is a subset of all_leaves - leaves_set?
            # This is getting complicated.
            
            # Simpler approach:
            # For each split s in valid_splits:
            #   side1 = s
            #   side2 = all_leaves - s
            #   If side1 <= leaves_set and len(side1) < len(leaves_set) and len(side1) > 1:
            #       candidate = side1
            #   Elif side2 <= leaves_set and len(side2) < len(leaves_set) and len(side2) > 1:
            #       candidate = side2
            #   If candidate found, use it.
            
        # Let's redo the loop properly
        found_split = None
        for s in valid_splits:
            side1 = s
            side2 = all_leaves - s
            
            if side1 <= leaves_set and 1 < len(side1) < len(leaves_set):
                found_split = side1
                break
            elif side2 <= leaves_set and 1 < len(side2) < len(leaves_set):
                found_split = side2
                break
                
        if found_split is None:
            # No split found? This means leaves_set is a clade that doesn't correspond to a single split in the consensus?
            # Or it's a star?
            # If no split, it might be a polytomy or a star.
            # For now, return a star tree for this subset? Or a random binary tree?
            # A star is safer for RF if we don't know.
            # But NJ trees are binary.
            # If we can't find a split, it implies the consensus is not fully resolved here.
            # Let's return a star.
            star = Node()
            # Build a star from leaves_set
            # We need to balance it or just chain it?
            # A star in Newick is (a,b,c,d). In our binary Node structure, we can't have >2 children.
            # We must return a binary tree.
            # Let's just pick an arbitrary binary tree, e.g., balanced.
            # Or return None and let the caller handle it?
            # Let's return a balanced binary tree of the leaves.
            leaves_list = list(leaves_set)
            leaves_list.sort() # Deterministic
            def build_balanced(l_list):
                if len(l_list) == 1:
                    return Node(label=l_list[0])
                if len(l_list) == 2:
                    n = Node()
                    n.left = Node(label=l_list[0])
                    n.right = Node(label=l_list[1])
                    return n
                mid = len(l_list) // 2
                n = Node()
                n.left = build_balanced(l_list[:mid])
                n.right = build_balanced(l_list[mid:])
                return n
            return build_balanced(leaves_list)
            
        # Recurse
        left_leaves = found_split
        right_leaves = leaves_set - found_split
        
        left_node = build_subtree(left_leaves)
        right_node = build_subtree(right_leaves)
        
        new_node = Node()
        new_node.left = left_node
        new_node.right = right_node
        return new_node
        
    return build_subtree(all_leaves)

def _nni_refine(root, dists, labels, max_iterations=10):
    """
    Simple NNI refinement using NJ distance matrix as a guide?
    Or using Parsimony?
    Parsimony is better for topology.
    Let's use Parsimony score.
    """
    # Fitch Parsimony Score
    def fitch_score(tree, seqs, labels):
        # seqs: dict label -> string
        # tree: Node
        # Post-order traversal
        nodes = []
        def collect(node):
            if node.is_leaf():
                nodes.append(node)
            else:
                collect(node.left)
                collect(node.right)
                nodes.append(node)
        collect(tree)
        
        n_sites = len(seqs[labels[0]])
        total_cost = 0
        
        # Pre-compute leaf indices?
        # Just iterate.
        
        for site in range(n_sites):
            state_sets = {}
            costs = {}
            for node in nodes:
                if node.is_leaf():
                    char = seqs[node.label][site]
                    state_sets[node] = frozenset([char])
                    costs[node] = 0
                else:
                    l_set = state_sets[node.left]
                    r_set = state_sets[node.right]
                    inter = l_set & r_set
                    if inter:
                        state_sets[node] = inter
                        costs[node] = costs[node.left] + costs[node.right]
                    else:
                        state_sets[node] = l_set | r_set
                        costs[node] = costs[node.left] + costs[node.right] + 1
            total_cost += costs[tree]
            
        return total_cost

    # Get initial score
    # We need the sequences. 
    # Wait, _nni_refine needs sequences.
    # The signature in the main function will pass them.
    
    # For now, this function is a stub in this thought process, 
    # but in the final code, I will implement a basic NNI.
    # Actually, implementing NNI on the Node structure is tricky.
    # Let's skip NNI for now to ensure the consensus works, 
    # as consensus alone (inspiration 1) improved the score.
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
            
    # Compute full distance matrix
    dists = _compute_distance_matrix(sequences, labels)
    
    # Determine number of replicates based on time and size
    # Smaller trees: more replicates. Larger trees: fewer.
    # Time limit 260s.
    # NJ is O(N^3). N=32 -> 32768 ops. Very fast.
    # Consensus is O(R * N^2).
    # Let's aim for ~50-100 replicates for small N, fewer for large N.
    
    # Estimate cost of one NJ: ~ N^3 / 60 (very rough python overhead)
    # Let's just try to fit as many as possible within 10 seconds?
    # Or fixed number?
    # Inspiration 1 used majority vote.
    # Let's use a fixed number of replicates, say 50, but subsample the distance matrix?
    # No, subsampling the *data* (sequences) is better for bootstrap.
    # But we don't have the original sites easily accessible for resampling in this function signature?
    # We have `sequences`. We can resample sites.
    
    # Strategy:
    # 1. Compute full distance matrix (1 run).
    # 2. Run R-1 additional NJs on subsampled site distance matrices.
    #    How to subsample? 
    #    We can't easily get the site-by-site matrix without recomputing.
    #    Recomputing distance matrix is O(N^2 * L).
    #    L is up to 2000. N is up to 32.
    #    32^2 * 2000 = 2M ops. Fast.
    #    So we can compute new distance matrices for each replicate.
    
    # Let's do 20 replicates.
    # Total time: 20 * (Distance Calc + NJ).
    # Distance Calc: ~1ms. NJ: ~1ms. Total 40ms. Very safe.
    # We can do 100 replicates.
    
    n_replicates = 50
    
    trees = []
    
    # First tree: Full data
    root0 = _nj_build(dists, labels)
    trees.append(root0)
    
    # Remaining trees: Bootstrap on sites
    # We need to resample sites.
    # sequences is a dict of strings.
    # We can convert to a list of lists of chars, or just use indices.
    
    n_sites = len(sequences[labels[0]])
    
    for _ in range(n_replicates - 1):
        if _time_left() < 5.0:
            break
            
        # Resample sites with replacement
        # Using numpy for speed
        site_indices = np.random.choice(n_sites, size=n_sites, replace=True)
        
        # Compute distance matrix on resampled sites
        # We can optimize by pre-computing the char arrays?
        # seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        # seqs_arr = np.array(...)
        # But we already did that in _compute_distance_matrix.
        # Let's just call a helper that takes indices.
        
        # To avoid code duplication, let's just extract the resampled sequences.
        # This creates new strings, which is slow.
        # Better: Pass indices to a modified distance function.
        
        # Let's write a quick inline distance calc for resampled sites.
        seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        # Get array of sequences
        # This is O(N*L) to build.
        # We can pre-build the full array once.
        
        # Let's refactor:
        # 1. Build full int array once.
        # 2. For each replicate, pick columns.
        
        # Since we are in the loop, let's just do it efficiently.
        
        # Pre-compute full matrix of ints
        # Actually, _compute_distance_matrix does this every time.
        # Let's optimize the main flow:
        
        # 1. Build full int matrix `full_seqs` (N x L)
        # 2. Compute full dists
        # 3. For replicates:
        #    a. Pick columns `idx`
        #    b. Compute dists from `full_seqs[:, idx]`
        
        # Let's restructure `reconstruct` to do this.
        
        pass # Placeholder for logic restructuring

    # Restructuring `reconstruct` for efficiency:
    
    # 1. Convert to int matrix
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    full_seqs = np.array([[seq_map[c] for c in sequences[label]] for label in labels])
    N, L = full_seqs.shape
    
    # 2. Compute full dists
    dists_full = _compute_distance_matrix(sequences, labels) # Reusing existing func for simplicity, though it rebuilds array.
    # Actually, let's just use the array we built.
    # I will write a local distance calc to use the pre-built array.
    
    def _dists_from_array(arr):
        n = arr.shape[0]
        diff = (arr[:, np.newaxis, :] != arr[np.newaxis, :, :])
        num_diffs = np.sum(diff, axis=2)
        L_curr = arr.shape[1]
        p = num_diffs / L_curr
        p_clipped = np.clip(p, 0, 0.7499)
        d = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
        np.fill_diagonal(d, 0.0)
        d = np.nan_to_num(d, nan=3.0, posinf=3.0, neginf=0.0)
        return d
        
    dists_full = _dists_from_array(full_seqs)
    
    trees = [_nj_build(dists_full, labels)]
    
    # 3. Bootstrap replicates
    # Use random state for reproducibility? Not required, but good.
    # np.random.seed(42) # Optional
    
    for i in range(n_replicates - 1):
        if _time_left() < 5.0:
            break
            
        # Resample columns
        idx = np.random.choice(L, size=L, replace=True)
        sub_seqs = full_seqs[:, idx]
        
        # Compute dists
        d_rep = _dists_from_array(sub_seqs)
        
        # Build tree
        root_rep = _nj_build(d_rep, labels)
        trees.append(root_rep)
        
    # 4. Consensus
    # If we only have 1 tree, return it.
    if len(trees) == 1:
        root = trees[0]
    else:
        root = _consensus_tree(trees, labels)
        
    # 5. Convert to Newick
    def _to_newick(node):
        if node.is_leaf():
            return node.label
        return f"({_to_newick(node.left)},{_to_newick(node.right)})"
        
    newick = _to_newick(root)
    return newick + ";"

# EVOLVE-BLOCK-END
