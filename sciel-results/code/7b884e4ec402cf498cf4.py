"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Maximum Likelihood tree reconstruction using NNI heuristic search.
   Uses Felsenstein's pruning algorithm with Jukes-Cantor model.
   Initializes with a random binary tree and optimizes topology via NNI moves.'''
import math
import random
import numpy as np
import time

# Precompute JC transition probabilities for a grid of branch lengths
# We will compute these on the fly or cache them if needed.
# Given the small number of taxa, on-the-fly is fine if optimized.

def _jc_probs(t):
    """
    Compute 4x4 transition probability matrix for Jukes-Cantor model.
    t: branch length
    Returns: 4x4 numpy array P where P[i,j] = P(base j | base i, t)
    """
    if t < 0:
        t = 0.0
    exp_term = math.exp(-4.0 * t / 3.0)
    p_match = 0.25 + 0.75 * exp_term
    p_mismatch = (1.0 - p_match) / 3.0
    # Matrix is symmetric, diagonal p_match, off-diagonal p_mismatch
    p = np.full((4, 4), p_mismatch)
    np.fill_diagonal(p, p_match)
    return p

def _calc_likelihood(tree, sequences, labels, root_idx):
    """
    Calculate log-likelihood of the tree given the sequences.
    tree: list of nodes. Each node is a dict or class.
    Let's use a simple class for tree nodes to manage structure easily during NNI.
    """
    pass

class TreeNode:
    __slots__ = ['children', 'parent', 'length', 'id']
    def __init__(self, children=None, parent=None, length=0.0, id=0):
        self.children = children if children else []
        self.parent = parent
        self.length = length
        self.id = id

def _build_random_tree(n):
    """Build a random binary tree with n leaves."""
    if n == 1:
        return TreeNode(children=[], parent=None, length=0.0, id=0)
    
    # Start with n leaves
    nodes = [TreeNode(children=[], parent=None, length=0.0, id=i) for i in range(n)]
    
    # Randomly merge nodes
    # We need to maintain a list of available nodes (roots of subtrees)
    available = list(range(n))
    next_id = n
    
    while len(available) > 1:
        # Pick two random distinct nodes
        i = random.randint(0, len(available) - 1)
        j = random.randint(0, len(available) - 1)
        while j == i:
            j = random.randint(0, len(available) - 1)
        
        node_i = nodes[available[i]]
        node_j = nodes[available[j]]
        
        # Create new internal node
        new_node = TreeNode(children=[node_i, node_j], parent=None, length=0.0, id=next_id)
        nodes.append(new_node)
        
        # Update parents
        node_i.parent = new_node
        node_j.parent = new_node
        
        # Update available list
        # Remove i and j, add new_node
        # Note: indices in 'available' refer to indices in 'nodes'
        # It's easier to maintain a list of node objects
        pass # This implementation is getting messy with indices. Let's use objects directly.

def _build_random_tree_v2(n):
    if n == 1:
        return TreeNode(children=[], parent=None, length=0.0, id=0)
    
    # List of current roots
    roots = [TreeNode(children=[], parent=None, length=0.0, id=i) for i in range(n)]
    next_id = n
    
    while len(roots) > 1:
        # Pick two distinct roots
        i = random.randint(0, len(roots) - 1)
        j = random.randint(0, len(roots) - 1)
        while j == i:
            j = random.randint(0, len(roots) - 1)
        
        node_i = roots[i]
        node_j = roots[j]
        
        new_node = TreeNode(children=[node_i, node_j], parent=None, length=0.0, id=next_id)
        node_i.parent = new_node
        node_j.parent = new_node
        
        # Replace one with new node, remove the other
        roots[i] = new_node
        roots.pop(j)
        if i > j:
             # If we popped j which was before i, index i shifted?
             # No, we popped j. If j < i, i is now i-1.
             # But we assigned roots[i] = new_node BEFORE pop?
             # Let's do it carefully.
             pass
        
        # Correct approach:
        # Remove j
        roots.pop(j)
        # If j < i, i decreases by 1
        if j < i:
            i -= 1
        roots[i] = new_node
        
        next_id += 1
        
    return roots[0]

# The above random tree builder has index bugs. Let's use a simpler approach:
# Just shuffle leaves and build a balanced tree or use a library? No library.
# Let's use a standard method:
def _random_binary_tree(n):
    if n == 1:
        return TreeNode(id=0)
    leaves = [TreeNode(id=i) for i in range(n)]
    next_id = n
    # Use a stack or list
    stack = leaves[:]
    random.shuffle(stack)
    # Actually, random merging is fine if implemented correctly.
    # Let's use a different strategy: 
    # 1. Create n leaves.
    # 2. While len > 1: pick 2 random, merge into new internal node, replace them in list.
    nodes = [TreeNode(id=i) for i in range(n)]
    next_id = n
    while len(nodes) > 1:
        i = random.randint(0, len(nodes) - 1)
        j = random.randint(0, len(nodes) - 1)
        while i == j:
            j = random.randint(0, len(nodes) - 1)
        n1 = nodes[i]
        n2 = nodes[j]
        new_node = TreeNode(children=[n1, n2], id=next_id)
        n1.parent = new_node
        n2.parent = new_node
        next_id += 1
        # Remove i and j, add new_node
        # To avoid index issues, remove the higher index first
        if i > j:
            nodes.pop(i)
            nodes.pop(j)
        else:
            nodes.pop(j)
            nodes.pop(i)
        nodes.append(new_node)
    return nodes[0]

def _get_postorder(root):
    """Return nodes in postorder traversal."""
    stack = [(root, False)]
    order = []
    while stack:
        node, processed = stack.pop()
        if processed:
            order.append(node)
        else:
            stack.append((node, True))
            for child in reversed(node.children):
                stack.append((child, False))
    return order

def _calc_log_likelihood(root, sequences, labels):
    """
    Calculate log-likelihood using Felsenstein's algorithm.
    sequences: dict {label: str}
    labels: list of labels in order t0..tn-1
    """
    # Map label to index for fast lookup
    # We assume labels are t0, t1...
    # Let's map leaf node id to sequence index
    # In our tree construction, leaf nodes have id 0..n-1 corresponding to labels[0]..labels[n-1]
    # Wait, the random tree builder assigns ids 0..n-1 to leaves.
    # So leaf with id k corresponds to labels[k].
    
    # Precompute sequences as arrays of ints (0=A, 1=C, 2=G, 3=T)
    # This is done outside to avoid repeated conversion.
    # Let's pass precomputed sequences.
    
    # We need to pass the precomputed sequences and a mapping from node id to seq index.
    # For leaves, node.id is the index.
    
    # Postorder traversal
    postorder = _get_postorder(root)
    
    # Likelihoods: list of arrays, one per node in postorder
    # L[node] = array of 4 floats (log-likelihoods for A,C,G,T)
    # We use log-space to avoid underflow.
    # However, JC probabilities are positive.
    # log(P) is fine.
    
    # Initialize log-likelihoods for leaves
    # We need to know which nodes are leaves.
    # A node is a leaf if it has no children.
    
    # Let's store log-likelihoods in a dictionary or list indexed by node id?
    # Node ids are unique.
    log_lh = {}
    
    for node in postorder:
        if not node.children:
            # Leaf
            # Get sequence for this leaf
            # node.id is the index into labels
            seq = sequences[labels[node.id]]
            # Convert to int array
            # We should precompute this. Let's assume `sequences_int` is passed.
            pass
            
    # This function signature needs to be adjusted to accept precomputed data.
    pass

# Let's refine the structure to be efficient.

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n < 2:
        return labels[0] + ";" if n == 1 else ";"
    
    # Precompute sequences as int arrays
    # A=0, C=1, G=2, T=3
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs_int = []
    for lab in labels:
        s = sequences[lab]
        seqs_int.append(np.array([base_map[c] for c in s], dtype=np.int8))
    
    # Align lengths
    L = len(sequences[labels[0]])
    
    # Initialize tree
    root = _random_binary_tree(n)
    
    # Optimize branch lengths? 
    # The task asks for topology recovery. Branch lengths are optional.
    # However, likelihood depends on branch lengths.
    # If we fix branch lengths to 0, likelihood is poor.
    # We should optimize branch lengths for each topology.
    # This is computationally expensive.
    # Alternative: Use a fixed "average" branch length or optimize locally.
    # Given the time limit, maybe we can skip branch length optimization and just optimize topology?
    # But NJ uses distances which are branch-length informed.
    # ML without branch length optimization is weak.
    # Let's try to optimize branch lengths using a simple 1D search (golden section) for each edge during NNI.
    # Or, simpler: assume all branch lengths are equal to some value estimated from data?
    # Or, use the fact that for JC, the MLE of branch length can be approximated.
    
    # Let's try a simpler ML approach:
    # 1. Estimate a global branch length scale from pairwise distances.
    # 2. Use that as initial guess.
    # 3. During NNI, keep branch lengths fixed? No, that's bad.
    # 4. Optimize branch lengths for the initial tree.
    # 5. During NNI, re-optimize branch lengths for the new topology.
    
    # This is getting complex.
    # Maybe a simpler approach: **Minimum Evolution (ME)** or **BIONJ** is distance-based.
    # The prompt asks for a NEW approach. ML is distinct.
    # But implementing full ML with branch length optimization in Python within 9s/instance for 32 taxa is risky.
    
    # Alternative New Approach: **Maximum Compatibility** or **Clustering**?
    # Clustering (UPGMA) is distance-based.
    # What about **Phylogenetic Inference using Information Theory**?
    
    # Let's stick to ML but simplify:
    # Use a **fixed** branch length for all edges?
    # If we fix branch lengths, we can compare topologies.
    # How to choose the fixed length?
    # Estimate average pairwise distance, divide by ~n?
    # Let's try this:
    # 1. Compute pairwise distances (JC).
    # 2. Estimate average branch length `t_avg`.
    # 3. Build random tree.
    # 4. NNI search maximizing likelihood with all branch lengths = t_avg.
    # This is a heuristic, but it's a new approach (ML-like scoring with fixed params).
    
    # Let's implement this.
    
    # Step 1: Distances
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            s1 = sequences[labels[i]]
            s2 = sequences[labels[j]]
            mism = sum(1 for a,b in zip(s1,s2) if a!=b)
            p = mism / L
            if p < 0.75:
                d = -0.75 * math.log(1 - 4*p/3)
            else:
                d = 3.0
            dist[i,j] = d
            dist[j,i] = d
            
    # Average distance
    avg_d = np.mean(dist[dist > 0])
    # Heuristic for average branch length: 
    # In a random tree, average distance is roughly 2 * avg_branch * (some factor).
    # Let's just use avg_d / 2 as a guess for edge length?
    t_avg = avg_d / 2.0
    if t_avg <= 0:
        t_avg = 0.1

    # Step 2: Random Tree
    root = _random_binary_tree(n)
    
    # Step 3: NNI Search
    # We need a fast likelihood calculator for a tree with fixed branch lengths.
    
    def _get_leaves_in_order(root):
        """Get leaf node ids in postorder."""
        leaves = []
        stack = [root]
        while stack:
            node = stack.pop()
            if not node.children:
                leaves.append(node.id)
            else:
                # Push children in reverse to maintain order?
                # Order doesn't matter for leaves list, just need to map id->seq
                stack.extend(node.children)
        return leaves

    # Precompute sequence data
    # seqs_int is a list of arrays.
    
    def _calc_ll_fixed(root, t_avg):
        """
        Calculate log-likelihood with all branch lengths = t_avg.
        """
        # Postorder traversal
        postorder = _get_postorder(root)
        
        # log_lh[node_id] = array of 4 log-probs
        # We use log-sum-exp for stability.
        
        # Precompute JC transition matrix for t_avg
        p_mat = _jc_probs(t_avg)
        # Log of p_mat
        # Handle zeros: log(0) is -inf.
        # p_mismatch is > 0 for t>0.
        log_p_mat = np.log(p_mat + 1e-10)
        
        log_lh = {}
        
        for node in postorder:
            if not node.children:
                # Leaf
                # Get sequence
                # node.id is the index
                seq = seqs_int[node.id]
                # Log-likelihood for each state
                # If state matches seq, log(1) = 0? No, likelihood is 1 for correct base, 0 for others.
                # In log space: 0 for correct, -inf for others.
                ll = np.full(4, -np.inf)
                # For each site, we sum log probabilities.
                # Wait, Felsenstein's algorithm multiplies probabilities across sites.
                # So for a leaf, L(a) = product over sites of P(site_base | a).
                # If a == site_base, P=1 (log=0). Else P=0 (log=-inf).
                # So L(a) is 1 if a matches ALL sites? No.
                # L(a) = P(Data | a).
                # If the sequence has mixed bases, P(Data|a) is 0 for any a?
                # No, the model is P(Data | Tree, Params).
                # The leaf likelihood is P(Data_leaf | state_at_leaf).
                # If state_at_leaf is A, and data is A at all sites, P=1.
                # If data is C at some site, P=0.
                # So yes, if there is any mismatch, the likelihood is 0.
                # This means we MUST have branch lengths > 0 to allow for changes.
                # But at the leaf, the state is the sequence itself?
                # No, the state at the leaf is the observed sequence.
                # The likelihood of the observed sequence given the state at the leaf is 1 if they match, 0 if not.
                # Since they always match (it's the leaf), the leaf likelihood is 1 for the observed state, 0 for others.
                # BUT, we are summing over internal states.
                # The leaf contributes a "delta" function.
                # So for a leaf with sequence S, L_leaf(A) = 1 if S is all A? No.
                # L_leaf(A) = P(S | A) = 1 if S==A else 0.
                # This is only true if there are no mutations at the leaf.
                # In Felsenstein's algorithm, the leaf likelihood is indeed a delta function for the observed base.
                # So, L_leaf(A) = 1 if the observed base is A, else 0.
                # Wait, the sequence has length L.
                # The likelihood is the product over sites.
                # For site k, observed base is b_k.
                # P(b_k | A) = 1 if b_k==A else 0.
                # So P(Data | A) = 1 if all b_k==A else 0.
                # This implies that if the sequence is not constant, the likelihood is 0 for all states?
                # That can't be right.
                # Ah, the state at the leaf is NOT the sequence. The state is the ancestral base at the leaf node.
                # The observed data is the sequence.
                # The probability of observing sequence S given state A at the leaf is:
                # Product over sites of P(site_base | A).
                # If the model is JC, P(site_base | A) is the transition probability from A to site_base over time 0?
                # No, the branch length from the leaf to the internal node is t.
                # The leaf node itself doesn't have a branch length to the data.
                # The data IS the leaf state.
                # So, in Felsenstein's algorithm, the leaf likelihood is:
                # L_leaf(a) = 1 if a == observed_base, else 0.
                # This is for a single site.
                # For multiple sites, we sum log-likelihoods.
                # So, for a leaf, L(a) = 0 if a matches the base at ALL sites? No.
                # L(a) = sum over sites of log(P(base_k | a)).
                # If a=A, and base_k=C, log(0) = -inf.
                # So if any site is C, L(A) = -inf.
                # This means the likelihood is 0 unless the sequence is constant?
                # This is incorrect for standard phylogenetic inference.
                # The error is: The leaf likelihood is NOT P(Data | state).
                # The leaf likelihood is P(Data | state) where the "state" is the base at the leaf tip.
                # But the base at the leaf tip IS the observed data.
                # So P(Data | Data) = 1.
                # So L_leaf(observed_base) = 1, L_leaf(other) = 0.
                # This is correct for a single site.
                # For multiple sites, we treat each site independently.
                # So we run Felsenstein's algorithm for EACH site separately and sum the log-likelihoods.
                
                # Optimization:
                # Instead of looping over sites in Python, we can vectorize.
                # But the tree structure is the same for all sites.
                # We can compute the likelihood for all sites simultaneously using NumPy arrays of shape (L, 4).
                
                pass
            else:
                # Internal node
                pass

    # This vectorization is key.
    # Let's implement the vectorized Felsenstein.
    
    def _calc_ll_vectorized(root, t_avg):
        postorder = _get_postorder(root)
        
        # Log transition matrix
        p_mat = _jc_probs(t_avg)
        log_p_mat = np.log(p_mat + 1e-10) # (4,4)
        
        # Initialize likelihoods
        # We need a storage for log-likelihoods for each node.
        # Shape: (num_nodes, L, 4) ? No, (num_nodes, 4) for each site?
        # Better: (L, num_nodes, 4)
        # Or process site by site? L up to 2000. num_nodes ~ 2*n.
        # 2000 * 64 * 4 operations. Fast enough.
        
        # Let's store log_lh as a list of arrays (L, 4) for each node.
        # Map node id to index in postorder list?
        # Or just use a dict.
        
        log_lh = {}
        
        for node in postorder:
            if not node.children:
                # Leaf
                # Get sequence for this leaf
                # node.id is index
                seq = seqs_int[node.id] # (L,)
                # Create (L, 4) array of -inf
                ll = np.full((L, 4), -np.inf)
                # For each site, set the observed base's log-prob to 0 (log 1)
                # ll[site, base] = 0
                # We can do this with advanced indexing
                # ll[np.arange(L), seq] = 0
                ll[np.arange(L), seq] = 0.0
                log_lh[node.id] = ll
            else:
                # Internal
                c1, c2 = node.children
                ll1 = log_lh[c1.id] # (L, 4)
                ll2 = log_lh[c2.id] # (L, 4)
                
                # Combine children
                # L_parent(a) = sum_b P(a->b|t) * L_child1(b) * L_child2(b)
                # In log space:
                # log L_parent(a) = logsumexp_b ( log P(a->b|t) + log L_child1(b) + log L_child2(b) )
                
                # log_p_mat is (4,4). log_p_mat[a,b] = log P(a->b)
                # We need to compute for each a (0..3) and each site.
                
                # Sum log likelihoods of children: (L, 4)
                log_sum_children = ll1 + ll2
                
                # Now for each a, compute logsumexp over b of (log_p_mat[a,b] + log_sum_children[:,b])
                # log_p_mat[a, :] is (4,)
                # log_sum_children is (L, 4)
                
                # Result shape (L, 4)
                ll_parent = np.zeros((L, 4))
                for a in range(4):
                    # term = log_p_mat[a, :] + log_sum_children # (L, 4)
                    # logsumexp over axis 1
                    term = log_p_mat[a, :] + log_sum_children
                    # logsumexp
                    max_term = np.max(term, axis=1)
                    # Handle -inf
                    # If all are -inf, max is -inf.
                    # logsumexp = max + log(sum(exp(term - max)))
                    # If max is -inf, result is -inf.
                    if np.all(np.isneginf(max_term)):
                         ll_parent[:, a] = -np.inf
                    else:
                         # Avoid overflow/underflow
                         exp_term = np.exp(term - max_term[:, None])
                         sum_exp = np.sum(exp_term, axis=1)
                         # log(sum_exp)
                         # If sum_exp is 0, log is -inf
                         log_sum = np.log(sum_exp)
                         ll_parent[:, a] = max_term + log_sum
                         
                log_lh[node.id] = ll_parent
                
        # Root likelihood
        # Sum over root states (uniform prior 1/4)
        # log L = logsumexp_a ( log(1/4) + log L_root(a) )
        # log(1/4) = -log(4)
        ll_root = log_lh[root.id] # (L, 4)
        
        # Sum over sites
        # log L_total = sum_sites log L_site
        # log L_site = logsumexp_a ( -log(4) + ll_root[site, a] )
        
        # Add prior
        ll_root_prior = ll_root - math.log(4)
        
        # Logsumexp over axis 1 (states)
        max_term = np.max(ll_root_prior, axis=1)
        exp_term = np.exp(ll_root_prior - max_term[:, None])
        sum_exp = np.sum(exp_term, axis=1)
        log_l_site = max_term + np.log(sum_exp)
        
        # Sum over sites
        total_log_l = np.sum(log_l_site)
        
        return total_log_l

    # NNI Search
    current_ll = _calc_ll_vectorized(root, t_avg)
    
    # Find internal edges
    # An edge is defined by (parent, child) where child is internal?
    # NNI applies to internal edges.
    # In a binary tree, an internal edge connects two internal nodes.
    # Let's list all internal edges.
    
    def _get_internal_edges(root):
        edges = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node.parent and not node.children:
                continue
            if node.children:
                # Check if this node is an internal node connected to another internal node
                # The edge is (node.parent, node)
                if node.parent and node.parent.children:
                    edges.append((node.parent, node))
                for child in node.children:
                    stack.append(child)
        return edges

    # NNI Move
    # For an edge (A, B) where B is child of A.
    # Let C be the other child of A.
    # Let D, E be children of B.
    # Original: (C, (D, E)) under A.
    # NNI 1: (D, (C, E))
    # NNI 2: (E, (C, D))
    
    # We need to modify the tree structure, calculate LL, and revert.
    
    # To make this efficient, we can't just call _calc_ll_vectorized (which is O(L*N)) for every NNI.
    # N ~ 32, L ~ 1000. O(32000) per LL.
    # Number of NNI moves ~ 2 * (N-3) ~ 60.
    # 60 * 32000 = 2 million ops per iteration.
    # If we do 100 iterations, 200 million ops.
    # In Python/NumPy, this might be slow.
    
    # Optimization: Only recalculate likelihood for the affected part of the tree?
    # Yes, Felsenstein's algorithm is local.
    # But implementing that is complex.
    # Let's try the full recalculation first and see if it fits in time.
    # 9 seconds per instance.
    # 200 million NumPy ops might take ~2-5 seconds.
    # So ~1-2 iterations per second?
    # We need to converge.
    
    # Let's implement the NNI loop.
    
    max_iter = 50
    for _ in range(max_iter):
        improved = False
        edges = _get_internal_edges(root)
        random.shuffle(edges)
        
        for parent, child in edges:
            # Get other child of parent
            c1, c2 = parent.children
            if c1 == child:
                other = c2
            else:
                other = c1
                
            # Get children of child
            d1, d2 = child.children
            
            # Try NNI 1: (other, d1) and (d2, ?)
            # Original: parent -> (other, child)
            # child -> (d1, d2)
            
            # NNI 1:
            # New structure:
            # parent -> (other, new_node1)
            # new_node1 -> (d1, new_node2)
            # new_node2 -> (d2, other) ?? No.
            
            # Standard NNI:
            # Topology: (A, (B, C))
            # NNI 1: (B, (A, C))
            # NNI 2: (C, (A, B))
            
            # Here A=other, B=d1, C=d2.
            # Node 'child' is the parent of B and C.
            # Node 'parent' is the parent of A and 'child'.
            
            # NNI 1:
            # Remove child.
            # Connect A to d1? No.
            # New node X1.
            # X1 children: d1, A.
            # New node X2.
            # X2 children: X1, d2.
            # parent children: X2, ...?
            
            # Let's trace:
            # Original:
            #   P
            #  / \
            # A   C_node
            #    / \
            #   B   D
            
            # NNI 1:
            #   P
            #  / \
            # X1  D
            # / \
            # A  B
            # Wait, this changes the degree.
            
            # Correct NNI 1:
            #   P
            #  / \
            # X1  D
            # / \
            # A  B
            # This is not binary if P has 2 children and X1 has 2.
            # But P originally had A and C_node.
            # Now P has X1 and D.
            # X1 has A and B.
            # This works.
            
            # NNI 2:
            #   P
            #  / \
            # X2  B
            # / \
            # A  D
            
            # Let's implement NNI 1.
            
            # Save original state
            orig_parent_children = parent.children[:]
            orig_child_children = child.children[:]
            orig_other_parent = other.parent
            orig_d1_parent = d1.parent
            orig_d2_parent = d2.parent
            
            # Apply NNI 1
            # New node X1
            x1 = TreeNode(children=[other, d1], id=1000)
            other.parent = x1
            d1.parent = x1
            
            # New node X2
            x2 = TreeNode(children=[x1, d2], id=1001)
            x1.parent = x2
            d2.parent = x2
            
            # Update parent
            parent.children = [x2, d2] # Wait, d2 is child of x2.
            # Original parent children: [other, child]
            # New parent children: [x2, d2]? No.
            # In NNI 1, D moves to be sibling of X1?
            # Let's re-verify NNI definition.
            
            # Edge (P, C).
            # Subtrees: A (sibling of C), B, D (children of C).
            # NNI 1: (B, (A, D)) ?
            # Reference: 
            # (A, (B, C)) -> (B, (A, C))
            # Here A=other, B=d1, C=d2.
            # Original: (other, (d1, d2))
            # NNI 1: (d1, (other, d2))
            
            # So:
            # New node X1 children: [other, d2]
            # New node X2 children: [d1, X1]
            # Parent children: [X2, ...]
            # Who is the other child of Parent?
            # In the original, Parent had [other, child].
            # In the new tree, Parent should have [X2, d1]? No.
            
            # Let's draw:
            #   P
            #  / \
            # O   C
            #    / \
            #   D1  D2
            
            # NNI 1:
            #   P
            #  / \
            # X   D1
            # / \
            # O  D2
            
            # So X children: [O, D2]
            # P children: [X, D1]
            
            # Let's implement this.
            
            x1 = TreeNode(children=[other, d2], id=1000)
            other.parent = x1
            d2.parent = x1
            
            x2 = x1 # No, X is the new node.
            # P children: [x1, d1]
            d1.parent = parent
            parent.children = [x1, d1]
            
            # Calculate LL
            new_ll = _calc_ll_vectorized(root, t_avg)
            
            # Revert
            parent.children = orig_parent_children
            child.children = orig_child_children
            other.parent = orig_other_parent
            d1.parent = orig_d1_parent
            d2.parent = orig_d2_parent
            
            if new_ll > current_ll:
                # Accept
                # Re-apply
                x1 = TreeNode(children=[other, d2], id=1000)
                other.parent = x1
                d2.parent = x1
                d1.parent = parent
                parent.children = [x1, d1]
                
                # Remove old child node from tree?
                # It's still in memory, but not connected.
                # We don't need to delete it.
                
                current_ll = new_ll
                improved = True
                break # Restart edge iteration? Or continue?
                # Usually, accept and continue to next edge or restart.
                # Let's restart to be safe.
        
        if not improved:
            break
            
    # Convert tree to Newick
    def _to_newick(node):
        if not node.children:
            return f"t{node.id}"
        child_strs = [_to_newick(c) for c in node.children]
        return f"({','.join(child_strs)})"
        
    newick = _to_newick(root) + ";"
    return newick

# EVOLVE-BLOCK-END
