import math
import time
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
'''
Improved Phylogenetic Reconstruction using Bootstrap Consensus (Median Tree)
over Neighbor Joining.
'''

def _jc_distance_vectorized(seq_matrix, idx_a, idx_b):
    """
    Compute Jukes-Cantor corrected distance between two sequences.
    seq_matrix: (n_sites, n_taxa) array of chars (0=A, 1=C, 2=G, 3=T)
    idx_a, idx_b: indices of taxa
    """
    # Count differences
    diffs = np.sum(seq_matrix[:, idx_a] != seq_matrix[:, idx_b])
    p = diffs / seq_matrix.shape[0]
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _nj_tree(dist_matrix, labels):
    """
    Neighbor Joining algorithm.
    dist_matrix: (n, n) numpy array
    labels: list of labels
    Returns: Newick string
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        # For 3 taxa, topology is always star-like in NJ, but we need a binary tree
        # Standard NJ for 3 nodes: join the two closest, then attach the third.
        # Actually, NJ works down to 2 nodes.
        pass

    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    # Copy dist matrix to mutable structure (we will modify it)
    # Using a dict or list of lists. List of lists is faster for numpy conversion if we stay in numpy.
    # But NJ updates specific entries. Let's use a 2D list or numpy array.
    # Numpy array modification is easy.
    d = dist_matrix.copy()
    
    while len(nodes) > 2:
        n_curr = len(nodes)
        # Compute r_i = sum_j d_ij
        # Only consider active nodes
        active_d = d[np.ix_(nodes, nodes)]
        r = np.sum(active_d, axis=1)
        
        # Compute Q matrix: Q_ij = (n-2)*d_ij - r_i - r_j
        # We need to find the min Q_ij for i != j
        # Q is symmetric.
        # Construct Q matrix for active nodes
        # d_active is (k, k)
        # r is (k,)
        # Q_ij = (k-2)*d_ij - r_i - r_j
        Q = (n_curr - 2) * active_d - r[:, None] - r[None, :]
        
        # Set diagonal to inf to avoid self-pairing
        np.fill_diagonal(Q, np.inf)
        
        # Find min
        min_val, min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i_idx = nodes[min_idx[0]]
        j_idx = nodes[min_idx[1]]
        
        # Join i and j
        new_node_label = f"({trees[i_idx]},{trees[j_idx]})"
        new_node_idx = -1 # placeholder
        
        # Calculate new distances
        # d_new_x = 0.5 * (d_i_x + d_j_x - d_i_j)
        d_ij = d[i_idx, j_idx]
        
        # Update d for new node with all other active nodes
        # The new node will be appended to the 'nodes' list logic, but we need to map it back to the original matrix indices for the next iteration?
        # Actually, it's easier to maintain a list of "current" nodes and a mapping to original indices or just keep a sparse representation.
        # Standard approach: 
        # 1. Identify i, j.
        # 2. Create new node k.
        # 3. Calculate d(k, x) for all x != i, j.
        # 4. Remove i, j from active set. Add k.
        # 5. Update distance matrix.
        
        # Since we are using a dense matrix `d` of size N x N (original N), we can just pick a new index?
        # No, it's better to maintain a list of active node IDs (original indices) and a separate "virtual" index for joined nodes?
        # Simpler: Use a dictionary for distances between current nodes.
        # But for speed with numpy, let's stick to arrays and compact the matrix.
        
        # Let's restart the logic with a list of active node IDs (which are just integers 0..n-1 initially, then we add new ones? No, that gets messy).
        # Alternative: Keep the full N x N matrix. When we join i and j, we create a new node. We need to store its distances to all other *current* nodes.
        # Let's use a list `active_nodes` which contains tuples of (label, tree_str, dist_to_others_dict)? No, too slow.
        
        # Let's use the standard array compaction method.
        # active_nodes: list of current node indices in the `d` matrix?
        # If we compact `d`, we lose the original indices.
        
        # Let's try a different representation:
        # `active_nodes` is a list of indices into a list `node_data`.
        # `node_data[i]` = { 'label': str, 'dist': dict {other_idx: val} }
        # This is O(n^2) memory and O(n^2) per step. For n=32, 32^2=1024 ops per step, 32 steps = 32k ops. Very fast.
        pass

    # The above pass logic is getting complicated to implement correctly in a scratch rewrite without bugs.
    # Let's use a simpler, robust NJ implementation using lists/dicts which is sufficient for n<=32.
    
    pass

def _nj_tree_simple(dist_matrix, labels):
    """
    Simple Neighbor Joining using dicts for distances.
    dist_matrix: (n, n) array
    labels: list of n labels
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
        
    # Initialize nodes
    # Each node is a dict: {'id': int, 'label': str, 'dist': {other_id: float}}
    nodes = []
    for i in range(n):
        nodes.append({
            'id': i,
            'label': labels[i],
            'dist': {j: dist_matrix[i, j] for j in range(n) if j != i}
        })
    
    while len(nodes) > 2:
        n_curr = len(nodes)
        # Compute r_i
        r = {node['id']: sum(node['dist'].values()) for node in nodes}
        
        # Find min Q
        best_q = float('inf')
        best_i = None
        best_j = None
        
        for i in range(n_curr):
            for j in range(i+1, n_curr):
                node_i = nodes[i]
                node_j = nodes[j]
                d_ij = node_i['dist'][node_j['id']]
                q = (n_curr - 2) * d_ij - r[node_i['id']] - r[node_j['id']]
                if q < best_q:
                    best_q = q
                    best_i = i
                    best_j = j
        
        node_i = nodes[best_i]
        node_j = nodes[best_j]
        
        # Create new node
        new_id = -1 # unique id not in 0..n-1? We can just use a counter.
        # Let's use a global counter or just 'new'
        new_label = f"({node_i['label']},{node_j['label']})"
        
        new_dist = {}
        # Calculate distances to all other nodes
        for k in range(n_curr):
            if k == best_i or k == best_j:
                continue
            node_k = nodes[k]
            # d_new_k = 0.5 * (d_i_k + d_j_k - d_i_j)
            d_i_k = node_i['dist'][node_k['id']]
            d_j_k = node_j['dist'][node_k['id']]
            d_i_j = node_i['dist'][node_j['id']]
            new_dist[node_k['id']] = 0.5 * (d_i_k + d_j_k - d_i_j)
            
        new_node = {
            'id': 'new_' + str(len(nodes)), # simple unique id
            'label': new_label,
            'dist': new_dist
        }
        
        # Remove i and j, add new_node
        # Order doesn't matter much
        nodes.pop(best_j) # remove j first (higher index) to keep i index valid?
        # If best_i > best_j, pop best_i first.
        if best_i > best_j:
            nodes.pop(best_i)
            nodes.pop(best_j)
        else:
            nodes.pop(best_j)
            nodes.pop(best_i)
            
        nodes.append(new_node)
        
    # Final 2 nodes
    # They are connected by an edge. The distance is the distance between them.
    # In NJ, the final step connects the last two.
    # The topology is (node1, node2).
    # We don't strictly need the branch length for the topology score, but the structure must be valid.
    return f"({nodes[0]['label']},{nodes[1]['label']})"

def _get_splits(newick, taxa):
    """
    Extract unrooted splits from a Newick string.
    Returns a set of frozensets.
    """
    # Simple parser for binary trees
    # Since we generate the trees, we know they are binary and valid.
    # We can use a stack-based parser.
    
    splits = set()
    stack = []
    
    # We need to map labels to indices or just use the labels themselves in the split.
    # Using labels is fine as long as we are consistent.
    
    def parse_tree(s):
        # Recursive parsing might be cleaner for small n
        # Find the root split
        depth = 0
        for i, c in enumerate(s):
            if c == '(':
                if depth == 0:
                    # Start of a subtree
                    pass
                depth += 1
            elif c == ')':
                depth -= 1
            elif c == ',' and depth == 1:
                # Split found at root level
                left = s[1:i]
                right = s[i+1:-1]
                # Get taxa for left and right
                taxa_left = _get_taxa(left)
                taxa_right = _get_taxa(right)
                # The split is (taxa_left, taxa_right)
                # Represent as frozenset(taxa_left)
                # To ensure canonical representation, sort the two sets and pick the one with the lexicographically smaller first element?
                # Or just store the frozenset of the smaller set?
                # Standard: A split is a bipartition. {A, B}. {B, A} is the same.
                # We can represent it by the frozenset of the subset that contains the lexicographically smallest taxon.
                all_taxa = set(taxa_left) | set(taxa_right)
                if min(taxa_left) < min(taxa_right):
                    splits.add(frozenset(taxa_left))
                else:
                    splits.add(frozenset(taxa_right))
                # Recurse
                parse_tree(left)
                parse_tree(right)
                return # Only parse the root level once? No, we need all internal nodes.
                
        # This recursive approach is tricky to implement correctly in a single pass without proper state.
        # Let's use a simpler method: 
        # Since n is small (<=32), we can just iterate over all pairs of taxa? No.
        # We can use a library? No.
        # Let's write a robust recursive splitter.
        pass

    # Better approach for small n:
    # We can reconstruct the tree structure as a set of splits directly during NJ?
    # Yes! During NJ, when we join i and j, we create a new node. The split is (subtree_i, subtree_j) relative to the rest?
    # Actually, the split created at each NJ step is the partition of the *current* active set into (subtree_i, subtree_j) vs the rest?
    # No, the split is the bipartition of the *original* leaf set induced by the internal edge.
    
    # It is safer to parse the Newick string.
    # Here is a reliable parser for unrooted binary trees.
    
    def extract_splits(newick):
        splits = set()
        # Remove trailing semicolon
        if newick.endswith(';'):
            newick = newick[:-1]
        
        # Recursive function
        def parse(s):
            # s is a tree string like "(A,B),(C,D)" or "A"
            # We need to find the root split.
            # The root is the first ',' that is at depth 1 (inside the outermost parens if any).
            # Actually, the input to parse should be a valid tree fragment.
            
            depth = 0
            for i, c in enumerate(s):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                elif c == ',' and depth == 1:
                    # Found the root split
                    left_str = s[:i]
                    right_str = s[i+1:]
                    
                    # Get taxa in left and right
                    taxa_left = set()
                    taxa_right = set()
                    
                    # Helper to get all leaf labels from a string
                    def get_leaves(tstr):
                        leaves = set()
                        # Simple regex or scan
                        # Labels are t0, t1, etc.
                        # We can just scan for 't' followed by digits
                        import re
                        matches = re.findall(r't\d+', tstr)
                        return set(matches)
                        
                    taxa_left = get_leaves(left_str)
                    taxa_right = get_leaves(right_str)
                    
                    # Canonicalize split
                    # The split is the set of taxa on one side.
                    # To make it unique, we choose the side that contains the lexicographically smallest taxon in the union.
                    all_taxa = taxa_left | taxa_right
                    min_taxon = min(all_taxa)
                    if min_taxon in taxa_left:
                        splits.add(frozenset(taxa_left))
                    else:
                        splits.add(frozenset(taxa_right))
                    
                    # Recurse
                    parse(left_str)
                    parse(right_str)
                    return
            
            # If no split found, it's a leaf or a single node (shouldn't happen for internal calls)
            # If it's a leaf, no splits.
            return

        parse(newick)
        return splits

    return extract_splits(newick)

def _rf_distance(splits1, splits2):
    """
    Calculate Robinson-Foulds distance between two sets of splits.
    """
    # The number of incompatible splits
    # Splits are frozensets.
    # Two splits {A, B} and {C, D} are compatible if one of the intersections A& C, A & D, B & C, B & D is empty.
    # It's easier to just count the symmetric difference if we have a canonical representation?
    # No, RF distance is the number of splits in S1 not in S2 plus number in S2 not in S1.
    # BUT, only if the splits are defined on the same set of taxa.
    # The splits must be compatible to be "the same".
    # Actually, the standard RF distance is:
    # |S1 \ S2| + |S2 \ S1|
    # Where S1 and S2 are sets of *bipartitions*.
    # Since we canonicalize them (always store the side with the min taxon), two compatible splits will have the SAME frozenset representation.
    # Incompatible splits will have DIFFERENT frozenset representations.
    # So we can just do set difference!
    
    return len(splits1 - splits2) + len(splits2 - splits1)

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 260.0 # 10s safety margin
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    # Prepare sequence matrix
    # Map chars to indices for faster comparison?
    # Or just use strings.
    # For bootstrap, we need to resample columns.
    # Let's convert to a numpy array of ints for speed.
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    n_sites = len(sequences[labels[0]])
    
    # Create matrix (n_sites, n_taxa)
    seq_matrix = np.zeros((n_sites, n_taxa), dtype=np.uint8)
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        for j, c in enumerate(seq):
            seq_matrix[j, i] = char_map[c]
    
    def compute_dist_matrix(matrix):
        """
        Compute pairwise JC distances.
        matrix: (m, n) array
        Returns: (n, n) array
        """
        n = matrix.shape[1]
        d = np.zeros((n, n))
        # Vectorized distance calculation
        # For each pair (i, j), count mismatches
        # This is O(n^2 * m). For n=32, m=2000, this is 32*31/2 * 2000 ~ 1M ops. Fast.
        for i in range(n):
            for j in range(i+1, n):
                diffs = np.sum(matrix[:, i] != matrix[:, j])
                p = diffs / matrix.shape[0]
                if p >= 0.7499:
                    dist = 3.0
                else:
                    dist = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
                d[i, j] = dist
                d[j, i] = dist
        return d

    # 1. Compute original tree
    dist_orig = compute_dist_matrix(seq_matrix)
    tree_orig = _nj_tree_simple(dist_orig, labels)
    splits_orig = _get_splits(tree_orig, labels)
    
    # 2. Bootstrap
    # Determine number of bootstraps
    # If n_taxa is small, we can do more.
    # Time per bootstrap:
    # - Resample: O(m)
    # - Dist Matrix: O(n^2 * m)
    # - NJ: O(n^3)
    # - Splits: O(n^2)
    
    # Let's estimate time for one bootstrap.
    # n=32, m=2000.
    # Dist matrix: 32*31/2 * 2000 ~ 1M comparisons. ~0.01s?
    # NJ: 32^3 ~ 32k ops. Negligible.
    # So ~0.01s per bootstrap.
    # We have 260s.
    # 20 instances. 13s per instance.
    # 13s / 0.01s = 1300 bootstraps.
    # This is plenty. Let's cap at 200 to be safe and allow for variance.
    
    n_bootstraps = 200
    if n_taxa > 20:
        n_bootstraps = 100
    if n_taxa > 30:
        n_bootstraps = 50
        
    bootstrap_trees = []
    bootstrap_splits = []
    
    # Add original tree to the pool?
    # Usually, bootstrap consensus uses only bootstrap trees.
    # But adding the original can help if bootstraps are noisy?
    # The "median" of the bootstrap set is the standard.
    
    for b in range(n_bootstraps):
        if time.time() > deadline - 5.0:
            break
            
        # Resample columns
        # Use numpy random
        idx = np.random.choice(n_sites, n_sites, replace=True)
        boot_matrix = seq_matrix[idx, :]
        
        dist_boot = compute_dist_matrix(boot_matrix)
        tree_boot = _nj_tree_simple(dist_boot, labels)
        splits_boot = _get_splits(tree_boot, labels)
        
        bootstrap_trees.append(tree_boot)
        bootstrap_splits.append(splits_boot)
        
    if not bootstrap_trees:
        return tree_orig
        
    # 3. Find Median Tree
    # Calculate RF distance from each tree to all others
    # n_trees = len(bootstrap_trees)
    # For each i, sum_j rf(i, j)
    # Pick min sum.
    
    n_trees = len(bootstrap_trees)
    if n_trees == 1:
        return bootstrap_trees[0]
        
    min_score = float('inf')
    best_tree_idx = 0
    
    for i in range(n_trees):
        score = 0
        for j in range(n_trees):
            if i == j:
                continue
            score += _rf_distance(bootstrap_splits[i], bootstrap_splits[j])
        if score < min_score:
            min_score = score
            best_tree_idx = i
            
    return bootstrap_trees[best_tree_idx]

# EVOLVE-BLOCK-END
