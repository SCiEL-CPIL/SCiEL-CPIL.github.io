import math
import time
import numpy as np

def _jc_distance_matrix(sequences, labels):
    """Compute JC-corrected distance matrix from sequences using vectorized operations."""
    n = len(labels)
    if n == 0:
        return np.zeros((0, 0))
    seq_len = len(next(iter(sequences.values())))
    
    char_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_arrays = np.zeros((n, seq_len), dtype=np.int8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j, c in enumerate(seq):
            seq_arrays[i, j] = char_to_int[c]
    
    # Compute pairwise differences vectorized
    # diff[i,j] = number of positions where seq i != seq j
    # Use broadcasting: seq_arrays[:, None, :] != seq_arrays[None, :, :]
    # This gives (n, n, seq_len) boolean array
    diff = np.sum(seq_arrays[:, None, :] != seq_arrays[None, :, :], axis=2)
    
    p = diff.astype(np.float64) / seq_len
    np.fill_diagonal(p, 0.0)
    
    # JC correction: d = -0.75 * ln(1 - 4p/3)
    # Handle p >= 0.75 (log of non-positive)
    p_safe = np.minimum(p, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p_safe / 3.0)
    np.fill_diagonal(d, 0.0)
    
    # Ensure symmetry
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    
    return d

def _nj_tree(dist_mat, n):
    """
    Build a binary tree using Neighbor Joining.
    Returns a Newick string.
    """
    if n <= 1:
        return "t0;" if n == 1 else ""
    
    if n == 2:
        return f"(t0,t1);"
    
    # active: list of current node indices
    # dist: distance matrix (will grow)
    # labels: maps node index to label string
    # parent: for tracking tree structure
    # children: for each node, list of (child, branch_len)
    
    active = list(range(n))
    labels_map = {i: f"t{i}" for i in range(n)}
    next_id = n
    
    # dist will be a dict of dicts or a growing matrix
    # For efficiency with up to 32 taxa, use a 2D array of size (2n, 2n)
    max_nodes = 2 * n
    dist = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            dist[i][j] = dist_mat[i][j]
    
    # children[node] = list of (child_node, branch_length)
    children = {i: [] for i in range(n)}
    
    while len(active) > 2:
        k = len(active)
        
        # Compute row sums
        r = np.zeros(k, dtype=np.float64)
        for i in range(k):
            node_i = active[i]
            for j in range(k):
                if i != j:
                    r[i] += dist[node_i][active[j]]
        
        # Find best pair (i, j) minimizing Q(i,j) = (k-2)*d(i,j) - r[i] - r[j]
        best_q = float('inf')
        best_i = -1
        best_j = -1
        
        for i in range(k):
            for j in range(i + 1, k):
                node_i = active[i]
                node_j = active[j]
                q = (k - 2) * dist[node_i][node_j] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i = i
                    best_j = j
        
        node_a = active[best_i]
        node_b = active[best_j]
        
        # Branch lengths
        delta = (r[best_i] - r[best_j]) / (k - 2)
        len_a = 0.5 * (dist[node_a][node_b] + delta)
        len_b = 0.5 * (dist[node_a][node_b] - delta)
        
        new_node = next_id
        next_id += 1
        
        # Add new node
        children[new_node] = []
        children[node_a].append((new_node, len_a))
        children[node_b].append((new_node, len_b))
        children[new_node].append((node_a, len_a))
        children[new_node].append((node_b, len_b))
        
        # Update distances from new_node to all other active nodes
        for x in active:
            if x != node_a and x != node_b:
                dist[new_node][x] = 0.5 * (dist[node_a][x] + dist[node_b][x] - dist[node_a][node_b])
                dist[x][new_node] = dist[new_node][x]
        
        # Remove a and b from active, add new_node
        active = [x for x in active if x != node_a and x != node_b]
        active.append(new_node)
    
    # Finish with 2 nodes: connect them
    if len(active) == 2:
        u, v = active
        d_uv = dist[u][v]
        # Split the distance equally (arbitrary for unrooted)
        len_u = 0.5 * d_uv
        len_v = 0.5 * d_uv
        root = next_id
        next_id += 1
        children[root] = []
        children[u].append((root, len_u))
        children[v].append((root, len_v))
        children[root].append((u, len_u))
        children[root].append((v, len_v))
    else:
        root = active[0]
    
    # Build Newick string
    def build_newick(node, parent=None):
        if node < n:
            return labels_map[node]
        
        # Find children that are not the parent
        my_children = [(c, l) for c, l in children[node] if c != parent]
        
        if len(my_children) == 0:
            return labels_map.get(node, f"n{node}")
        
        parts = []
        for c, l in my_children:
            child_str = build_newick(c, node)
            parts.append(f"{child_str}:{l:.4f}")
        
        return f"({','.join(parts)})"
    
    newick = build_newick(root)
    if not newick.endswith(";"):
        newick += ";"
    
    return newick

def _get_splits_from_newick(newick, n):
    """Parse Newick and extract non-trivial splits."""
    # Remove trailing semicolon
    newick = newick.strip().rstrip(";")
    
    # Parse to get tree structure
    # Returns a set of frozensets representing splits
    
    # We need to parse the Newick string
    # For a binary unrooted tree, each internal edge defines a split
    
    # Let's use a stack-based parser
    # Each node: (leaf_set, is_internal)
    
    splits = set()
    all_leaves = frozenset(range(n))
    
    def parse_newick(s):
        """
        Parse Newick string, return (leaf_set, list_of_splits)
        """
        i = 0
        
        def parse_node():
            nonlocal i
            if i >= len(s):
                return frozenset(), []
            
            if s[i] == '(':
                i += 1
                # Parse children
                child_sets = []
                child_splits = []
                while i < len(s) and s[i] != ')':
                    cs, csplit = parse_node()
                    child_sets.append(cs)
                    child_splits.extend(csplit)
                    if i < len(s) and s[i] == ',':
                        i += 1
                if i < len(s) and s[i] == ')':
                    i += 1
                
                # Merge all child leaf sets
                total_set = frozenset()
                for cs in child_sets:
                    total_set = total_set | cs
                
                # Add splits from children
                # Also, for each child, if it's a non-trivial subset, add it
                # But we need to be careful: the split is defined by the edge
                # In a binary tree, each internal node has exactly 2 children
                # The split for an internal node is (child1_leaves | rest)
                
                if len(child_sets) == 2:
                    # Binary node
                    c1 = child_sets[0]
                    c2 = child_sets[1]
                    # The split is c1 vs (c2 | rest) but since this is the whole subtree,
                    # the split is c1 vs c2 only if this is the root
                    # Actually, for a subtree, the split is c1 vs (total - c1)
                    # But total is just c1 | c2 here
                    # So the split is c1 vs c2
                    if 0 < len(c1) < n and 0 < len(c2) < n:
                        # Canonical form: use the smaller side
                        if len(c1) < len(c2):
                            splits.add(c1)
                        elif len(c2) < len(c1):
                            splits.add(c2)
                        else:
                            # Equal size, use lexicographically smaller
                            s1 = tuple(sorted(c1))
                            s2 = tuple(sorted(c2))
                            splits.add(frozenset(s1 if s1 < s2 else s2))
                
                return total_set, child_splits
            else:
                # Leaf
                start = i
                while i < len(s) and s[i] not in ',):':
                    i += 1
                label = s[start:i]
                # Parse label to get index
                # Labels are t0, t1, ..., t(n-1)
                try:
                    idx = int(label[1:])
                except:
                    idx = -1
                return frozenset([idx]), []
        
        result_set, result_splits = parse_node()
        return result_set, result_splits
    
    parse_newick(newick)
    return splits

def _rf_distance(splits1, splits2, n):
    """Compute normalized Robinson-Foulds distance between two sets of splits."""
    if not splits1 and not splits2:
        return 0.0
    
    # Non-trivial splits: size strictly between 1 and n-1
    # (We already filtered in _get_splits_from_newick)
    
    common = splits1 & splits2
    total = len(splits1 | splits2)
    
    if total == 0:
        return 0.0
    
    # Normalized RF distance: (|s1| + |s2| - 2*|intersection|) / (|s1| + |s2|)
    # Or: 1 - 2*|intersection| / (|s1| + |s2|)
    # Standard: (|s1 \ s2| + |s2 \ s1|) / (|s1| + |s2|)
    # = (|s1| + |s2| - 2*|s1 ∩ s2|) / (|s1| + |s2|)
    
    d = (len(splits1) + len(splits2) - 2 * len(common)) / (len(splits1) + len(splits2))
    return d

def reconstruct(sequences):
    start_time = time.time()
    time_limit = 250.0  # Safety margin before 270s kill
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return f"{labels[0]};" if n == 1 else ""
    
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    # 1. Compute Distance Matrix
    dist_mat = _jc_distance_matrix(sequences, labels)
    
    # 2. Build initial tree with NJ
    newick = _nj_tree(dist_mat, n)
    
    # Check time
    if time.time() - start_time > time_limit * 0.5:
        return newick
    
    # 3. NNI Refinement
    # For NNI, we need to work with the tree structure
    # Let's implement a simple NNI using the distance matrix
    
    # First, let's parse the Newick to get the tree structure
    # Then perform NNI moves
    
    # For efficiency, let's use a different approach:
    # Use the distance matrix to evaluate NNI moves
    
    # Get splits from current tree
    current_splits = _get_splits_from_newick(newick, n)
    
    # Perform NNI refinement
    # This is complex, so let's use a simpler heuristic:
    # Try to improve the tree by checking if the current splits
    # are consistent with the distance matrix
    
    # For now, let's just return the NJ tree
    # NNI refinement would require significant additional code
    
    return newick
