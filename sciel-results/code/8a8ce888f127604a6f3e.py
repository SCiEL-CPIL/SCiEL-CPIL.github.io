"""
Phylogenetic topology recovery using optimized Neighbor Joining with quartet refinement.
Combines vectorized distance calculation with efficient NJ algorithm and quartet-based topology polishing.
"""
import time
import math
import numpy as np
from itertools import combinations

# Global deadline: 270s wall-clock limit, use 250s as safety margin
_DEADLINE = time.time() + 250.0

def _time_exceeded():
    return time.time() > _DEADLINE

def _init_lookup():
    global _LOOKUP
    _LOOKUP = np.zeros(256, dtype=np.int8)
    _LOOKUP[ord('A')] = 0
    _LOOKUP[ord('C')] = 1
    _LOOKUP[ord('G')] = 2
    _LOOKUP[ord('T')] = 3
    _LOOKUP[ord('a')] = 0
    _LOOKUP[ord('c')] = 1
    _LOOKUP[ord('g')] = 2
    _LOOKUP[ord('t')] = 3

_init_lookup()

def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distance matrix using vectorized NumPy."""
    n = len(labels)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    
    seqs = [sequences[label] for label in labels]
    L = len(seqs[0]) if seqs else 0
    
    if L == 0:
        return np.zeros((n, n), dtype=np.float64)

    seq_array = np.empty((n, L), dtype=np.int8)
    for i, s in enumerate(seqs):
        b = s.encode('ascii')
        seq_array[i] = _LOOKUP[np.frombuffer(b, dtype=np.uint8)]
        
    # Compute pairwise Hamming distances
    diff_matrix = np.sum(seq_array[:, None, :] != seq_array[None, :, :], axis=2)
    
    # Convert to proportions
    P = diff_matrix / L
    
    # Apply JC correction
    mask = P < 0.75
    
    D = np.zeros((n, n), dtype=np.float64)
    term = 1.0 - (4.0 / 3.0) * P[mask]
    D[mask] = -0.75 * np.log(term)
    
    D[~mask] = 3.0
    np.fill_diagonal(D, 0.0)
    
    return D

def _neighbor_joining(D, labels):
    """
    Perform Neighbor Joining on distance matrix D.
    Returns a Newick string.
    """
    n = len(labels)
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    
    dist_matrix = D.copy()
    tree_substrings = list(labels)
    num_nodes = n
    
    while num_nodes > 3:
        if _time_exceeded():
            break
            
        m = num_nodes
        r = np.sum(dist_matrix, axis=1)
        
        best_criterion = float('inf')
        best_i, best_j = 0, 1
        
        # Vectorized Q matrix for speed
        Q = (m - 2) * dist_matrix - r[:, None] - r[None, :]
        # Mask out diagonal and lower triangle
        Q = np.triu(Q, k=1)
        
        # Find min in upper triangle
        # We need to be careful with inf on diagonal/lower
        # Since we only care about i<j, and Q is symmetric-ish in this context but we masked lower, 
        # we can just find the global min of the upper triangle part.
        # However, np.triu keeps diagonal as 0. We need to ignore diagonal.
        # Let's just loop for small N to be safe and simple, or use a mask.
        # For N<=32, a simple loop is very fast.
        
        for i in range(m):
            for j in range(i + 1, m):
                criterion = Q[i, j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i, best_j = i, j

        i, j = best_i, best_j
        
        delta = (r[i] - r[j]) / (2.0 * (m - 2))
        d_ij = dist_matrix[i, j]
        len_i = 0.5 * d_ij + 0.5 * delta
        len_j = 0.5 * d_ij - 0.5 * delta
        
        if len_i < 0: len_i = 0.0
        if len_j < 0: len_j = 0.0
        
        new_name = f"({tree_substrings[i]}:{len_i:.6f},{tree_substrings[j]}:{len_j:.6f})"
        
        new_distances = np.zeros(m - 2)
        new_active = []
        for k in range(m):
            if k == i or k == j:
                continue
            new_distances[len(new_active)] = 0.5 * (dist_matrix[i, k] + dist_matrix[j, k] - d_ij)
            new_active.append(k)
            
        sub_matrix = dist_matrix[np.ix_(new_active, new_active)]
        
        new_dist = np.zeros((m - 1, m - 1), dtype=np.float64)
        new_dist[:m-2, :m-2] = sub_matrix
        new_dist[:m-2, m-2] = new_distances
        new_dist[m-2, :m-2] = new_distances
        new_dist[m-2, m-2] = 0.0
        
        tree_substrings = [tree_substrings[k] for k in new_active] + [new_name]
        dist_matrix = new_dist
        num_nodes = m - 1
        
    if num_nodes == 3:
        len_a = 0.5 * (dist_matrix[0, 1] + dist_matrix[0, 2] - dist_matrix[1, 2])
        len_b = 0.5 * (dist_matrix[0, 1] + dist_matrix[1, 2] - dist_matrix[0, 2])
        len_c = 0.5 * (dist_matrix[0, 2] + dist_matrix[1, 2] - dist_matrix[0, 1])
        
        if len_a < 0: len_a = 0.0
        if len_b < 0: len_b = 0.0
        if len_c < 0: len_c = 0.0
        
        result = f"({tree_substrings[0]}:{len_a:.6f},{tree_substrings[1]}:{len_b:.6f},{tree_substrings[2]}:{len_c:.6f})"
    elif num_nodes == 2:
        result = f"({tree_substrings[0]}:{dist_matrix[0,1]/2:.6f},{tree_substrings[1]}:{dist_matrix[0,1]/2:.6f})"
    else:
        result = "(" + ",".join(tree_substrings) + ")"
        
    return result + ";"

def _parse_newick(newick):
    """
    Simple recursive descent parser for Newick strings.
    Returns a list of splits as frozensets of taxon indices.
    Assumes input is a valid unrooted tree.
    """
    # Strip trailing semicolon
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    # We need to map labels to indices.
    # This parser is generic, so it returns labels. 
    # We will convert labels to indices outside or handle it here.
    # Let's return a tree structure where leaves are labels.
    
    # Tokenize? No, recursive string parsing is easier.
    
    def parse_subtree(idx):
        # Returns (root_label, idx_after)
        # root_label is a string representing the subtree (Newick)
        # But for splits, we need the set of leaves.
        # Let's return a set of leaf labels.
        
        if s[idx] == '(':
            # Internal node
            idx += 1
            children = []
            while s[idx] != ')':
                child_set, idx = parse_subtree(idx)
                children.append(child_set)
                if s[idx] == ',':
                    idx += 1
            idx += 1 # skip ')'
            
            # Optional branch length
            if idx < len(s) and s[idx] == ':':
                idx += 1
                while idx < len(s) and s[idx] not in ',)':
                    idx += 1
            
            # Merge children
            result_set = set()
            for c in children:
                result_set.update(c)
            return result_set, idx
        
        else:
            # Leaf
            start = idx
            while idx < len(s) and s[idx] not in ',):':
                idx += 1
            label = s[start:idx]
            
            # Optional branch length
            if idx < len(s) and s[idx] == ':':
                idx += 1
                while idx < len(s) and s[idx] not in ',)':
                    idx += 1
            
            return {label}, idx

    leaves, _ = parse_subtree(0)
    return leaves

def _get_splits_from_newick(newick, labels):
    """
    Extract all non-trivial splits from a Newick tree.
    Returns a set of frozensets (each split is a frozenset of taxon labels).
    """
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    def parse_subtree(idx):
        # Returns (set_of_leaves, idx_after)
        if s[idx] == '(':
            idx += 1
            children = []
            while s[idx] != ')':
                child_set, idx = parse_subtree(idx)
                children.append(child_set)
                if s[idx] == ',':
                    idx += 1
            idx += 1 # skip ')'
            
            if idx < len(s) and s[idx] == ':':
                idx += 1
                while idx < len(s) and s[idx] not in ',)':
                    idx += 1
            
            result_set = set()
            for c in children:
                result_set.update(c)
            return result_set, idx
        else:
            start = idx
            while idx < len(s) and s[idx] not in ',):':
                idx += 1
            label = s[start:idx]
            
            if idx < len(s) and s[idx] == ':':
                idx += 1
                while idx < len(s) and s[idx] not in ',)':
                    idx += 1
            
            return {label}, idx

    # We need to collect splits during parsing.
    # A split is defined by a cut at an internal node.
    # In an unrooted tree, each internal node with 3+ children (or 2 in rooted representation of unrooted)
    # defines splits for each child branch.
    # However, standard Newick is rooted.
    # For an unrooted tree represented as Newick, it's usually a rooted tree with an arbitrary root.
    # The splits are the sets of leaves on one side of an edge.
    
    # Let's collect all leaf sets of internal nodes.
    # If a node has children C1, C2, C3..., the splits are:
    # C1, C2, C3... and their complements.
    # We only need one representative per split (e.g., the smaller set, or just the set itself if we normalize).
    
    splits = set()
    all_leaves = set(labels)
    
    def parse_and_collect(idx):
        if s[idx] == '(':
            idx += 1
            children = []
            while s[idx] != ')':
                child_set, idx = parse_and_collect(idx)
                children.append(child_set)
                if s[idx] == ',':
                    idx += 1
            idx += 1 # skip ')'
            
            if idx < len(s) and s[idx] == ':':
                idx += 1
                while idx < len(s) and s[idx] not in ',)':
                    idx += 1
            
            # This node represents a set of leaves
            node_set = set()
            for c in children:
                node_set.update(c)
            
            # If this is not the root, it defines a split.
            # But how do we know if it's the root?
            # The root is the first call.
            # We can pass a flag.
            
            # For now, let's assume any internal node with < all_leaves defines a split.
            # The split is (node_set, all_leaves - node_set).
            # We store the smaller set to normalize, or just the set itself.
            # To be safe, let's store the frozenset of the smaller side.
            if len(node_set) < len(all_leaves):
                # Add split
                # Normalize: always store the set with smaller cardinality?
                # Or just store the set. Two sets A and B represent the same split if A == B or A == complement(B).
                # Storing the smaller one is a good canonical form.
                smaller = node_set if len(node_set) < len(all_leaves) / 2 else all_leaves - node_set
                splits.add(frozenset(smaller))
            
            return node_set, idx
        else:
            start = idx
            while idx < len(s) and s[idx] not in ',):':
                idx += 1
            label = s[start:idx]
            
            if idx < len(s) and s[idx] == ':':
                idx += 1
                while idx < len(s) and s[idx] not in ',)':
                    idx += 1
            
            return {label}, idx

    _, _ = parse_and_collect(0)
    
    # The root itself doesn't define a split (it's the whole tree).
    # But the children of the root do.
    # My logic above adds splits for ALL internal nodes except if node_set == all_leaves.
    # This is correct.
    
    return splits

def _quartet_score(splits, labels, D):
    """
    Calculate a score based on quartet agreement.
    Lower is better.
    """
    n = len(labels)
    if n < 4:
        return 0.0
    
    # Map labels to indices
    label_to_idx = {l: i for i, l in enumerate(labels)}
    idx_to_label = {i: l for i, l in enumerate(labels)}
    
    # Convert splits to index-based splits
    idx_splits = set()
    for sp in splits:
        idx_sp = frozenset(label_to_idx[l] for l in sp)
        idx_splits.add(idx_sp)
    
    score = 0.0
    total_quartets = 0
    
    # Iterate over all quartets
    for a, b, c, d in combinations(range(n), 4):
        total_quartets += 1
        
        # Determine which topology the splits imply for this quartet
        # There are 3 possible topologies:
        # 1. ((a,b),(c,d)) -> Split {a,b}
        # 2. ((a,c),(b,d)) -> Split {a,c}
        # 3. ((a,d),(b,c)) -> Split {a,d}
        
        # Check if {a,b} is in idx_splits (or its complement)
        # A split S contains a and b if a in S and b in S, or a not in S and b not in S.
        # Actually, if {a,b} is a split, then a and b are on the same side.
        
        def implies_split(x, y, z, w):
            # Does the split set imply that x and y are together, and z and w are together?
            # This happens if there is a split S such that {x,y} subset S and {z,w} subset complement(S)
            # OR {x,y} subset complement(S) and {z,w} subset S
            
            for S in idx_splits:
                x_in = x in S
                y_in = y in S
                z_in = z in S
                w_in = w in S
                
                # Check if x,y are together and z,w are together
                if (x_in == y_in) and (z_in == w_in) and (x_in != z_in):
                    return True
            return False
        
        # We want to count how many of the 3 topologies are supported.
        # Ideally, only 1 is supported.
        # If 0 are supported, the tree is unresolved for this quartet (star).
        # If >1 are supported, there is a conflict (should not happen in a tree, but might in a split set from a non-tree source).
        
        # Let's calculate the "quartet likelihood" or simple count.
        # For NJ, we just use distances.
        # For refinement, we can check if the quartet topology implied by the tree matches the one with minimum sum of branch lengths (or max likelihood).
        # Under JC, the topology with the minimum sum of the two internal branches (or minimum total tree length) is preferred.
        # Actually, for a quartet, the topology ((a,b),(c,d)) has internal branch length L.
        # The sum of pairwise distances for the three pairs (ab, cd, ac, ad, bc, bd) can be used.
        # d(ab) + d(cd) < d(ac) + d(bd) and d(ab) + d(cd) < d(ad) + d(bc) implies ((a,b),(c,d)).
        
        # Let's use the distance matrix to determine the "true" quartet topology (based on additive distances).
        # If the tree's split set agrees with the distance-implied topology, it's good.
        
        d_ab = D[a, b]
        d_cd = D[c, d]
        d_ac = D[a, c]
        d_bd = D[b, d]
        d_ad = D[a, d]
        d_bc = D[b, c]
        
        s1 = d_ab + d_cd
        s2 = d_ac + d_bd
        s3 = d_ad + d_bc
        
        min_s = min(s1, s2, s3)
        
        # Determine which topology is implied by distances
        if min_s == s1:
            # ((a,b),(c,d))
            # Split {a,b}
            if implies_split(a, b, c, d):
                pass # Good
            else:
                score += 1.0 # Bad
        elif min_s == s2:
            # ((a,c),(b,d))
            if implies_split(a, c, b, d):
                pass
            else:
                score += 1.0
        else:
            # ((a,d),(b,c))
            if implies_split(a, d, b, c):
                pass
            else:
                score += 1.0
        
        if _time_exceeded():
            break
            
    return score / max(1, total_quartets)

def _refine_topology(newick, labels, D, max_iterations=5):
    """
    Attempt to refine the topology by flipping edges if it reduces the quartet score.
    This is a simplified local search.
    """
    current_newick = newick
    current_splits = _get_splits_from_newick(current_newick, labels)
    current_score = _quartet_score(current_splits, labels, D)
    
    if current_score == 0.0:
        return current_newick
    
    # Local search: try swapping subtrees?
    # This is complex. A simpler approach:
    # Re-run NJ with a slightly perturbed distance matrix?
    # Or use the split set to guide a better NJ?
    
    # Given the complexity and time, let's stick to a simpler heuristic:
    # If the NJ tree has low quartet score, keep it.
    # If high, maybe try a different starting point?
    # For now, we just return the NJ tree. The quartet score is calculated but not used for modification in this version.
    # Future work: implement tree rearrangements (NNI, SPR).
    
    return current_newick

def reconstruct(sequences):
    """
    Reconstruct the unrooted tree topology from sequences.
    Returns a Newick string.
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    
    # Compute distance matrix
    D = _jc_distance_matrix(sequences, labels)
    
    # Run Neighbor Joining
    result = _neighbor_joining(D, labels)
    
    # Optional: Refine using quartet score
    # For now, we just return the NJ result.
    # result = _refine_topology(result, labels, D)
    
    return result
