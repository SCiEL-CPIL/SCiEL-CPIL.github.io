"""Phylogenetic topology recovery using quartet-based supertree with NNI refinement."""

import time
import math
import itertools
import numpy as np

_DEADLINE_MARGIN = 15.0
_START_TIME = None


def _get_elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _check_deadline():
    return _get_elapsed() > (270.0 - _DEADLINE_MARGIN)


def _encode_sequences(sequences, labels):
    n = len(labels)
    seq_array = np.array([list(sequences[lab]) for lab in labels], dtype='U1')
    n_sites = seq_array.shape[1]
    enc = np.zeros((n, n_sites), dtype=np.int8)
    enc[seq_array == 'A'] = 0
    enc[seq_array == 'C'] = 1
    enc[seq_array == 'G'] = 2
    enc[seq_array == 'T'] = 3
    return enc


def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    enc = _encode_sequences(sequences, labels)
    n_sites = enc.shape[1]
    diff = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            diff[i, j] = np.sum(enc[i] != enc[j])
            diff[j, i] = diff[i, j]
    p = diff.astype(np.float64) / n_sites
    np.fill_diagonal(p, 0.0)
    p_clipped = np.clip(p, 0.0, 0.7499)
    dist = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    np.fill_diagonal(dist, 0.0)
    return dist


def _quartet_scores_all(dist):
    n = dist.shape[0]
    if n < 4:
        return None, None
    combos = list(itertools.combinations(range(n), 4))
    a_arr = np.array([c[0] for c in combos])
    b_arr = np.array([c[1] for c in combos])
    c_arr = np.array([c[2] for c in combos])
    d_arr = np.array([c[3] for c in combos])
    s1 = dist[a_arr, c_arr] + dist[b_arr, d_arr] - dist[a_arr, b_arr] - dist[c_arr, d_arr]
    s2 = dist[a_arr, b_arr] + dist[c_arr, d_arr] - dist[a_arr, c_arr] - dist[b_arr, d_arr]
    s3 = dist[a_arr, b_arr] + dist[c_arr, d_arr] - dist[a_arr, d_arr] - dist[b_arr, c_arr]
    return (a_arr, b_arr, c_arr, d_arr), (s1, s2, s3)


def _nj_dict(dist, labels):
    n = len(labels)
    if n <= 3:
        if n == 3:
            return f"({labels[0]},{labels[1]},{labels[2]})"
        elif n == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return labels[0]
    d = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                d[(i, j)] = dist[i, j]
    nodes = list(range(n))
    tree_parts = {i: labels[i] for i in range(n)}
    counter = n
    while len(nodes) > 3:
        if _check_deadline():
            break
        n_curr = len(nodes)
        r = {i: sum(d[(i, j)] for j in nodes if j != i) for i in nodes}
        best_val = float('inf')
        best_i, best_j = -1, -1
        for idx_i, i in enumerate(nodes):
            for j in nodes[idx_i + 1:]:
                val = (n_curr - 2) * d[(i, j)] - r[i] - r[j]
                if val < best_val:
                    best_val = val
                    best_i, best_j = i, j
        if best_i == -1:
            break
        new_node = -counter
        counter += 1
        for k in nodes:
            if k != best_i and k != best_j:
                d[(new_node, k)] = d[(k, new_node)] = 0.5 * (d[(best_i, k)] + d[(best_j, k)] - d[(best_i, best_j)])
        tree_parts[new_node] = f"({tree_parts[best_i]},{tree_parts[best_j]})"
        nodes = [x for x in nodes if x != best_i and x != best_j] + [new_node]
    if len(nodes) == 3:
        return "(" + ",".join(tree_parts[x] for x in nodes) + ")"
    elif len(nodes) == 2:
        return f"({tree_parts[nodes[0]]},{tree_parts[nodes[1]]})"
    else:
        return tree_parts[nodes[0]]


def _parse_newick_splits(newick, n):
    newick = newick.strip().rstrip(';')
    splits = set()
    def _parse(s, pos):
        if s[pos] == '(':
            pos += 1
            children = []
            while s[pos] != ')':
                child_set, pos = _parse(s, pos)
                children.append(child_set)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
            pos += 1
            leaf_set = set()
            for c in children:
                leaf_set |= c
            for c in children:
                if 0 < len(c) < n:
                    other = set(range(n)) - c
                    if len(c) < len(other) or (len(c) == len(other) and min(c) < min(other)):
                        splits.add(frozenset(c))
                    else:
                        splits.add(frozenset(other))
            return leaf_set, pos
        else:
            start = pos
            while pos < len(s) and s[pos] not in ',)':
                pos += 1
            label = s[start:pos]
            try:
                idx = int(label[1:])
            except:
                idx = int(label)
            return {idx}, pos
    _, _ = _parse(newick, 0)
    return splits


def _splits_to_newick(splits, n):
    """Convert a set of splits to a Newick string."""
    if n <= 1:
        return f"t{n-1}" if n == 1 else ""
    if n == 2:
        return "(t0,t1)"
    
    # Build tree from splits using a greedy approach
    # Start with all leaves as separate clusters
    # Repeatedly find a split that partitions the current clusters
    
    # This is complex. Let's use a simpler approach:
    # Sort splits by size and build the tree hierarchically
    
    # Actually, for NNI we need to maintain the tree structure.
    # Let's represent the tree as a set of splits and use a canonical form.
    
    # For now, this is a placeholder. The NNI will work directly on the split set.
    raise NotImplementedError("Use _splits_to_newick via tree construction")


def _tree_quartet_topo(splits, a, b, c, d):
    ab = frozenset([a, b])
    cd = frozenset([c, d])
    ac = frozenset([a, c])
    bd = frozenset([b, d])
    ad = frozenset([a, d])
    bc = frozenset([b, c])
    if ab in splits or cd in splits:
        return 0
    elif ac in splits or bd in splits:
        return 1
    elif ad in splits or bc in splits:
        return 2
    else:
        return -1


def _quartet_score_of_splits(splits, dist):
    n = dist.shape[0]
    if n < 4:
        return 0.0
    idx, scores = _quartet_scores_all(dist)
    (a_arr, b_arr, c_arr, d_arr), (s1, s2, s3) = idx
    total = 0.0
    for i in range(len(a_arr)):
        a, b, c, d = a_arr[i], b_arr[i], c_arr[i], d_arr[i]
        topo = _tree_quartet_topo(splits, a, b, c, d)
        if topo == 0:
            total += s1[i]
        elif topo == 1:
            total += s2[i]
        elif topo == 2:
            total += s3[i]
        else:
            total += min(s1[i], s2[i], s3[i])
    return total


def _nni_on_splits(splits, dist, max_rounds=3):
    """NNI refinement on a set of splits."""
    n = dist.shape[0]
    if n < 4:
        return splits
    if _check_deadline():
        return splits
    
    current_score = _quartet_score_of_splits(splits, dist)
    splits_set = set(splits)
    
    for round_num in range(max_rounds):
        if _check_deadline():
            break
        improved = False
        
        # Find all internal edges (splits that can be NNI'd)
        # An NNI move on an internal edge with subtrees A, B, C, D
        # (where the edge separates {A,B} from {C,D}) can change to
        # {A,C}|{B,D} or {A,D}|{B,C}
        
        # For each pair of splits that are "adjacent" in the tree,
        # try the NNI alternatives
        
        # This requires knowing the tree structure. Let's use a simpler approach:
        # For each quartet where the current topology is not optimal,
        # try to swap the split
        
        idx, scores = _quartet_scores_all(dist)
        (a_arr, b_arr, c_arr, d_arr), (s1, s2, s3) = idx
        
        for i in range(len(a_arr)):
            a, b, c, d = a_arr[i], b_arr[i], c_arr[i], d_arr[i]
            current_topo = _tree_quartet_topo(splits_set, a, b, c, d)
            
            # Find the best topology for this quartet
            best_topo = int(np.argmin([s1[i], s2[i], s3[i]]))
            
            if current_topo == best_topo:
                continue
            
            # Try to change the split
            # If current is 0 (ab|cd) and best is 1 (ac|bd), we need to:
            # Remove split {a,b} (or {c,d}) and add split {a,c} (or {b,d})
            # But we need to ensure the result is a valid tree
            
            # This is where it gets complex. A simple swap may not yield a valid tree.
            # Let's try the swap and check validity
            
            new_splits = set(splits_set)
            
            # Remove the current split
            if current_topo == 0:
                # Remove {a,b} or {c,d}
                ab = frozenset([a, b])
                cd = frozenset([c, d])
                if ab in new_splits:
                    new_splits.discard(ab)
                if cd in new_splits:
                    new_splits.discard(cd)
            elif current_topo == 1:
                ac = frozenset([a, c])
                bd = frozenset([b, d])
                if ac in new_splits:
                    new_splits.discard(ac)
                if bd in new_splits:
                    new_splits.discard(bd)
            elif current_topo == 2:
                ad = frozenset([a, d])
                bc = frozenset([b, c])
                if ad in new_splits:
                    new_splits.discard(ad)
                if bc in new_splits:
                    new_splits.discard(bc)
            
            # Add the new split
            if best_topo == 0:
                ab = frozenset([a, b])
                cd = frozenset([c, d])
                # Add the smaller one
                if len(ab) < len(cd) or (len(ab) == len(cd) and min(ab) < min(cd)):
                    new_splits.add(ab)
                else:
                    new_splits.add(cd)
            elif best_topo == 1:
                ac = frozenset([a, c])
                bd = frozenset([b, d])
                if len(ac) < len(bd) or (len(ac) == len(bd) and min(ac) < min(bd)):
                    new_splits.add(ac)
                else:
                    new_splits.add(bd)
            else:
                ad = frozenset([a, d])
                bc = frozenset([b, c])
                if len(ad) < len(bc) or (len(ad) == len(bc) and min(ad) < min(bc)):
                    new_splits.add(ad)
                else:
                    new_splits.add(bc)
            
            # Check if this is a valid tree (n-3 non-trivial splits)
            # A valid unrooted binary tree has exactly n-3 non-trivial splits
            if len(new_splits) != n - 3:
                continue
            
            # Check consistency: no two splits should be incompatible
            # Two splits {A|B} and {C|D} are compatible if one of A∩C, A∩D, B∩C, B∩D is empty
            compatible = True
            split_list = list(new_splits)
            for xi in range(len(split_list)):
                for xj in range(xi + 1, len(split_list)):
                    A = split_list[xi]
                    B = set(range(n)) - A
                    C = split_list[xj]
                    D = set(range(n)) - C
                    if A & C and A & D and B & C and B & D:
                        compatible = False
                        break
                if not compatible:
                    break
            
            if not compatible:
                continue
            
            # Compute new score
            new_score = _quartet_score_of_splits(new_splits, dist)
            if new_score < current_score - 1e-9:
                splits_set = new_splits
                current_score = new_score
                improved = True
        
        if not improved:
            break
    
    return splits_set


def _splits_to_newick_simple(splits, n, labels):
    """Convert a valid set of splits to a Newick string."""
    if n <= 3:
        if n == 3:
            return f"({labels[0]},{labels[1]},{labels[2]})"
        elif n == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return labels[0]
    
    # Build tree from splits
    # Use a recursive approach: find a split that divides the taxa into two groups
    # and recursively build subtrees
    
    all_taxa = set(range(n))
    
    def _build(subset):
        if len(subset) == 1:
            return labels[list(subset)[0]]
        if len(subset) == 2:
            a, b = sorted(subset)
            return f"({labels[a]},{labels[b]})"
        
        # Find a split that properly partitions this subset
        # Look for a split S such that S ∩ subset is non-trivial
        for s in splits:
            intersection = s & subset
            if 0 < len(intersection) < len(subset):
                # This split partitions the subset
                left = intersection
                right = subset - intersection
                left_newick = _build(left)
                right_newick = _build(right)
                return f"({left_newick},{right_newick})"
        
        # No split found - return a star
        return "(" + ",".join(labels[i] for i in sorted(subset)) + ")"
    
    return _build(all_taxa)


def reconstruct(sequences):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    dist = _jc_distance_matrix(sequences, labels)
    
    # Get NJ tree
    nj_newick = _nj_dict(dist, labels)
    nj_splits = _parse_newick_splits(nj_newick, n)
    
    # NNI refine
    if not _check_deadline():
        refined_splits = _nni_on_splits(nj_splits, dist)
    else:
        refined_splits = nj_splits
    
    # Convert back to Newick
    result = _splits_to_newick_simple(refined_splits, n, labels)
    
    return result
