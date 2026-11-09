"""Quartet-consensus phylogenetic reconstruction with NJ fallback and deadline guard."""
import math
import time
import numpy as np

DEADLINE_SECONDS = 260.0
SAFETY_MARGIN = 10.0


def _elapsed(start):
    return time.time() - start


def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor distances using NumPy."""
    n = len(labels)
    L = len(sequences[labels[0]])
    # Map chars to ints: A=0, C=1, G=2, T=3
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_array = np.zeros((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        for j in range(L):
            seq_array[i, j] = char_map.get(seq[j], 0)
    
    # Compute Hamming distances vectorized per pair
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diffs = np.count_nonzero(seq_array[i] != seq_array[j])
            p = diffs / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix


def _quartet_scores(dist_matrix, n):
    """
    For each quartet (a,b,c,d), compute the three possible split lengths.
    The true split minimizes the sum of squared branch lengths (or equivalently,
    we pick the split with the smallest "quartet score").
    
    For unrooted quartet with taxa a,b,c,d and distances d_ij:
    Three possible splits:
      1. (a,b)|(c,d): internal branch = (d_ac + d_bd - d_ab - d_cd) / 2 ... 
         Actually the standard quartet method uses:
         For split (a,b)|(c,d): score = d_ac + d_bd - d_ab - d_cd
         For split (a,c)|(b,d): score = d_ad + d_bc - d_ab - d_cd
         For split (a,d)|(b,c): score = d_ab + d_cd - d_ac - d_bd
         
    The correct split has the most negative score (or smallest score).
    """
    # Returns a dict mapping frozenset of 4 taxa -> the winning split (as a frozenset of the two groups)
    quartet_results = {}
    
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    d_ab = dist_matrix[a, b]
                    d_ac = dist_matrix[a, c]
                    d_ad = dist_matrix[a, d]
                    d_bc = dist_matrix[b, c]
                    d_bd = dist_matrix[b, d]
                    d_cd = dist_matrix[c, d]
                    
                    # Three possible splits and their scores
                    # Split 1: (a,b)|(c,d)
                    score1 = d_ac + d_bd - d_ab - d_cd
                    # Split 2: (a,c)|(b,d)
                    score2 = d_ad + d_bc - d_ab - d_cd
                    # Split 3: (a,d)|(b,c)
                    score3 = d_ab + d_cd - d_ac - d_bd
                    
                    if score1 <= score2 and score1 <= score3:
                        winning_split = (frozenset([a, b]), frozenset([c, d]))
                    elif score2 <= score1 and score2 <= score3:
                        winning_split = (frozenset([a, c]), frozenset([b, d]))
                    else:
                        winning_split = (frozenset([a, d]), frozenset([b, c]))
                    
                    quartet_results[frozenset([a, b, c, d])] = winning_split
    
    return quartet_results


def _build_consensus_tree(quartet_results, n, labels):
    """
    Build a consensus tree from quartet results.
    We collect all splits (as partitions of the full taxon set) and find a compatible set.
    
    For each quartet, the winning split (e.g., {a,b}|{c,d}) induces a split on all n taxa:
    One side contains {a,b} plus any other taxa that are "closer" to a,b in the quartet sense.
    Actually, the standard approach: the split from quartet (a,b)|(c,d) on the full set is:
    Side 1 = {a, b} ∪ {x : x is on the same side as a or b in the quartet}
    But we don't know which side x is on without more info.
    
    Alternative: use the quartet split directly as a 2-way split on the 4 taxa,
    and extend it to all taxa by: for each other taxon x, x goes on the side of the
    quartet split where it has smaller distance to the pair.
    
    Simpler approach: collect all candidate splits (as frozensets of one side),
    check compatibility (two splits are compatible if they don't cross),
    and build a tree from a maximal compatible set.
    """
    # Collect all splits as frozensets (one side of the split, with size between 2 and n-2)
    splits = []
    for quartet, winning_split in quartet_results.items():
        side1, side2 = winning_split
        # Extend to full taxon set
        # For each taxon not in the quartet, assign to the side with smaller average distance
        all_taxa = set(range(n))
        quartet_taxa = set(quartet)
        other_taxa = all_taxa - quartet_taxa
        
        # Determine which side each other taxon belongs to
        # Use distance heuristic: taxon x goes to side S if avg dist(x, S) < avg dist(x, other side)
        full_side1 = set(side1)
        full_side2 = set(side2)
        
        for x in other_taxa:
            avg_d1 = sum(dist_matrix[x, y] for y in side1) / len(side1)
            avg_d2 = sum(dist_matrix[x, y] for y in side2) / len(side2)
            if avg_d1 <= avg_d2:
                full_side1.add(x)
            else:
                full_side2.add(x)
        
        # Store the smaller side (to canonicalize)
        if len(full_side1) <= n // 2:
            split = frozenset(full_side1)
        else:
            split = frozenset(full_side2)
        
        if 2 <= len(split) <= n - 2:
            splits.append(split)
    
    # Remove duplicates
    unique_splits = list(set(splits))
    
    # Find a maximal compatible set of splits
    # Two splits S and T are compatible if none of S∩T, S∩(U\T), (U\S)∩T, (U\S)∩(U\T) is empty
    # where U is the full set
    full_set = frozenset(range(n))
    
    def compatible(s1, s2):
        s1_comp = full_set - s1
        s2_comp = full_set - s2
        return not (s1 & s2 and s1 & s2_comp and s1_comp & s2 and s1_comp & s2_comp)
    
    # Greedy: sort splits by frequency (more common first), then add if compatible
    # Count frequency
    from collections import Counter
    split_counts = Counter(splits)
    unique_splits.sort(key=lambda s: -split_counts[s])
    
    selected_splits = []
    for s in unique_splits:
        if all(compatible(s, t) for t in selected_splits):
            selected_splits.append(s)
    
    # Build tree from selected splits
    # Use a recursive approach: find a split that partitions the current set
    # and recurse on each side
    
    def build_tree(taxa_set):
        if len(taxa_set) <= 2:
            if len(taxa_set) == 1:
                return labels[list(taxa_set)[0]]
            else:
                return f"({labels[list(taxa_set)[0]]},{labels[list(taxa_set)[1]]})"
        
        # Find a split that properly partitions taxa_set
        best_split = None
        for s in selected_splits:
            s_in = taxa_set & s
            s_out = taxa_set - s_in
            if 2 <= len(s_in) <= len(taxa_set) - 2:
                best_split = s_in
                break
        
        if best_split is None:
            # No split found, return polytomy
            return "(" + ",".join(labels[t] for t in sorted(taxa_set)) + ")"
        
        left = build_tree(frozenset(best_split))
        right = build_tree(frozenset(taxa_set - best_split))
        return f"({left},{right})"
    
    return build_tree(frozenset(range(n)))


def _neighbor_joining(dist_matrix, labels):
    """Standard Neighbor-Joining as fallback."""
    n = len(labels)
    if n <= 2:
        return f"({','.join(labels)});"
    
    tree_repr = {i: labels[i] for i in range(n)}
    dist = dist_matrix.copy()
    active = list(range(n))
    next_node_id = n
    
    while len(active) > 3:
        current_n = len(active)
        r = np.zeros(current_n, dtype=np.float64)
        for i, idx in enumerate(active):
            r[i] = sum(dist[idx, active[j]] for j in range(current_n) if j != i)
        
        best_criterion = float('inf')
        best_i_idx = -1
        best_j_idx = -1
        best_dist = 0
        
        for i in range(current_n):
            idx_i = active[i]
            for j in range(i + 1, current_n):
                idx_j = active[j]
                d_ij = dist[idx_i, idx_j]
                criterion = (current_n - 2) * d_ij - r[i] - r[j]
                if criterion < best_criterion - 1e-9:
                    best_criterion = criterion
                    best_i_idx = i
                    best_j_idx = j
                    best_dist = d_ij
                elif abs(criterion - best_criterion) < 1e-9:
                    if d_ij < best_dist:
                        best_criterion = criterion
                        best_i_idx = i
                        best_j_idx = j
                        best_dist = d_ij
        
        i, j = best_i_idx, best_j_idx
        idx_i = active[i]
        idx_j = active[j]
        
        new_label = f"n{next_node_id}"
        next_node_id += 1
        tree_repr[new_label] = f"({tree_repr[idx_i]},{tree_repr[idx_j]})"
        
        new_active = [x for k, x in enumerate(active) if k not in (i, j)]
        new_active.append(new_label)
        
        for k in range(len(active)):
            idx_k = active[k]
            if k not in (i, j):
                d_new_k = 0.5 * (dist[idx_i, idx_k] + dist[idx_j, idx_k] - dist[idx_i, idx_j])
                # Store in a temporary dict
                if not hasattr(_neighbor_joining, '_temp_dist'):
                    _neighbor_joining._temp_dist = {}
                _neighbor_joining._temp_dist[(new_label, idx_k)] = d_new_k
                _neighbor_joining._temp_dist[(idx_k, new_label)] = d_new_k
        
        # Update dist for the new active set
        # We need to maintain a full dist matrix for active nodes
        # This is getting complex; let's use a dict-based approach
        active = new_active
        # Rebuild dist as a dict
        # Actually, let's just continue with the numpy matrix and handle the new node separately
        # For simplicity, break out if we can't handle it
        # Let's use a different approach: maintain dist as a dict from the start
    
    # This is getting too complex. Let's use a clean dict-based NJ.
    return None


def _nj_dict(dist_matrix, labels):
    """Clean dict-based Neighbor-Joining."""
    n = len(labels)
    if n <= 2:
        return f"({','.join(labels)});"
    
    dist = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[(labels[i], labels[j])] = dist_matrix[i, j]
    
    nodes = list(labels)
    trees = {name: name for name in labels}
    next_id = 0
    
    while len(nodes) > 3:
        current_n = len(nodes)
        total = {}
        for a in nodes:
            total[a] = sum(dist[(a, b)] for b in nodes if b != a)
        
        best_criterion = float('inf')
        best_a = None
        best_b = None
        best_d = 0
        
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                d_ab = dist[(a, b)]
                criterion = (current_n - 2) * d_ab - total[a] - total[b]
                if criterion < best_criterion - 1e-9:
                    best_criterion = criterion
                    best_a = a
                    best_b = b
                    best_d = d_ab
                elif abs(criterion - best_criterion) < 1e-9:
                    if d_ab < best_d:
                        best_criterion = criterion
                        best_a = a
                        best_b = b
                        best_d = d_ab
        
        a, b = best_a, best_b
        d_ab = dist[(a, b)]
        
        joined = f"({trees[a]},{trees[b]})"
        new_label = f"n{next_id}"
        next_id += 1
        
        rest = [x for x in nodes if x not in (a, b)]
        
        for x in rest:
            new_d = 0.5 * (dist[(a, x)] + dist[(b, x)] - d_ab)
            dist[(new_label, x)] = new_d
            dist[(x, new_label)] = new_d
        
        nodes = rest + [new_label]
        trees[new_label] = joined
    
    return "(" + ",".join(trees[x] for x in nodes) + ");"


def reconstruct(sequences):
    start_time = time.time()
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        return f"({','.join(labels)});"
    
    # Compute distance matrix
    dist_matrix = _jc_distance_matrix(sequences, labels)
    
    # Try quartet consensus first
    if n >= 4 and _elapsed(start_time) < DEADLINE_SECONDS - SAFETY_MARGIN:
        try:
            quartet_results = _quartet_scores(dist_matrix, n)
            tree = _build_consensus_tree(quartet_results, n, labels)
            if tree and len(tree) > 10:
                return tree
        except Exception:
            pass
    
    # Fallback to NJ
    if _elapsed(start_time) < DEADLINE_SECONDS - SAFETY_MARGIN:
        try:
            tree = _nj_dict(dist_matrix, labels)
            if tree:
                return tree
        except Exception:
            pass
    
    # Star tree fallback
    return "(" + ",".join(labels) + ");"
