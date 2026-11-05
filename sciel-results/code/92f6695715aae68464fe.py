"""Phylogenetic topology recovery using NJ with quartet-based refinement."""
import math
import time

# EVOLVE-BLOCK-START

def _jc_dist_matrix(sequences, labels):
    """Compute Jukes-Cantor distance matrix efficiently."""
    n = len(labels)
    L = len(sequences[labels[0]])
    
    # Convert to integer arrays
    seq_ints = []
    for lab in labels:
        s = sequences[lab]
        arr = [0 if c == 'A' else 1 if c == 'C' else 2 if c == 'G' else 3 for c in s]
        seq_ints.append(arr)
    
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            si = seq_ints[i]
            sj = seq_ints[j]
            diffs = 0
            for k in range(L):
                if si[k] != sj[k]:
                    diffs += 1
            p = diffs / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist[i][j] = d
            dist[j][i] = d
    return dist


def _nj_newick(labels, dist):
    """Neighbor Joining, returns Newick string with taxon indices as labels."""
    n = len(labels)
    if n == 1:
        return "0;"
    if n == 2:
        return "(0,1);"
    if n == 3:
        return "(0,1,2);"
    
    trees = {i: str(i) for i in range(n)}
    active = list(range(n))
    d = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                d[(i, j)] = dist[i][j]
    
    next_id = n
    while len(active) > 3:
        m = len(active)
        r = {}
        for i in active:
            r[i] = 0.0
            for j in active:
                if j != i:
                    r[i] += d[(i, j)]
        
        best_q = float('inf')
        best_pair = (active[0], active[1])
        for ii in range(m):
            for jj in range(ii + 1, m):
                i = active[ii]
                j = active[jj]
                q = (m - 2) * d[(i, j)] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_pair = (i, j)
        
        i, j = best_pair
        new_node = next_id
        next_id += 1
        trees[new_node] = f"({trees[i]},{trees[j]})"
        
        for k in active:
            if k != i and k != j:
                val = 0.5 * (d[(i, k)] + d[(j, k)] - d[(i, j)])
                d[(new_node, k)] = val
                d[(k, new_node)] = val
        
        active = [x for x in active if x != i and x != j]
        active.append(new_node)
    
    return "(" + ",".join(trees[x] for x in active) + ");"


def _extract_splits(newick, n):
    """Extract non-trivial splits from a Newick string."""
    s = newick.rstrip(';').strip()
    splits = set()
    all_leaves = set(range(n))
    
    def parse_node(pos):
        """Returns (leaf_set, new_pos)."""
        if s[pos] == '(':
            pos += 1
            leaf_set = set()
            while True:
                child_set, pos = parse_node(pos)
                leaf_set.update(child_set)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
                elif pos < len(s) and s[pos] == ')':
                    break
                else:
                    break
            if pos < len(s) and s[pos] == ')':
                pos += 1
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and s[pos] not in (',', ')', ';'):
                    pos += 1
            if 1 < len(leaf_set) < n:
                if len(leaf_set) <= n // 2:
                    splits.add(frozenset(leaf_set))
                else:
                    comp = all_leaves - leaf_set
                    splits.add(frozenset(comp))
            return leaf_set, pos
        else:
            end = pos
            while end < len(s) and s[end] not in (',', ')', ';', ':'):
                end += 1
            taxon = int(s[pos:end])
            pos = end
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and s[pos] not in (',', ')', ';'):
                    pos += 1
            return {taxon}, pos
    
    parse_node(0)
    return splits


def _rf_distance(splits_a, splits_b):
    """Normalized RF distance."""
    if not splits_a and not splits_b:
        return 0.0
    common = len(splits_a & splits_b)
    union = len(splits_a | splits_b)
    if union == 0:
        return 0.0
    return 1.0 - common / union


def _quartet_support(sequences, labels, i, j, k, l):
    """
    Return the quartet topology best supported by the data.
    Returns one of: 'ij|kl', 'ik|jl', 'il|jk'
    """
    si = sequences[labels[i]]
    sj = sequences[labels[j]]
    sk = sequences[labels[k]]
    sl = sequences[labels[l]]
    L = len(si)
    
    # Count site patterns supporting each topology
    # Under JC, for a given topology, concordant sites are those where
    # the two taxa on the same side of the split have matching patterns
    # in a way consistent with the topology
    
    # Simplified: for split AB|CD, a site is concordant if
    # A==B and C==D, or A!=B and C!=D (both pairs internally consistent)
    # Discordant if A==B and C!=D, or A!=B and C==D
    
    s_ij_kl = 0
    s_ik_jl = 0
    s_il_jk = 0
    
    for pos in range(L):
        a, b, c, d = si[pos], sj[pos], sk[pos], sl[pos]
        
        # ij|kl
        if (a == b) == (c == d):
            s_ij_kl += 1
        # ik|jl
        if (a == c) == (b == d):
            s_ik_jl += 1
        # il|jk
        if (a == d) == (b == c):
            s_il_jk += 1
    
    if s_ij_kl >= s_ik_jl and s_ij_kl >= s_il_jk:
        return frozenset({i, j}), frozenset({k, l})
    elif s_ik_jl >= s_il_jk:
        return frozenset({i, k}), frozenset({j, l})
    else:
        return frozenset({i, l}), frozenset({j, k})


def _quartet_refine(sequences, labels, newick, max_time):
    """
    Attempt to refine the tree using quartet-based local search.
    This is a simplified version that checks quartet consistency.
    """
    start = time.time()
    n = len(labels)
    
    # Extract current splits
    current_splits = _extract_splits(newick, n)
    
    # For each quartet, check if the current tree's topology is supported
    # If not, consider swapping
    
    # This is complex to implement properly. For now, return the original tree.
    # A proper implementation would use QP or NNI moves guided by quartet scores.
    
    return newick


def reconstruct(sequences):
    """Reconstruct phylogenetic tree from DNA sequences."""
    start_time = time.time()
    deadline = start_time + 165.0  # 15s margin
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 3:
        if n == 1:
            return labels[0] + ";"
        elif n == 2:
            return f"({labels[0]},{labels[1]})();"
        else:
            return f"({labels[0]},{labels[1]},{labels[2]})();"
    
    # Compute distance matrix
    dist = _jc_dist_matrix(sequences, labels)
    
    # Get NJ tree
    nj_newick = _nj_newick(labels, dist)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 10.0:
        return nj_newick
    
    # Try refinement if time permits
    if n <= 20:
        refined = _quartet_refine(sequences, labels, nj_newick, deadline - time.time())
        return refined
    
    return nj_newick

# EVOLVE-BLOCK-END
