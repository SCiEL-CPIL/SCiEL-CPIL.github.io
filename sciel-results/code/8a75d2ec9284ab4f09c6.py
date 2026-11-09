"""Fast Distance-based Quartet (FDQ) topology reconstruction under Jukes-Cantor."""
import time
import math
import numpy as np

_DEADLINE_MARGIN = 2.0  # seconds to keep before the 270s wall-clock kill
_START = time.time()


def _elapsed():
    return time.time() - _START


def _remaining():
    return 270.0 - _elapsed() - _DEADLINE_MARGIN


def _jc_distance_matrix(sequences, labels):
    """Compute JC-corrected pairwise distances using vectorized NumPy ops."""
    n = len(labels)
    # Build a numeric matrix: map A,C,G,T to 0,1,2,3
    seqs = [sequences[lb] for lb in labels]
    L = len(seqs[0])
    # Encode each sequence as an integer array
    enc = {}
    for ch, val in (('A', 0), ('C', 1), ('G', 2), ('T', 3)):
        enc[ch] = val
    # Use a list of bytearrays for speed, or just arrays
    mat = np.zeros((n, L), dtype=np.int8)
    for i, s in enumerate(seqs):
        # s is a string; encode
        arr = np.frombuffer(s.encode('ascii'), dtype=np.uint8)
        # Map: A=65, C=67, G=71, T=84
        # We'll do a lookup via np.where
        # Actually, let's just build a lookup table
        lut = np.zeros(256, dtype=np.int8)
        lut[ord('A')] = 0
        lut[ord('C')] = 1
        lut[ord('G')] = 2
        lut[ord('T')] = 3
        mat[i] = lut[arr]

    # Pairwise differences: for each pair (i,j), count mismatches
    # d[i,j] = sum over positions where mat[i,pos] != mat[j,pos]
    # This is O(n^2 * L) which for n=32, L=2000 is ~2M ops, fine.
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.count_nonzero(mat[i] != mat[j])
            p = diff / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _quartet_scores_from_dist(dist):
    """
    For every quartet (a,b,c,d), compute the score for each of the 3 topologies.
    Returns a dict mapping (a,b,c,d) sorted tuple -> list of (score, topology_code)
    where topology_code is 0 for ((a,b),(c,d)), 1 for ((a,c),(b,d)), 2 for ((a,d),(b,c)).
    Score = d_ab + d_cd - d_ac - d_bd (lower is better for topology 0, etc.)
    Actually, the standard quartet test: the correct topology minimizes the sum of two opposite branch lengths.
    Under JC, the expected distance for a split is proportional to the number of sites in the internal branch.
    The quartet score for topology ((a,b)|(c,d)) is d_ab + d_cd.
    We pick the topology with the smallest d_xy + d_zw.
    """
    n = dist.shape[0]
    # scores[quartet] = [score0, score1, score2]
    # where score0 = d[a,b]+d[c,d], score1 = d[a,c]+d[b,d], score2 = d[a,d]+d[b,c]
    scores = {}
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    s0 = dist[a, b] + dist[c, d]
                    s1 = dist[a, c] + dist[b, d]
                    s2 = dist[a, d] + dist[b, c]
                    scores[(a, b, c, d)] = (s0, s1, s2)
    return scores


def _build_tree_from_quartets(scores, n):
    """
    Agglomerative tree construction from quartet votes.
    We use a simple approach: start with a star, then repeatedly add the most supported
    quartet that is compatible with the current tree.
    For simplicity and speed, we'll use a Bionj-like or FastTree-like heuristic:
    1. Compute all quartet scores.
    2. For each quartet, determine the winning topology.
    3. Use a compatibility-based agglomerative method.
    
    Given time constraints, let's use a simpler but effective method:
    Use the quartet scores to build a weighted split matrix, then use a greedy
    agglomerative clustering that respects quartet constraints.
    
    Actually, let's use a well-known fast method: 
    - Compute the "quartet score matrix" where for each pair (i,j), we accumulate
      support for splits.
    - Then use a nearest-neighbor joining variant guided by quartets.
    
    For robustness within the time budget, let's use a hybrid:
    1. Compute quartet scores.
    2. For each leaf pair, compute a "quartet-based distance" adjustment.
    3. Use NJ with quartet-informed distances.
    
    Alternative: Use the quartet scores to build a tree via the "Quartet Fitting" 
    method, but that's complex. Let's use a simpler agglomerative approach:
    
    Start with n leaves. At each step, find the pair (i,j) that maximizes the 
    number of quartets supporting their adjacency (i.e., quartets where i and j 
    are on the same side of the split). Merge them.
    """
    # Build a support matrix: support[i,j] = number of quartets where i and j 
    # are in the same clade (i.e., the quartet's winning topology groups i and j together)
    support = np.zeros((n, n))
    for (a, b, c, d), (s0, s1, s2) in scores.items():
        # Determine winning topology
        min_s = min(s0, s1, s2)
        if min_s == s0:
            # ((a,b),(c,d)): a&b together, c&d together
            support[a, b] += 1
            support[b, a] += 1
            support[c, d] += 1
            support[d, c] += 1
        elif min_s == s1:
            # ((a,c),(b,d)): a&c together, b&d together
            support[a, c] += 1
            support[c, a] += 1
            support[b, d] += 1
            support[d, b] += 1
        else:
            # ((a,d),(b,c)): a&d together, b&c together
            support[a, d] += 1
            support[d, a] += 1
            support[b, c] += 1
            support[c, b] += 1

    # Agglomerative: repeatedly merge the pair with highest support
    # Maintain a tree structure
    clusters = list(range(n))
    tree = {i: i for i in range(n)}
    remaining = list(range(n))
    
    while len(remaining) > 2:
        if _remaining() < 0.5:
            break
        best_pair = None
        best_score = -1
        for i_idx in range(len(remaining)):
            for j_idx in range(i_idx + 1, len(remaining)):
                a = remaining[i_idx]
                b = remaining[j_idx]
                if support[a, b] > best_score:
                    best_score = support[a, b]
                    best_pair = (a, b)
        if best_pair is None:
            break
        a, b = best_pair
        # Merge a and b into a new cluster
        new_id = max(remaining) + 1
        tree[new_id] = (tree[a], tree[b])
        # Update support for the new cluster
        for x in remaining:
            if x != a and x != b:
                support[new_id, x] = support[a, x] + support[b, x]
                support[x, new_id] = support[x, a] + support[x, b]
        remaining.remove(a)
        remaining.remove(b)
        remaining.append(new_id)
    
    if len(remaining) == 2:
        a, b = remaining
        tree['root'] = (tree[a], tree[b])
    elif len(remaining) == 1:
        tree['root'] = (tree[remaining[0]], None)
    else:
        # Should not happen
        tree['root'] = tuple(tree[x] for x in remaining)
    
    return tree['root']


def _tree_to_newick(tree, labels):
    """Convert a tuple tree to Newick string."""
    if tree is None:
        return ""
    if isinstance(tree, int):
        return labels[tree]
    if isinstance(tree, tuple):
        left = _tree_to_newick(tree[0], labels)
        right = _tree_to_newick(tree[1], labels)
        return f"({left},{right})"
    return str(tree)


def _nj_fallback(sequences, labels):
    """Standard Neighbor Joining as fallback."""
    n = len(labels)
    dist = _jc_distance_matrix(sequences, labels)
    
    nodes = list(range(n))
    tree = {i: i for i in range(n)}
    
    while len(nodes) > 3:
        if _remaining() < 0.5:
            break
        m = len(nodes)
        r = np.array([sum(dist[i, j] for j in nodes if j != i) for i in nodes])
        best = None
        best_criterion = float('inf')
        for i in range(m):
            for j in range(i + 1, m):
                a = nodes[i]
                b = nodes[j]
                criterion = (m - 2) * dist[a, b] - r[i] - r[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best = (a, b)
        a, b = best
        new_id = max(nodes) + 1
        tree[new_id] = (tree[a], tree[b])
        rest = [x for x in nodes if x != a and x != b]
        for x in rest:
            dist[new_id, x] = 0.5 * (dist[a, x] + dist[b, x] - dist[a, b])
            dist[x, new_id] = dist[new_id, x]
        nodes = rest + [new_id]
    
    if len(nodes) == 3:
        a, b, c = nodes
        tree['root'] = (tree[a], (tree[b], tree[c]))
    else:
        tree['root'] = tuple(tree[x] for x in nodes)
    
    return _tree_to_newick(tree['root'], labels)


def reconstruct(sequences):
    """Reconstruct unrooted tree topology from sequences."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    # Compute distance matrix
    dist = _jc_distance_matrix(sequences, labels)
    
    # For small n, use quartet-based method; for large n, use NJ as fallback
    # Quartet method is O(n^4) which is fine for n <= 20
    if n <= 20 and _remaining() > 5.0:
        try:
            scores = _quartet_scores_from_dist(dist)
            tree = _build_tree_from_quartets(scores, n)
            newick = _tree_to_newick(tree, labels)
            # Validate: ensure all labels present
            if all(lb in newick for lb in labels):
                return newick + ";"
        except Exception:
            pass
    
    # Fallback to NJ
    newick = _nj_fallback(sequences, labels)
    return newick + ";"
