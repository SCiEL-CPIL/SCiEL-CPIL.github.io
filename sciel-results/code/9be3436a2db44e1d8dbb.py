"""Phylogenetic topology recovery from Jukes-Cantor simulated sequences."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 255.0  # wall-clock safety margin before 270s kill


def _encode_sequences(sequences, labels):
    """Encode DNA sequences into int8 array (A=0,C=1,G=2,T=3)."""
    n = len(labels)
    seq_len = len(sequences[labels[0]])
    enc = np.empty((n, seq_len), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        # Vectorized char-to-int mapping
        arr = np.frombuffer(s.encode('ascii'), dtype=np.int8)
        # A=65, C=67, G=71, T=84
        enc[i] = (arr == 65) * 0 + (arr == 67) * 1 + (arr == 71) * 2 + (arr == 84) * 3
    return enc


def _compute_distance_matrix(enc):
    """Compute pairwise JC-corrected distance matrix."""
    n, L = enc.shape
    diff = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        diff[i] = np.count_nonzero(enc[i] != enc, axis=1)
    p = diff / L
    p_clipped = np.minimum(p, 0.749999)
    D = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    D[p >= 0.75] = 3.0
    D = (D + D.T) * 0.5
    np.fill_diagonal(D, 0.0)
    return D


def _neighbor_joining(D, labels, deadline=None):
    """Neighbor joining. Returns Newick string (no trailing semicolon)."""
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    D = D.astype(np.float64).copy()
    node_names = list(labels)

    while len(node_names) > 3:
        if deadline is not None and time.time() > deadline:
            return "(" + ",".join(node_names) + ")"
        m = len(node_names)
        r = D.sum(axis=1)
        Q = (m - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = idx
        if i > j:
            i, j = j, i
        new_name = f"({node_names[i]},{node_names[j]})"
        new_dist = 0.5 * (D[i, :] + D[j, :] - D[i, j])
        keep = [k for k in range(m) if k != i and k != j]
        keep_arr = np.array(keep, dtype=np.intp)
        new_m = m - 1
        new_D = np.empty((new_m, new_m), dtype=np.float64)
        new_D[:-1, :-1] = D[np.ix_(keep_arr, keep_arr)]
        new_node_idx = m - 2
        new_D[:-1, new_node_idx] = new_dist[keep_arr]
        new_D[new_node_idx, :-1] = new_dist[keep_arr]
        new_D[new_node_idx, new_node_idx] = 0.0
        node_names = [node_names[k] for k in keep] + [new_name]
        D = new_D
    return f"({node_names[0]},{node_names[1]},{node_names[2]})"


def _quartet_count_topologies(enc, idx1, idx2, idx3, idx4):
    """
    For four taxa, count how many sites support each of the three
    unrooted quartet topologies under JC.
    Returns (c12|34, c13|24, c14|23) where c_ab|cd = number of sites
    where the two taxa in the first partition agree with each other
    more than with the other partition (i.e., the split a|b vs c|d
    has smaller sum-of-distances).
    For JC, the true topology is the one where the sum of the two
    within-group distances is smaller than the sum of the cross
    distances (equivalently, the partition with smaller total
    pairwise distance within groups).
    """
    s1 = enc[idx1]
    s2 = enc[idx2]
    s3 = enc[idx3]
    s4 = enc[idx4]
    # Pairwise mismatches (0/1 per site)
    d12 = (s1 != s2)
    d34 = (s3 != s4)
    d13 = (s1 != s3)
    d24 = (s2 != s4)
    d14 = (s1 != s4)
    d23 = (s2 != s3)
    # Topology 1: (1,2)|(3,4) supported when d12 + d34 < d13 + d24 and d12 + d34 < d14 + d23
    # Topology 2: (1,3)|(2,4) supported when d13 + d24 < d12 + d34 and d13 + d24 < d14 + d23
    # Topology 3: (1,4)|(2,3) supported when d14 + d23 < d12 + d34 and d14 + d23 < d13 + d24
    # Use strict inequality; ties contribute to none (or we can break by majority)
    sum1 = d12 + d34
    sum2 = d13 + d24
    sum3 = d14 + d23
    c1 = np.count_nonzero((sum1 < sum2) & (sum1 < sum3))
    c2 = np.count_nonzero((sum2 < sum1) & (sum2 < sum3))
    c3 = np.count_nonzero((sum3 < sum1) & (sum3 < sum2))
    return c1, c2, c3


def _quartet_census(enc, n, max_quartets=None, deadline=None):
    """
    For every 4-taxon subset, determine the winning topology and tally
    which of the three splits it supports. Returns a dict mapping
    frozenset of 4 taxa -> winning split as a frozenset pair.
    Also returns per-split vote counts.
    """
    if max_quartets is None:
        max_quartets = 10_000_000
    # We'll accumulate votes for each of the three possible splits per quartet
    # A split is represented as (frozenset({a,b}), frozenset({c,d}))
    # We'll store votes as a dict: split_key -> count
    split_votes = {}
    quartets_done = 0
    start = time.time()
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                if quartets_done >= max_quartets:
                    break
                for d in range(c + 1, n):
                    if quartets_done >= max_quartets:
                        break
                    if deadline is not None and (quartets_done % 1000 == 0) and time.time() > deadline:
                        return split_votes, quartets_done
                    c1, c2, c3 = _quartet_count_topologies(enc, a, b, c, d)
                    # Topology 1: (a,b)|(c,d)
                    # Topology 2: (a,c)|(b,d)
                    # Topology 3: (a,d)|(b,c)
                    if c1 >= c2 and c1 >= c3:
                        key = (frozenset((a, b)), frozenset((c, d)))
                    elif c2 >= c1 and c2 >= c3:
                        key = (frozenset((a, c)), frozenset((b, d)))
                    else:
                        key = (frozenset((a, d)), frozenset((b, c)))
                    split_votes[key] = split_votes.get(key, 0) + 1
                    quartets_done += 1
                if quartets_done >= max_quartets:
                    break
            if quartets_done >= max_quartets:
                break
        if quartets_done >= max_quartets:
            break
    return split_votes, quartets_done


def _build_tree_from_splits(labels, split_votes, deadline=None):
    """
    Build an unrooted tree from a set of weighted splits using a
    greedy compatible-split closure + NJ on the induced distance.
    Returns a Newick string.
    """
    n = len(labels)
    if n <= 3:
        if n == 1:
            return labels[0]
        if n == 2:
            return f"({labels[0]},{labels[1]})"
        return f"({labels[0]},{labels[1]},{labels[2]})"

    # Convert split votes to a list of (split, weight)
    splits = []
    for (A, B), w in split_votes.items():
        # Ensure A is the smaller set for canonical form
        if len(A) > len(B):
            A, B = B, A
        elif len(A) == len(B) and min(A) > min(B):
            A, B = B, A
        splits.append((A, B, w))

    # Greedily select compatible splits (by weight)
    selected = []
    # Sort by weight descending
    splits.sort(key=lambda x: -x[2])
    for A, B, w in splits:
        if deadline is not None and time.time() > deadline:
            break
        # Check compatibility with all selected splits
        compatible = True
        for (A2, B2, _) in selected:
            # Two splits {A|B} and {A2|B2} on the same leaf set are compatible
            # iff at most one of A∩A2, A∩B2, B∩A2, B∩B2 is non-empty
            # (i.e., they don't cross)
            # For splits that may not cover all leaves, we check on the union
            A_set = set(A)
            B_set = set(B)
            A2_set = set(A2)
            B2_set = set(B2)
            # Only consider leaves in both splits
            common = A_set | B_set | A2_set | B2_set
            nA = A_set & common
            nB = B_set & common
            nA2 = A2_set & common
            nB2 = B2_set & common
            nonempty = 0
            if nA & nA2:
                nonempty += 1
            if nA & nB2:
                nonempty += 1
            if nB & nA2:
                nonempty += 1
            if nB & nB2:
                nonempty += 1
            if nonempty > 1:
                compatible = False
                break
        if compatible:
            selected.append((A, B, w))

    # Build a distance matrix from selected splits:
    # d(i,j) = number of selected splits that separate i and j
    # weighted by split weight
    D = np.zeros((n, n), dtype=np.float64)
    for A, B, w in selected:
        A_set = set(A)
        B_set = set(B)
        for i in A_set:
            for j in B_set:
                D[i, j] += w
                D[j, i] += w
    # Normalize
    if D.max() > 0:
        D /= D.max()
    else:
        # Fallback: use all-ones off-diagonal
        D = np.ones((n, n)) - np.eye(n)

    # Run NJ on this distance matrix
    return _neighbor_joining(D, labels, deadline)


def _nj_with_deadline(sequences, labels, deadline):
    """Run NJ with deadline guard. Returns Newick string or None if timeout."""
    start = time.time()
    enc = _encode_sequences(sequences, labels)
    D = _compute_distance_matrix(enc)
    elapsed = time.time() - start
    if elapsed > deadline - 5:
        return None
    return _neighbor_joining(D, labels, deadline=start + deadline - 2)


def reconstruct(sequences):
    """
    Reconstruct unrooted phylogenetic tree topology.
    Returns Newick string over labels t0..t(n-1).
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 3:
        if n == 1:
            return labels[0]
        elif n == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return f"({labels[0]},{labels[1]},{labels[2]})"

    # Encode once
    enc = _encode_sequences(sequences, labels)

    # Strategy: try quartet-based method first (more robust for JC data),
    # fall back to NJ if quartet census is too slow or yields few splits.
    result = None
    remaining = deadline - time.time()

    # Quartet-based approach: for small n, do full census; for larger n,
    # sample quartets or use all if time permits.
    # n=8: C(8,4)=70 quartets; n=12: 495; n=16: 1820; n=20: 4845;
    # n=24: 10626; n=28: 20475; n=32: 35960
    total_quartets = 0
    if n >= 4:
        # Compute total number of quartets
        from math import comb
        total_quartets = comb(n, 4)

    # Decide budget for quartet census based on n and time
    # Each quartet takes ~4 vector ops of length L. For L=2000, that's fast.
    # Estimate: ~10us per quartet for small L, ~50us for large L.
    # Be conservative: limit to what fits in remaining time.
    max_quartets = min(total_quartets, 200_000)
    quartet_deadline = time.time() + min(remaining * 0.6, 120.0)

    split_votes, q_done = _quartet_census(enc, n, max_quartets=max_quartets,
                                           deadline=quartet_deadline)

    if split_votes and q_done >= 10:
        # Build tree from splits
        build_deadline = time.time() + min(remaining * 0.3, 60.0)
        result = _build_tree_from_splits(labels, split_votes, deadline=build_deadline)

    # Fallback or refinement: if quartet result is None or we have time, run NJ
    if result is None or (time.time() < deadline - 10):
        nj_result = _nj_with_deadline(sequences, labels, deadline - start_time)
        if nj_result is not None:
            # If quartet gave a result, we could in principle combine, but for
            # simplicity, prefer quartet if it produced a valid tree with
            # reasonable split coverage. Otherwise use NJ.
            if result is None:
                result = nj_result
            # Keep quartet result if it's valid; NJ as backup
    if result is None:
        result = "(" + ",".join(labels) + ")"
    return result + ";"
# EVOLVE-BLOCK-END
