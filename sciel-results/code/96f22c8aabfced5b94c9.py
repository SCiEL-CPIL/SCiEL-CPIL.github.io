"""Phylogenetic topology recovery: NJ with JC69 + time-guarded local search."""
# EVOLVE-BLOCK-START
import time
import math
import random

_START = time.time()
_DEADLINE = 240.0  # wall-clock safety margin before the 270s kill


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE - _elapsed()


def _jc_distance(a, b):
    """Jukes-Cantor corrected pairwise distance."""
    L = len(a)
    diff = sum(1 for x, y in zip(a, b) if x != y)
    p = diff / L
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _nj_distance_matrix(labels, seqs):
    """Build the full JC69 distance matrix as a list-of-lists."""
    n = len(labels)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        ai = seqs[labels[i]]
        for j in range(i + 1, n):
            d = _jc_distance(ai, seqs[labels[j]])
            D[i][j] = d
            D[j][i] = d
    return D


def _newick_from_clusters(clusters):
    """Build a Newick string from the final 3 clusters (unrooted)."""
    parts = []
    for c in clusters:
        if c in _leaf_map:
            parts.append(_leaf_map[c])
        else:
            parts.append(c)
    return "(" + ",".join(parts) + ");"


_leaf_map = {}


def _nj_tree(labels, seqs):
    """Neighbour-Joining tree reconstruction. Returns Newick string."""
    global _leaf_map
    _leaf_map = {l: l for l in labels}

    n = len(labels)
    if n <= 2:
        return "(" + ",".join(labels) + ");"

    D = _nj_distance_matrix(labels, seqs)
    # active: list of node indices; node_ids: map index -> original label or cluster string
    active = list(range(n))
    # cluster_str: for each node index, the Newick substring representing it
    cluster_str = {i: labels[i] for i in range(n)}

    # r[i] = sum of distances from i to all active nodes
    r = [sum(D[i][j] for j in active if j != i) for i in active]

    while len(active) > 3:
        if _remaining() < 2.0:
            break

        m = len(active)
        # Compute q matrix and find min
        best_q = float('inf')
        bi, bj = -1, -1
        for ii, i in enumerate(active):
            for jj in range(ii + 1, m):
                j = active[jj]
                q = (m - 2) * D[i][j] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    bi, bj = i, j

        # Join bi and bj
        new_idx = max(active) + 1
        # Compute distances from new node to remaining active nodes
        for x in active:
            if x == bi or x == bj:
                continue
            D[new_idx][x] = 0.5 * (D[bi][x] + D[bj][x] - D[bi][bj])
            D[x][new_idx] = D[new_idx][x]
        D[new_idx][new_idx] = 0.0

        # Newick for the joined pair
        joined = "(" + cluster_str[bi] + "," + cluster_str[bj] + ")"
        cluster_str[new_idx] = joined

        # Update r for new node
        r_new = 0.0
        for x in active:
            if x == bi or x == bj:
                continue
            r_new += D[new_idx][x]
        r.append(r_new)

        # Remove bi, bj; add new_idx
        active.remove(bi)
        active.remove(bj)
        active.append(new_idx)

        # Update r for remaining active nodes
        new_r = []
        for x in active:
            if x == new_idx:
                new_r.append(r_new)
            else:
                new_r.append(r[x] - D[x][bi] - D[x][bj] + D[x][new_idx])
        # r list now has extra entries; rebuild cleanly
        r = new_r

    # Final 3 nodes
    if len(active) == 3:
        a, b, c = active
        # Unrooted tree: (a,b,c)
        return "(" + cluster_str[a] + "," + cluster_str[b] + "," + cluster_str[c] + ");"
    elif len(active) == 2:
        a, b = active
        return "(" + cluster_str[a] + "," + cluster_str[b] + ");"
    else:
        # Fallback: star tree
        return "(" + ",".join(cluster_str[x] for x in active) + ");"


def _parse_newick(newick):
    """Parse a Newick string into a set of splits (frozensets)."""
    newick = newick.strip().rstrip(';').strip()
    taxa = set()
    splits = set()

    def parse(s, pos):
        """Returns (node_name, new_pos). node_name is a string or a list of children."""
        if s[pos] == '(':
            children = []
            pos += 1
            while s[pos] != ')':
                child, pos = parse(s, pos)
                children.append(child)
                if s[pos] == ',':
                    pos += 1
            pos += 1  # skip ')'
            return children, pos
        else:
            # leaf name
            start = pos
            while pos < len(s) and s[pos] not in (',', ')', ';'):
                pos += 1
            name = s[start:pos].strip()
            if name:
                taxa.add(name)
            return name, pos

    root, _ = parse(newick, 0)

    # Collect splits by traversing the tree
    def collect_splits(node, parent_side=None):
        """For each internal node, the split is the set of leaves in its subtree vs the rest."""
        if isinstance(node, str):
            return {node}
        leaves = set()
        for child in node:
            leaves |= collect_splits(child)
        if len(leaves) > 1 and len(leaves) < len(taxa):
            splits.add(frozenset(leaves))
        return leaves

    collect_splits(root)
    return splits, taxa


def _rf_distance(splits1, splits2, n_taxa):
    """Normalized Robinson-Foulds distance."""
    s1 = set(splits1)
    s2 = set(splits2)
    common = s1 & s2
    total = len(s1) + len(s2)
    if total == 0:
        return 0.0
    return (total - 2 * len(common)) / total


def _star_tree(labels):
    """Return a star tree Newick string."""
    return "(" + ",".join(labels) + ");"


def reconstruct(sequences):
    """Reconstruct unrooted phylogenetic topology from aligned DNA sequences."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 2:
        return "(" + ",".join(labels) + ");"
    if n == 3:
        return "(" + ",".join(labels) + ");"

    # Initial NJ tree
    nj_newick = _nj_tree(labels, sequences)

    # Parse to get splits for scoring (internal use)
    try:
        nj_splits, _ = _parse_newick(nj_newick)
    except Exception:
        nj_splits = set()

    # If time is running out, return NJ immediately
    if _remaining() < 5.0:
        return nj_newick

    # Local search: try random 1-NNR (neighbor net) / 3-opt moves
    # For topology-only search on unrooted trees, we can do random cherry swaps
    # or use a simple approach: generate random alternative topologies and keep the best
    # by a quick heuristic (since we don't have the true tree, we use NJ as baseline
    # and try to improve via distance-based criteria).

    # A simpler and effective approach: run NJ with different random orderings
    # (randomized NJ) and pick the one with the lowest total branch length
    # (which correlates with fit under JC).

    best_newick = nj_newick
    best_score = float('inf')

    # Score NJ by total tree length (sum of positive branch lengths)
    # We'll use a proxy: the sum of all pairwise distances is fixed, so instead
    # we use the NJ criterion sum (lower is better)

    # Actually, for topology comparison without the true tree, the best we can do
    # is try multiple NJ variants and pick the one that seems most consistent.
    # Let's do randomized NJ: shuffle the order of joining by adding small noise.

    rng = random.Random(42)
    max_iters = 5  # number of randomized NJ restarts
    for it in range(max_iters):
        if _remaining() < 3.0:
            break

        # Add small random perturbation to distances to break ties differently
        D = _nj_distance_matrix(labels, sequences)
        # Perturb
        for i in range(n):
            for j in range(i + 1, n):
                D[i][j] += rng.uniform(-0.001, 0.001)
                D[j][i] = D[i][j]

        # Run NJ with perturbed distances
        global _leaf_map
        _leaf_map = {l: l for l in labels}
        active = list(range(n))
        cluster_str = {i: labels[i] for i in range(n)}
        r = [sum(D[i][j] for j in active if j != i) for i in active]

        while len(active) > 3:
            if _remaining() < 2.0:
                break
            m = len(active)
            best_q = float('inf')
            bi, bj = -1, -1
            for ii, i in enumerate(active):
                for jj in range(ii + 1, m):
                    j = active[jj]
                    q = (m - 2) * D[i][j] - r[i] - r[j]
                    if q < best_q:
                        best_q = q
                        bi, bj = i, j

            new_idx = max(active) + 1
            for x in active:
                if x == bi or x == bj:
                    continue
                D[new_idx][x] = 0.5 * (D[bi][x] + D[bj][x] - D[bi][bj])
                D[x][new_idx] = D[new_idx][x]
            D[new_idx][new_idx] = 0.0

            joined = "(" + cluster_str[bi] + "," + cluster_str[bj] + ")"
            cluster_str[new_idx] = joined

            r_new = sum(D[new_idx][x] for x in active if x not in (bi, bj))
            r.append(r_new)

            active.remove(bi)
            active.remove(bj)
            active.append(new_idx)

            new_r = []
            for x in active:
                if x == new_idx:
                    new_r.append(r_new)
                else:
                    new_r.append(r[x] - D[x][bi] - D[x][bj] + D[x][new_idx])
            r = new_r

        if len(active) == 3:
            a, b, c = active
            alt_newick = "(" + cluster_str[a] + "," + cluster_str[b] + "," + cluster_str[c] + ");"
        elif len(active) == 2:
            a, b = active
            alt_newick = "(" + cluster_str[a] + "," + cluster_str[b] + ");"
        else:
            alt_newick = "(" + ",".join(cluster_str[x] for x in active) + ");"

        # Compare topology: use RF distance to the original NJ as a proxy for diversity
        # Keep the original NJ (it's the most principled)
        # Actually, let's just keep NJ as the best answer

    return best_newick
# EVOLVE-BLOCK-END
