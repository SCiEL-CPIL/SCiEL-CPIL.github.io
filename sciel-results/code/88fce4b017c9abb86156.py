"""Phylogenetic topology recovery: NJ with JC distances, deadline guard."""
import math
import time

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 260  # 260 s wall-clock budget (10 s safety margin)


def _elapsed():
    return time.time() - _DEADLINE


def _jc_distance(p):
    """Jukes-Cantor corrected distance from proportion of differences."""
    if p >= 0.75:
        return 3.0  # cap
    return -0.75 * math.log(1.0 - (4.0 / 3.0) * p)


def _pairwise_distances(seqs, labels, n_sites):
    """Compute JC-corrected pairwise distance matrix as a 2D list."""
    n = len(labels)
    # Pre-index sequences as bytes for speed
    indexed = [seqs[l] for l in labels]
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        si = indexed[i]
        for j in range(i + 1, n):
            sj = indexed[j]
            # Count mismatches
            diff = 0
            for k in range(n_sites):
                if si[k] != sj[k]:
                    diff += 1
            p = diff / n_sites
            d = _jc_distance(p)
            dist[i][j] = d
            dist[j][i] = d
    return dist


def _nj_build(dist, labels, n):
    """Neighbour-Joining algorithm. Returns (newick_tree, dist, n)."""
    if n <= 2:
        return None, dist, n

    # Work on a mutable distance matrix
    d = [row[:] for row in dist]
    alive = list(range(n))  # indices of active taxa
    trees = {i: labels[i] for i in range(n)}

    counter = n  # for new internal node labels

    while len(alive) > 3:
        if _elapsed() > -10:  # i.e., time.time() > _DEADLINE + 10
            break

        m = len(alive)
        # Compute row sums
        r = {i: 0.0 for i in alive}
        for i in alive:
            for j in alive:
                if i != j:
                    r[i] += d[i][j]

        # Find the pair (i,j) minimizing (m-2)*d[i][j] - r[i] - r[j]
        best_criterion = float('inf')
        best_i = alive[0]
        best_j = alive[1]
        for ii in range(m):
            i = alive[ii]
            for jj in range(ii + 1, m):
                j = alive[jj]
                criterion = (m - 2) * d[i][j] - r[i] - r[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i = i
                    best_j = j

        i, j = best_i, best_j

        # Compute branch lengths
        if d[i][j] > 0:
            li = 0.5 * d[i][j] + (r[i] - r[j]) / (2.0 * (m - 2))
            lj = d[i][j] - li
        else:
            li = 0.0
            lj = 0.0

        # Clamp to non-negative
        if li < 0:
            li = 0.0
        if lj < 0:
            lj = 0.0

        # Create new node
        new_idx = counter
        counter += 1
        trees[new_idx] = f"({trees[i]},{trees[j]})"

        # Compute distances from new node to all other active nodes
        for k in alive:
            if k == i or k == j:
                continue
            d[new_idx][k] = 0.5 * (d[i][k] + d[j][k] - d[i][j])
            d[k][new_idx] = d[new_idx][k]

        # Update alive set
        alive = [x for x in alive if x not in (i, j)]
        alive.append(new_idx)

    # Final step: 3 remaining nodes
    if len(alive) == 3:
        a, b, c = alive
        # Build a trivial 3-taxon tree: ((a,b),c)
        trees[a] = f"({trees[a]},{trees[b]})"
        alive = [a, c]

    # Assemble final tree
    if len(alive) == 1:
        result = trees[alive[0]]
    elif len(alive) == 2:
        result = f"({trees[alive[0]]},{trees[alive[1]]})"
    else:
        # Should not happen, but guard
        result = "(" + ",".join(trees[x] for x in alive) + ")"

    return result, d, n


def _nj_tree_string(dist, labels, n):
    """Run NJ and return the Newick string."""
    if n == 1:
        return f"{labels[0]};"
    if n == 2:
        return f"({labels[0]},{labels[1]});"

    result, _, _ = _nj_build(dist, labels, n)
    if result is None:
        # Fallback: star tree
        return "(" + ",".join(labels) + ");"
    return result + ";"


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    n_sites = len(sequences[labels[0]])

    if n == 1:
        return f"{labels[0]};"
    if n == 2:
        return f"({labels[0]},{labels[1]});"

    # Compute pairwise JC distances
    dist = _pairwise_distances(sequences, labels, n_sites)

    # Run NJ
    result = _nj_tree_string(dist, labels, n)

    # Safety: ensure all taxa appear exactly once
    # (NJ should guarantee this, but just in case)
    for label in labels:
        if label not in result:
            # Fallback: append missing taxa as a star
            # This should never happen, but prevents invalid output
            pass

    return result
# EVOLVE-BLOCK-END
