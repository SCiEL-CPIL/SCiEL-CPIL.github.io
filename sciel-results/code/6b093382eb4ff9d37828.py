"""Phylogenetic topology recovery from simulated sequences under Jukes-Cantor."""
# EVOLVE-BLOCK-START
import math
import time
from collections import Counter

_DEADLINE = time.time() + 260.0  # safety margin before 270s kill


def _jc_distance(p):
    """Jukes-Cantor corrected distance from proportion of differences."""
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distances using vectorized numpy."""
    import numpy as np

    n = len(labels)
    seq_len = len(sequences[labels[0]])
    # Build a matrix of characters: n x seq_len, with 0=A,1=C,2=G,3=T
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    mat = np.empty((n, seq_len), dtype=np.uint8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        # Vectorize character mapping
        mat[i] = np.frombuffer(s.encode(), dtype=np.uint8)
        # Convert ASCII to 0-3: A=65->0, C=67->1, G=71->2, T=84->3
        # Use lookup
        lookup = np.array([0]*65 + [1] + [0]*3 + [2] + [0]*12 + [3] + [0]*20, dtype=np.uint8)
        mat[i] = lookup[mat[i]]

    # Compute pairwise differences: for each pair, count mismatches
    # diff[i,j] = sum over sites where mat[i] != mat[j]
    # Vectorized: broadcast comparison
    # This is O(n^2 * L) but with numpy it's fast for n<=32, L<=2000
    diff = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        diff[i, i+1:] = np.sum(mat[i] != mat[i+1:], axis=1)
    diff = diff + diff.T

    # Convert to JC distances
    p = diff.astype(np.float64) / seq_len
    p = np.clip(p, 0, 0.7499)
    dist = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(dist, 0.0)
    return dist


def _nj_build(dist, labels):
    """Neighbour Joining on a distance matrix. Returns Newick string."""
    n = len(labels)
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"

    # Work with indices
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    # dist is n x n numpy array; we'll maintain a mutable copy
    D = dist.copy()

    while len(nodes) > 3:
        m = len(nodes)
        # Compute row sums
        rsum = np.array([sum(D[a, b] for b in nodes if b != a) for a in nodes])

        # Find best pair by NJ criterion: (m-2)*d(a,b) - rsum[a] - rsum[b]
        best_val = float('inf')
        best_a, best_b = -1, -1
        for i in range(m):
            for j in range(i + 1, m):
                a, b = nodes[i], nodes[j]
                crit = (m - 2) * D[a, b] - rsum[a] - rsum[b]
                if crit < best_val:
                    best_val = crit
                    best_a, best_b = a, b

        a, b = best_a, best_b

        # Compute branch lengths
        da = 0.5 * (D[a, b] + rsum[a] - rsum[b])
        db = D[a, b] - da

        # Build new node
        new_idx = max(nodes) + 1
        joined = f"({trees[a]}:{max(0.0, da):.4f},{trees[b]}:{max(0.0, db):.4f})"

        # Update distances to new node
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            D[new_idx, x] = D[x, new_idx] = 0.5 * (D[a, x] + D[b, x] - D[a, b])

        # Remove a, b; add new_idx
        nodes = rest + [new_idx]
        trees[new_idx] = joined

    # Final three nodes
    if len(nodes) == 3:
        i, j, k = nodes
        dij = D[i, j]
        dik = D[i, k]
        djk = D[j, k]
        bi = 0.5 * (dij + dik - djk)
        bj = 0.5 * (dij + djk - dik)
        bk = 0.5 * (dik + djk - dij)
        return f"(({trees[i]}:{max(0.0, bi):.4f},{trees[j]}:{max(0.0, bj):.4f}),{trees[k]}:{max(0.0, bk):.4f})"
    elif len(nodes) == 2:
        i, j = nodes
        return f"({trees[i]}:{D[i,j]/2:.4f},{trees[j]}:{D[i,j]/2:.4f})"
    else:
        return trees[nodes[0]]


def _bootstrap_resample(sequences, labels, rng):
    """Resample sites with replacement for bootstrap."""
    n_sites = len(sequences[labels[0]])
    indices = rng.integers(0, n_sites, size=n_sites)
    # Build resampled sequences
    resampled = {}
    for lab in labels:
        s = sequences[lab]
        resampled[lab] = ''.join(s[i] for i in indices)
    return resampled


def _parse_newick_splits(newick, labels):
    """Extract non-trivial unrooted splits from a Newick string."""
    # Remove branch lengths
    s = newick.replace(';', '')
    # Remove lengths
    import re
    s = re.sub(r':[\d.]+', '', s)
    # Remove spaces
    s = s.replace(' ', '')

    # Parse to get all internal splits
    # We'll do a recursive parse
    def parse(expr):
        """Return list of taxa in this subtree."""
        if expr.startswith('('):
            # Find matching close paren
            depth = 0
            end = -1
            for i, c in enumerate(expr):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            inner = expr[1:end]
            # Split inner by commas at depth 0
            parts = []
            d = 0
            start = 0
            for i, c in enumerate(inner):
                if c == '(':
                    d += 1
                elif c == ')':
                    d -= 1
                elif c == ',' and d == 0:
                    parts.append(inner[start:i])
                    start = i + 1
            parts.append(inner[start:])
            taxa = []
            for part in parts:
                if part.startswith('('):
                    taxa.extend(parse(part))
                else:
                    if part:
                        taxa.append(part)
            return taxa
        else:
            return [expr] if expr else []

    all_taxa = parse(s)
    n = len(all_taxa)

    # To get splits, we need to find each internal node and its subtree
    # Alternative: for each possible split, check if it's present
    # Better: recursively collect splits
    splits = set()

    def collect_splits(expr, complement):
        """expr is a subtree expression; complement is the set of taxa outside."""
        if expr.startswith('('):
            depth = 0
            end = -1
            for i, c in enumerate(expr):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            inner = expr[1:end]
            parts = []
            d = 0
            start = 0
            for i, c in enumerate(inner):
                if c == '(':
                    d += 1
                elif c == ')':
                    d -= 1
                elif c == ',' and d == 0:
                    parts.append(inner[start:i])
                    start = i + 1
            parts.append(inner[start:])

            subtree_taxa = []
            for part in parts:
                subtree_taxa.extend(parse(part))
                collect_splits(part, set(all_taxa) - set(subtree_taxa))

            # The split defined by this internal node
            if 1 < len(subtree_taxa) < n - 1:
                froz = frozenset(subtree_taxa)
                splits.add(froz)
        # leaves don't define splits

    collect_splits(s, set())
    return splits


def _rf_distance(splits1, splits2):
    """Compute normalized Robinson-Foulds distance between two sets of splits."""
    if not splits1 and not splits2:
        return 0.0
    common = splits1 & splits2
    # Non-trivial splits count
    n1 = len(splits1)
    n2 = len(splits2)
    if n1 + n2 == 0:
        return 0.0
    # RF distance = (n1 - common) + (n2 - common) = n1 + n2 - 2*common
    rf = (n1 - len(common)) + (n2 - len(common))
    max_rf = n1 + n2
    if max_rf == 0:
        return 0.0
    return rf / max_rf


def _evaluate_topologies(candidates, sequences, labels, true_splits_hint=None):
    """Evaluate candidate topologies by RF distance. Returns best Newick."""
    # Since we don't have the true tree, we use a heuristic:
    # For bootstrap consensus, we want the topology that appears most often
    # But we can also use the one with the lowest internal support inconsistency
    # For now, return the most frequent one
    return candidates[0]


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        # For 3 taxa, all topologies are equivalent (unrooted star)
        return f"({labels[0]},({labels[1]},{labels[2]}))"

    # Compute full distance matrix
    import numpy as np
    dist = _compute_distance_matrix(sequences, labels)

    # Build NJ tree from full data
    nj_tree = _nj_build(dist, labels)

    # Check time
    if time.time() > _DEADLINE - 30:
        return nj_tree

    # Bootstrap analysis
    rng = np.random.default_rng(42)
    n_sites = len(sequences[labels[0]])

    # Determine number of bootstrap replicates based on size and time
    # For small trees (n<=10), we can do more; for large, fewer
    if n <= 10:
        n_boot = 200
    elif n <= 20:
        n_boot = 100
    else:
        n_boot = 50

    # But we need to check time budget
    # Estimate: each NJ on n taxa takes roughly O(n^3) which for n=32 is ~32768 ops
    # With numpy distance computation being the bottleneck at O(n^2 * L)
    # Let's be conservative

    topology_count = Counter()
    # Store trees for consensus
    tree_list = []

    for boot in range(n_boot):
        if time.time() > _DEADLINE - 30:
            break

        resampled = _bootstrap_resample(sequences, labels, rng)
        boot_dist = _compute_distance_matrix(resampled, labels)
        boot_tree = _nj_build(boot_dist, labels)
        tree_list.append(boot_tree)
        topology_count[boot_tree] += 1

    # Also add the full-data NJ tree
    topology_count[nj_tree] += 1
    tree_list.append(nj_tree)

    if not topology_count:
        return nj_tree

    # Majority vote: pick the most common topology
    best_tree = topology_count.most_common(1)[0][0]

    # Check time for a polish step
    if time.time() > _DEADLINE - 15:
        return best_tree

    # Try to improve by checking if NJ on full data is better
    # We can't directly compute RF to true tree, but we can use
    # the fact that the full-data NJ is a strong candidate
    # Return the consensus (majority vote) tree
    return best_tree
# EVOLVE-BLOCK-END
