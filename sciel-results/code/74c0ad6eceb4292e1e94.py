"""
Phylogenetic reconstruction: fast quartet scoring + greedy compatible-split
consensus, with a clean Neighbor-Joining fallback and a strict deadline guard.

The score is the mean normalized Robinson-Foulds distance to the generating
trees (lower is better).  We keep the entry point `reconstruct(sequences) ->
Newick string` exactly as required.
"""
import math
import time
from collections import Counter
import numpy as np

DEADLINE_SECONDS = 265.0
SAFETY_MARGIN = 8.0


def _elapsed(start):
    return time.time() - start


def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor distances using NumPy."""
    n = len(labels)
    L = len(sequences[labels[0]])
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_array = np.zeros((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        for j in range(L):
            seq_array[i, j] = char_map.get(seq[j], 0)

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
    For every 4-taxon subset, pick the split with the smallest quartet score:
      (a,b)|(c,d):  d_ac + d_bd - d_ab - d_cd
      (a,c)|(b,d):  d_ad + d_bc - d_ab - d_cd
      (a,d)|(b,c):  d_ab + d_cd - d_ac - d_bd
    Returns a dict: frozenset of 4 taxa -> (side1, side2) where each side is a
    frozenset of two taxon indices.
    """
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

                    s1 = d_ac + d_bd - d_ab - d_cd
                    s2 = d_ad + d_bc - d_ab - d_cd
                    s3 = d_ab + d_cd - d_ac - d_bd

                    if s1 <= s2 and s1 <= s3:
                        win = (frozenset((a, b)), frozenset((c, d)))
                    elif s2 <= s1 and s2 <= s3:
                        win = (frozenset((a, c)), frozenset((b, d)))
                    else:
                        win = (frozenset((a, d)), frozenset((b, c)))

                    quartet_results[frozenset((a, b, c, d))] = win
    return quartet_results


def _build_consensus_tree(quartet_results, n, labels, dist_matrix):
    """
    Extend each winning quartet split to a full taxon split using a distance
    heuristic, then greedily select a maximal compatible set of splits, and
    recursively build the unrooted tree.
    """
    all_taxa = frozenset(range(n))
    splits = []

    for quartet, (side1, side2) in quartet_results.items():
        quartet_taxa = set(quartet)
        other_taxa = all_taxa - quartet_taxa

        full_side1 = set(side1)
        full_side2 = set(side2)

        for x in other_taxa:
            avg_d1 = sum(dist_matrix[x, y] for y in side1) / len(side1)
            avg_d2 = sum(dist_matrix[x, y] for y in side2) / len(side2)
            if avg_d1 <= avg_d2:
                full_side1.add(x)
            else:
                full_side2.add(x)

        # Canonicalize: keep the smaller side
        if len(full_side1) <= n // 2:
            split = frozenset(full_side1)
        else:
            split = frozenset(full_side2)

        if 2 <= len(split) <= n - 2:
            splits.append(split)

    if not splits:
        return "(" + ",".join(labels[t] for t in range(n)) + ");"

    unique_splits = list(set(splits))
    split_counts = Counter(splits)
    unique_splits.sort(key=lambda s: -split_counts[s])

    def compatible(s1, s2):
        s1c = all_taxa - s1
        s2c = all_taxa - s2
        # Incompatible iff all four intersections are non-empty (crossing)
        return not (s1 & s2 and s1 & s2c and s1c & s2 and s1c & s2c)

    selected = []
    for s in unique_splits:
        if all(compatible(s, t) for t in selected):
            selected.append(s)

    if not selected:
        return "(" + ",".join(labels[t] for t in range(n)) + ");"

    def build(taxa_set):
        if len(taxa_set) == 1:
            return labels[list(taxa_set)[0]]
        if len(taxa_set) == 2:
            a, b = sorted(taxa_set)
            return f"({labels[a]},{labels[b]})"

        best = None
        for s in selected:
            s_in = taxa_set & s
            s_out = taxa_set - s_in
            if 2 <= len(s_in) <= len(taxa_set) - 2:
                best = s_in
                break

        if best is None:
            return "(" + ",".join(labels[t] for t in sorted(taxa_set)) + ")"

        left = build(frozenset(best))
        right = build(frozenset(taxa_set - best))
        return f"({left},{right})"

    return build(all_taxa) + ";"


def _nj_dict(dist_matrix, labels):
    """Clean dict-based Neighbor-Joining fallback."""
    n = len(labels)
    if n <= 2:
        return "(" + ",".join(labels) + ");"

    dist = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[(labels[i], labels[j])] = dist_matrix[i, j]

    nodes = list(labels)
    trees = {name: name for name in labels}
    next_id = 0

    while len(nodes) > 3:
        cur = len(nodes)
        total = {}
        for a in nodes:
            total[a] = sum(dist[(a, b)] for b in nodes if b != a)

        best_c = float('inf')
        best_a = best_b = None
        best_d = 0.0

        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                d_ab = dist[(a, b)]
                c = (cur - 2) * d_ab - total[a] - total[b]
                if c < best_c - 1e-12:
                    best_c = c
                    best_a, best_b, best_d = a, b, d_ab
                elif abs(c - best_c) < 1e-12 and d_ab < best_d:
                    best_a, best_b, best_d = a, b, d_ab

        a, b = best_a, best_b
        d_ab = dist[(a, b)]
        new_label = f"n{next_id}"
        next_id += 1
        trees[new_label] = f"({trees[a]},{trees[b]})"
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            nd = 0.5 * (dist[(a, x)] + dist[(b, x)] - d_ab)
            dist[(new_label, x)] = nd
            dist[(x, new_label)] = nd
        nodes = rest + [new_label]

    return "(" + ",".join(trees[x] for x in nodes) + ");"


def reconstruct(sequences):
    start = time.time()
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 2:
        return "(" + ",".join(labels) + ");"

    dist_matrix = _jc_distance_matrix(sequences, labels)

    # Primary: quartet consensus
    if n >= 4 and _elapsed(start) < DEADLINE_SECONDS - SAFETY_MARGIN:
        try:
            qr = _quartet_scores(dist_matrix, n)
            tree = _build_consensus_tree(qr, n, labels, dist_matrix)
            if tree and len(tree) > 10:
                return tree
        except Exception:
            pass

    # Fallback: Neighbor-Joining
    if _elapsed(start) < DEADLINE_SECONDS - SAFETY_MARGIN:
        try:
            tree = _nj_dict(dist_matrix, labels)
            if tree:
                return tree
        except Exception:
            pass

    # Last resort: star tree
    return "(" + ",".join(labels) + ");"
