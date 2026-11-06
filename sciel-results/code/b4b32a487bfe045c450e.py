"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''NJ + quartet-score NNI refinement with deadline guard; small-n exact search.'''
import math
import time
import random
from itertools import combinations

import numpy as np

_DEADLINE = 260.0  # seconds; safety margin before 270 s kill
_START = None


def _elapsed():
    if _START is None:
        return 0.0
    return time.time() - _START


def _deadline_exceeded():
    return _elapsed() > _DEADLINE


def _jc_distance_matrix(labels, sequences):
    """Vectorized pairwise JC distance matrix using NumPy."""
    n = len(labels)
    L = len(sequences[labels[0]])
    enc = np.zeros((n, L), dtype=np.int8)
    for i, l in enumerate(labels):
        s = sequences[l]
        for j, c in enumerate(s):
            enc[i, j] = {'A': 0, 'C': 1, 'G': 2, 'T': 3}[c]
    diff = (enc[:, None, :] != enc[None, :, :])
    n_diff = diff.sum(axis=2)
    p = n_diff.astype(float) / L
    d = np.zeros((n, n))
    mask = p < 0.7499
    d[mask] = -0.75 * np.log(1.0 - 4.0 * p[mask] / 3.0)
    d[~mask] = 3.0
    np.fill_diagonal(d, 0.0)
    return d


def _nj_build(labels, dmat):
    """Standard neighbour-joining; returns Newick string."""
    n = len(labels)
    if n <= 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        a, b, c = labels
        return f"({a},({b},{c}))"

    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    d = dmat.copy()

    while len(nodes) > 3:
        m = len(nodes)
        r = np.zeros(m)
        for i, a in enumerate(nodes):
            r[i] = sum(d[a, b] for b in nodes if b != a)
        best_c = None
        best_pair = None
        for i, a in enumerate(nodes):
            for j in range(i + 1, m):
                b = nodes[j]
                c = (m - 2) * d[a, b] - r[i] - r[j]
                if best_c is None or c < best_c:
                    best_c = c
                    best_pair = (a, b)
        a, b = best_pair
        new_label = -1 - len(nodes)
        joined = f"({trees[a]},{trees[b]})"
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            new_dist = 0.5 * (d[a, x] + d[b, x] - d[a, b])
            d[new_label, x] = new_dist
            d[x, new_label] = new_dist
        nodes = rest + [new_label]
        trees[new_label] = joined

    result = "(" + ",".join(trees[x] for x in nodes) + ")"
    return result


def _get_splits(newick, n_labels):
    """Extract non-trivial unrooted splits from Newick string."""
    s = newick.rstrip(';').strip()
    n = n_labels
    splits = set()

    def parse(idx):
        if idx >= len(s):
            return frozenset(), idx
        if s[idx] == '(':
            idx += 1
            total = set()
            while idx < len(s) and s[idx] != ')':
                child, idx = parse(idx)
                total |= child
                if idx < len(s) and s[idx] == ',':
                    idx += 1
            if idx < len(s) and s[idx] == ')':
                idx += 1
            while idx < len(s) and s[idx] not in (',', ')', ';'):
                idx += 1
            if 1 < len(total) < n:
                splits.add(frozenset(total))
            return frozenset(total), idx
        else:
            j = idx
            while j < len(s) and s[j] not in (',', ')', ';'):
                j += 1
            name = s[idx:j]
            while j < len(s) and s[j] not in (',', ')', ';'):
                j += 1
            return frozenset([name]), j

    parse(0)
    return splits


def _quartet_score_fast(newick, labels, dmat, idx_map):
    """Compute total quartet score for a tree (higher = better)."""
    n = len(labels)
    if n < 4:
        return 0.0
    splits = _get_splits(newick, n)
    all_leaves = frozenset(labels)
    score = 0.0

    for a, b, c, d in combinations(labels, 4):
        ia, ib, ic, id = idx_map[a], idx_map[b], idx_map[c], idx_map[d]
        s1 = dmat[ia, ic] + dmat[ib, id] - dmat[ia, id] - dmat[ib, ic]
        s2 = dmat[ia, ib] + dmat[ic, id] - dmat[ia, id] - dmat[ib, ic]
        s3 = dmat[ia, ib] + dmat[ic, id] - dmat[ia, ic] - dmat[ib, id]

        # Determine which topology the tree implies
        implies_ab_cd = False
        implies_ac_bd = False
        implies_ad_bc = False

        for sp in splits:
            comp = all_leaves - sp
            if a in sp and b in sp and c in comp and d in comp:
                implies_ab_cd = True
            elif c in sp and d in sp and a in comp and b in comp:
                implies_ab_cd = True
            elif a in sp and c in sp and b in comp and d in comp:
                implies_ac_bd = True
            elif b in sp and d in sp and a in comp and c in comp:
                implies_ac_bd = True
            elif a in sp and d in sp and b in comp and c in comp:
                implies_ad_bc = True
            elif b in sp and c in sp and a in comp and d in comp:
                implies_ad_bc = True

        if implies_ab_cd:
            score += -s1
        elif implies_ac_bd:
            score += -s2
        elif implies_ad_bc:
            score += -s3
        else:
            score += -min(s1, s2, s3)

    return score


def _build_tree(newick, labels):
    """Build a tree structure from Newick. Returns (root_id, nodes_dict)."""
    s = newick.rstrip(';').strip()
    nodes = {}
    counter = [0]

    def new_node(name, children):
        idx = counter[0]
        counter[0] += 1
        nodes[idx] = (name, children)
        return idx

    def parse(idx):
        if idx >= len(s):
            return None
        if s[idx] == '(':
            idx += 1
            children = []
            while idx < len(s) and s[idx] != ')':
                child = parse(idx)
                if child is not None:
                    children.append(child)
                if idx < len(s) and s[idx] == ',':
                    idx += 1
            if idx < len(s) and s[idx] == ')':
                idx += 1
            while idx < len(s) and s[idx] not in (',', ')', ';'):
                idx += 1
            return new_node('', children)
        else:
            j = idx
            while j < len(s) and s[j] not in (',', ')', ';'):
                j += 1
            name = s[idx:j]
            while j < len(s) and s[j] not in (',', ')', ';'):
                j += 1
            return new_node(name, [])

    root = parse(0)
    if root is None:
        return None, None
    return root, nodes


def _get_internal_edges(nodes):
    """Get list of (parent, child) where child is internal (has children)."""
    edges = []

    def is_leaf(node_id):
        return len(nodes[node_id][1]) == 0

    def walk(node_id, parent_id):
        if parent_id is not None and not is_leaf(node_id):
            edges.append((parent_id, node_id))
        for child in nodes[node_id][1]:
            walk(child, node_id)

    return edges


def _nni_refine(newick, labels, dmat, idx_map, max_rounds=5):
    """NNI refinement using quartet-based scoring."""
    if _deadline_exceeded():
        return newick

    n = len(labels)
    if n < 4:
        return newick

    best_newick = newick
    best_score = _quartet_score_fast(newick, labels, dmat, idx_map)

    root, nodes = _build_tree(newick, labels)
    if root is None:
        return best_newick

    rng = random.Random(42)

    for rnd in range(max_rounds):
        if _deadline_exceeded():
            break
        improved = False
        internal_edges = _get_internal_edges(nodes)
        rng.shuffle(internal_edges)

        for edge in internal_edges:
            if _deadline_exceeded():
                break
            parent, child = edge
            parent_children = nodes[parent][1]
            child_children = nodes[child][1]

            if len(parent_children) != 3 or len(child_children) != 2:
                continue

            siblings = [c for c in parent_children if c != child]
            if len(siblings) != 2:
                continue
            A, B = siblings[0], siblings[1]
            C1, C2 = child_children

            def subtree_newick(node_id):
                name, children = nodes[node_id]
                if not children:
                    return name
                return "(" + ",".join(subtree_newick(c) for c in children) + ")"

            def make_newick(structure):
                if isinstance(structure, int):
                    return subtree_newick(structure)
                return "(" + make_newick(structure[0]) + "," + make_newick(structure[1]) + ")"

            # NNI 1: ((A, C1), (B, C2))
            newick1 = make_newick(((A, C1), (B, C2)))
            cand_score = _quartet_score_fast(newick1, labels, dmat, idx_map)
            if cand_score > best_score:
                best_score = cand_score
                best_newick = newick1
                root, nodes = _build_tree(newick1, labels)
                if root is None:
                    return best_newick
                improved = True
                break

            # NNI 2: ((A, C2), (B, C1))
            newick2 = make_newick(((A, C2), (B, C1)))
            cand_score = _quartet_score_fast(newick2, labels, dmat, idx_map)
            if cand_score > best_score:
                best_score = cand_score
                best_newick = newick2
                root, nodes = _build_tree(newick2, labels)
                if root is None:
                    return best_newick
                improved = True
                break

        if not improved:
            break

    return best_newick


def _small_n_exact(labels, dmat, idx_map):
    """Exhaustive search over all unrooted topologies for n=4."""
    n = len(labels)
    if n != 4:
        return None

    a, b, c, d = labels
    topologies = [
        f"(({a},{b}),({c},{d}))",
        f"(({a},{c}),({b},{d}))",
        f"(({a},{d}),({b},{c}))",
    ]

    best_score = -float('inf')
    best_newick = None
    for newick in topologies:
        score = _quartet_score_fast(newick, labels, dmat, idx_map)
        if score > best_score:
            best_score = score
            best_newick = newick

    return best_newick


def reconstruct(sequences):
    global _START
    _START = time.time()

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"

    dmat = _jc_distance_matrix(labels, sequences)
    idx_map = {l: i for i, l in enumerate(labels)}

    if _deadline_exceeded():
        return "(" + ",".join(labels) + ");"

    # Small-n exact search for n=4
    if n == 4:
        result = _small_n_exact(labels, dmat, idx_map)
        if result:
            return result + ";"

    # NJ scaffold
    newick = _nj_build(labels, dmat)

    if _deadline_exceeded():
        return newick + ";"

    # NNI refinement
    refined = _nni_refine(newick, labels, dmat, idx_map, max_rounds=5)

    return refined + ";"
# EVOLVE-BLOCK-END
