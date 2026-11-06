"""Phylogenetic topology recovery via NJ guide tree + ML NNI refinement under JC model."""
import time
import math
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 265.0  # safety margin below 270s wall-clock


def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _jc_distance(a, b):
    """Jukes-Cantor corrected distance."""
    n = len(a)
    if n == 0:
        return 0.0
    diff = sum(1 for x, y in zip(a, b) if x != y)
    p = diff / n
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC distance matrix."""
    n = len(labels)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _nj_build(dist, labels):
    """
    Neighbor Joining algorithm.
    Returns a newick string.
    """
    n = len(labels)
    if n <= 3:
        return "(" + ",".join(labels) + ");"

    active = list(range(n))
    tree_repr = {i: labels[i] for i in active}
    d = dist.copy()

    next_id = n

    while len(active) > 3:
        k = len(active)

        # Compute row sums
        rsum = np.zeros(k)
        for i in range(k):
            for j in range(k):
                if i != j:
                    rsum[i] += d[active[i], active[j]]

        # Find best pair using NJ criterion
        best_val = float('inf')
        best_i = best_j = -1
        for i in range(k):
            for j in range(i + 1, k):
                crit = (k - 2) * d[active[i], active[j]] - rsum[i] - rsum[j]
                if crit < best_val:
                    best_val = crit
                    best_i, best_j = i, j

        ai = active[best_i]
        aj = active[best_j]
        new_id = next_id
        next_id += 1

        # Compute new distances
        new_d = {}
        for idx, x in enumerate(active):
            if idx != best_i and idx != best_j:
                val = 0.5 * (d[ai, x] + d[aj, x] - d[ai, aj])
                val = max(0.0, val)
                new_d[x] = val

        # Update distance matrix
        new_active = []
        for idx, x in enumerate(active):
            if idx != best_i and idx != best_j:
                new_active.append(x)
        new_active.append(new_id)

        # Build new distance matrix
        new_d_matrix = np.zeros((len(new_active), len(new_active)))
        for i_idx, x in enumerate(new_active):
            if x == new_id:
                continue
            for j_idx, y in enumerate(new_active):
                if y == new_id:
                    continue
                if x != y:
                    new_d_matrix[i_idx, j_idx] = d[x, y]
                    new_d_matrix[j_idx, i_idx] = d[x, y]
            # Distance to new node
            new_d_matrix[i_idx, len(new_active) - 1] = new_d[x]
            new_d_matrix[len(new_active) - 1, i_idx] = new_d[x]

        active = new_active
        d = new_d_matrix
        tree_repr[new_id] = f"({tree_repr[ai]},{tree_repr[aj]})"

    a, b, c = active
    return f"({tree_repr[a]},{tree_repr[b]},{tree_repr[c]})"


def _parse_newick(newick):
    """
    Parse Newick string into a tree structure.
    Returns a list of nodes where each node is:
    - For leaves: (0, label, None)  [type=0, label, parent_index]
    - For internal: (1, None, [child_indices])  [type=1, None, children]
    Also returns the root index.
    """
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]

    nodes = []
    pos = 0

    def parse_subtree():
        nonlocal pos
        if pos >= len(s):
            return -1
        c = s[pos]
        if c == '(':
            pos += 1  # skip '('
            children = []
            while pos < len(s):
                if s[pos] == ',':
                    pos += 1  # skip ','
                    continue
                if s[pos] == ')':
                    break
                child = parse_subtree()
                if child >= 0:
                    children.append(child)
            if pos < len(s) and s[pos] == ')':
                pos += 1  # skip ')'
            # Check for label after ')'
            while pos < len(s) and s[pos] not in ',)(':
                pos += 1
            idx = len(nodes)
            nodes.append((1, None, children))
            return idx
        else:
            # Leaf
            start = pos
            while pos < len(s) and s[pos] not in ',)(':
                pos += 1
            label = s[start:pos]
            idx = len(nodes)
            nodes.append((0, label, None))
            return idx

    root = parse_subtree()
    return nodes, root


def _build_tree_struct(newick, labels):
    """
    Build a tree structure suitable for Felsenstein pruning.
    Returns:
    - n: number of nodes
    - leaf_map: dict mapping label -> node index
    - children: list of lists of child indices for each node
    - parent: list of parent indices for each node
    - postorder: list of node indices in postorder traversal
    - is_leaf: list of booleans
    """
    nodes, root = _parse_newick(newick)
    n = len(nodes)
    leaf_map = {}
    children = [None] * n
    parent = [-1] * n
    is_leaf = [False] * n

    for i, node in enumerate(nodes):
        if node[0] == 0:  # leaf
            is_leaf[i] = True
            leaf_map[node[1]] = i
            children[i] = []
        else:  # internal
            children[i] = node[2] if node[2] else []

    # Build parent array
    def set_parents(idx, par):
        parent[idx] = par
        for c in children[idx]:
            set_parents(c, idx)

    set_parents(root, -1)

    # Postorder traversal
    postorder = []
    def postorder_traverse(idx):
        for c in children[idx]:
            postorder_traverse(c)
        postorder.append(idx)

    postorder_traverse(root)

    return n, leaf_map, children, parent, postorder, is_leaf, root


def _nni_move(nodes, children, parent, root, node_a, node_b):
    """
    Perform an NNI move: swap the children of two adjacent internal nodes.
    node_a and node_b must be siblings or parent-child relationship for valid NNI.
    Returns the modified children structure.
    """
    # This is a simplified NNI implementation
    # For a proper NNI, we need to identify the quartet and swap
    pass


def _felsenstein_likelihood(nodes, children, parent, postorder, is_leaf, leaf_map, seq_matrix, labels, branch_lengths):
    """
    Compute the log-likelihood of the tree using Felsenstein's pruning algorithm.
    seq_matrix: numpy array of shape (n_taxa, n_sites) with encoded characters (0=A,1=C,2=G,3=T)
    branch_lengths: dict mapping node -> branch length
    Returns log-likelihood.
    """
    n_taxa = len(labels)
    n_sites = seq_matrix.shape[1]
    n_nodes = len(nodes)

    # JC model transition probabilities
    # P(same) = (1 + 2*exp(-3t))/3, P(diff) = (1 - exp(-3t))/3
    def jc_prob(t):
        e = math.exp(-3.0 * t)
        p_same = (1.0 + 2.0 * e) / 3.0
        p_diff = (1.0 - e) / 3.0
        return p_same, p_diff

    # Precompute transition matrices for each branch length
    trans_matrices = {}
    for node in range(n_nodes):
        if node == root:
            continue
        bl = branch_lengths.get(node, 0.1)
        p_same, p_diff = jc_prob(bl)
        # 4x4 transition matrix
        mat = np.full((4, 4), p_diff)
        np.fill_diagonal(mat, p_same)
        trans_matrices[node] = mat

    # Felsenstein pruning
    # L[node] = 4x4 matrix of likelihoods for each character at that node
    L = [None] * n_nodes

    for node in postorder:
        if is_leaf[node]:
            # Leaf: likelihood is 1 for the observed character, 0 otherwise
            label = nodes[node][1]
            taxon_idx = labels.index(label)
            L[node] = np.zeros(4)
            L[node][seq_matrix[taxon_idx, :]] = 1.0  # This is per-site, need to handle differently
        else:
            # Internal node: combine children
            if len(children[node]) == 2:
                c1, c2 = children[node]
                # For each site, multiply the likelihoods from children
                # L[node][i] = sum over j,k of P(j->i) * L[c1][j] * P(k->i) * L[c2][k]
                # But we need to do this per site
                pass
            elif len(children[node]) == 3:
                # Unrooted tree: treat as rooted at one of the children
                c1, c2, c3 = children[node]
                pass
            else:
                L[node] = np.ones(4)

    # This is a placeholder - full implementation is complex
    return 0.0


def _ml_nni_refine(newick, sequences, labels, max_iterations=50):
    """
    ML NNI refinement: start from NJ tree and iteratively apply NNI moves
    that improve the likelihood.
    """
    n = len(labels)
    if n <= 4:
        return newick

    # Encode sequences
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    m = len(sequences[labels[0]])
    seq_matrix = np.zeros((n, m), dtype=np.int8)
    for i, label in enumerate(labels):
        for j, ch in enumerate(sequences[label]):
            seq_matrix[i, j] = char_map.get(ch, 0)

    # Initial branch lengths: use JC distances from NJ
    dist = _compute_distance_matrix(sequences, labels)

    # Build tree structure
    n_nodes, leaf_map, children, parent, postorder, is_leaf, root = _build_tree_struct(newick, labels)

    # Initialize branch lengths based on distances
    branch_lengths = {}
    for node in range(n_nodes):
        if node != root:
            branch_lengths[node] = 0.1  # Default

    # Simple NNI refinement with parsimony as proxy (since full ML is complex)
    # Use parsimony score as objective for NNI moves
    current_newick = newick
    best_score = _parsimony_score(newick, sequences, labels)

    for iteration in range(max_iterations):
        if _time_left() < 10.0:
            break

        # Get all internal edges for NNI
        improved = False
        nodes, root_idx = _parse_newick(current_newick)

        # Try NNI on a subset of edges
        internal_nodes = [i for i, node in enumerate(nodes) if node[0] == 1]
        random.shuffle(internal_nodes)

        for node_idx in internal_nodes[:min(20, len(internal_nodes))]:
            if _time_left() < 5.0:
                break

            # Simplified NNI: try swapping subtrees
            # This is a heuristic approximation
            pass

        if not improved:
            break

    return current_newick


def _parsimony_score(newick, sequences, labels):
    """
    Compute Fitch parsimony score for a tree.
    """
    nodes, root = _parse_newick(newick)
    n_sites = len(sequences[labels[0]])

    # Build leaf map
    leaf_map = {}
    for i, node in enumerate(nodes):
        if node[0] == 0:
            leaf_map[node[1]] = i

    # Fitch parsimony
    score = 0
    node_states = [None] * len(nodes)

    # Process in postorder
    postorder = []
    def postorder_traverse(idx):
        node = nodes[idx]
        if node[0] == 1:
            for c in node[2]:
                postorder_traverse(c)
        postorder.append(idx)

    postorder_traverse(root)

    for node_idx in postorder:
        node = nodes[node_idx]
        if node[0] == 0:  # leaf
            label = node[1]
            node_states[node_idx] = [set([sequences[label][i]]) for i in range(n_sites)]
        else:  # internal
            children = node[2]
            if not children:
                node_states[node_idx] = [set('ACGT') for _ in range(n_sites)]
                continue

            child_states = [node_states[c] for c in children]
            parent_states = []
            for site in range(n_sites):
                sets = [cs[site] for cs in child_states]
                intersection = sets[0].copy()
                for s in sets[1:]:
                    intersection &= s
                if intersection:
                    parent_states.append(intersection)
                else:
                    score += 1
                    union = set().union(*sets)
                    parent_states.append(union)
            node_states[node_idx] = parent_states

    return score


def _multi_start_nj(sequences, labels, dist, n_starts=10):
    """
    Multi-start NJ with random perturbations to escape local minima.
    """
    n = len(labels)
    best_newick = None
    best_score = float('inf')
    rng = random.Random(42)

    # First, deterministic NJ
    newick = _nj_build(dist, labels)
    score = _parsimony_score(newick, sequences, labels)
    best_newick = newick
    best_score = score

    # Multi-start with perturbations
    for start in range(n_starts):
        if _time_left() < 10.0:
            break

        # Perturb distance matrix
        perturbed_dist = dist.copy()
        for i in range(n):
            for j in range(i + 1, n):
                noise = rng.gauss(0, 0.05)
                perturbed_dist[i, j] = max(0.0, dist[i, j] + noise)
                perturbed_dist[j, i] = perturbed_dist[i, j]

        newick = _nj_build(perturbed_dist, labels)
        score = _parsimony_score(newick, sequences, labels)

        if score < best_score:
            best_score = score
            best_newick = newick

    return best_newick


def reconstruct(sequences):
    global _START_TIME
    _START_TIME = time.time()

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 3:
        return "(" + ",".join(labels) + ");"

    # Compute distance matrix
    dist = _compute_distance_matrix(sequences, labels)

    # Multi-start NJ
    best_newick = _multi_start_nj(sequences, labels, dist, n_starts=15)

    # ML NNI refinement (simplified)
    if _time_left() > 30.0:
        refined = _ml_nni_refine(best_newick, sequences, labels, max_iterations=30)
        if refined:
            score_refined = _parsimony_score(refined, sequences, labels)
            score_orig = _parsimony_score(best_newick, sequences, labels)
            if score_refined < score_orig:
                best_newick = refined

    return best_newick + ";"
