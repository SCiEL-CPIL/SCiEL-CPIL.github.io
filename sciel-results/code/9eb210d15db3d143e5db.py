"""Phylogenetic tree reconstruction: NJ init + ML NNI refinement under JC model."""
import time
import math
import numpy as np


def _jc_distance_matrix(aln_matrix):
    """Compute JC-corrected distance matrix from a binary alignment matrix."""
    n_taxa = aln_matrix.shape[0]
    n_sites = aln_matrix.shape[1]
    a = aln_matrix[:, None, :]
    b = aln_matrix[None, :, :]
    diff = (a != b).sum(axis=2)
    p = diff.astype(np.float64) / n_sites
    np.fill_diagonal(p, 0.0)
    p_clipped = np.clip(p, 0.0, 0.7499)
    arg = 1.0 - (4.0 / 3.0) * p_clipped
    dist = -0.75 * np.log(arg)
    np.fill_diagonal(dist, 0.0)
    return dist


def _nj_tree(aln_matrix, labels):
    """Build initial tree using Neighbor Joining."""
    n = len(labels)
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"

    D_init = _jc_distance_matrix(aln_matrix)

    max_nodes = 2 * n
    D = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    D[:n, :n] = D_init

    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    next_id = n

    while len(active) > 3:
        m = len(active)
        sub_D = D[np.ix_(active, active)]
        row_sums = sub_D.sum(axis=1)
        crit = (m - 2) * sub_D - row_sums[:, None] - row_sums[None, :]
        np.fill_diagonal(crit, np.inf)
        idx = np.unravel_index(crit.argmin(), crit.shape)
        i, j = idx
        a = active[i]
        b = active[j]

        joined = f"({trees[a]},{trees[b]})"
        new_node = next_id
        next_id += 1

        d_ab = D[a, b]
        for x in active:
            if x == a or x == b:
                continue
            D[new_node, x] = 0.5 * (D[a, x] + D[b, x] - d_ab)
            D[x, new_node] = D[new_node, x]

        active = [x for x in active if x not in (a, b)]
        active.append(new_node)
        trees[new_node] = joined

    if len(active) == 3:
        a, b, c = active
        return f"({trees[a]},{trees[b]},{trees[c]});"
    elif len(active) == 2:
        return f"({trees[active[0]]},{trees[active[1]]});"
    elif len(active) == 1:
        return trees[active[0]] + ";"
    else:
        parts = [trees[x] for x in active]
        return "(" + ",".join(parts) + ");"


def _parse_newick(newick):
    """Parse Newick into (nodes, root). nodes[i] = [children_list_or_None, label_or_None]."""
    newick = newick.strip().rstrip(';')
    nodes = []
    next_id = 0
    stack = []

    i = 0
    while i < len(newick):
        c = newick[i]
        if c == '(':
            new_id = next_id
            next_id += 1
            nodes.append([None, None])
            stack.append(new_id)
            i += 1
        elif c == ')':
            i += 1
            if i < len(newick) and newick[i] == ':':
                while i < len(newick) and newick[i] not in (',', ')', ';'):
                    i += 1
        elif c == ',':
            i += 1
        elif c == ':':
            while i < len(newick) and newick[i] not in (',', ')', ';'):
                i += 1
        else:
            label = ''
            while i < len(newick) and newick[i] not in (')', ',', ':', ';'):
                label += newick[i]
                i += 1
            new_id = next_id
            next_id += 1
            nodes.append([None, label])
            if stack:
                parent = stack[-1]
                if nodes[parent][0] is None:
                    nodes[parent][0] = [new_id, None]
                else:
                    nodes[parent][0][1] = new_id
            else:
                pass

    all_children = set()
    for node in nodes:
        if node[0] is not None:
            for child in node[0]:
                if child is not None:
                    all_children.add(child)

    root = -1
    for nid in range(len(nodes)):
        if nid not in all_children:
            root = nid
            break

    return nodes, root


def _tree_to_newick(nodes, root):
    """Convert tree structure to Newick string."""
    def _to_newick(idx):
        node = nodes[idx]
        if node[0] is None:
            return node[1] if node[1] is not None else ""
        children = node[0]
        parts = []
        for child in children:
            if child is not None:
                parts.append(_to_newick(child))
        return "(" + ",".join(parts) + ")"
    return _to_newick(root) + ";"


def _build_postorder(nodes, root):
    """Build postorder traversal list."""
    postorder = []
    visited = set()

    def _postorder(idx):
        if idx in visited:
            return
        visited.add(idx)
        node = nodes[idx]
        if node[0] is None:
            postorder.append(idx)
        else:
            for child in node[0]:
                if child is not None:
                    _postorder(child)
            postorder.append(idx)

    _postorder(root)
    return postorder


def _compute_jc_likelihood(nodes, root, aln_matrix, taxa_order, branch_lengths=None):
    """Compute JC log-likelihood of tree given alignment.
    taxa_order: list of taxon labels in order matching aln_matrix rows.
    branch_lengths: dict node_idx -> length (for edges to parent), or None for uniform.
    """
    n_sites = aln_matrix.shape[1]
    label_to_idx = {label: i for i, label in enumerate(taxa_order)}

    postorder = _build_postorder(nodes, root)

    # Precompute transition matrices for each branch length
    # For JC: P_same = 0.75 + 0.25*exp(-4d/3), P_diff = 0.25*(1-exp(-4d/3))

    # Get branch lengths for each edge (child -> parent)
    # We need to know which node is parent of which
    parent_map = {}
    for node_idx in range(len(nodes)):
        node = nodes[node_idx]
        if node[0] is not None:
            for child in node[0]:
                if child is not None:
                    parent_map[child] = node_idx

    if branch_lengths is None:
        branch_lengths = {}

    # Compute likelihood vectors
    likelihood = {}

    for node_idx in postorder:
        node = nodes[node_idx]
        if node[0] is None:
            # Leaf
            taxon = node[1]
            if taxon not in label_to_idx:
                continue
            taxon_idx = label_to_idx[taxon]
            seq = aln_matrix[taxon_idx]
            ll = np.zeros((n_sites, 4), dtype=np.float64)
            ll[np.arange(n_sites), seq] = 1.0
            likelihood[node_idx] = ll
        else:
            child_lls = []
            for child in node[0]:
                if child is not None and child in likelihood:
                    # Get branch length for edge child -> node_idx
                    bl = branch_lengths.get(child, 0.5)
                    p_same = 0.75 + 0.25 * math.exp(-4.0 * bl / 3.0)
                    p_diff = 0.25 * (1.0 - math.exp(-4.0 * bl / 3.0))

                    child_ll = likelihood[child]
                    # Transition: for each site, multiply child_ll by transition matrix
                    # trans[s,t] = P(obs_t | obs_s) = p_same if s==t else p_diff
                    # (child_ll @ trans.T)[site, t] = sum_s child_ll[site,s] * trans[s,t]
                    # = child_ll[site,t]*p_same + sum_{s!=t} child_ll[site,s]*p_diff
                    # = child_ll[site,t]*(p_same - p_diff) + p_diff * sum_s child_ll[site,s]
                    # = child_ll[site,t]*(0.5 + 0.5*exp(-4d/3)) + p_diff

                    exp_term = math.exp(-4.0 * bl / 3.0)
                    coef_same = 0.5 + 0.25 * exp_term  # p_same - p_diff
                    coef_diff = 0.25 * (1.0 - exp_term)  # p_diff

                    # temp[site, t] = child_ll[site, t] * coef_same + coef_diff * sum_s child_ll[site, s]
                    row_sums = child_ll.sum(axis=1, keepdims=True)
                    temp = child_ll * coef_same + coef_diff * row_sums
                    child_lls.append(temp)

            if len(child_lls) < 2:
                continue

            likelihood[node_idx] = child_lls[0] * child_lls[1]

    if root not in likelihood:
        return -np.inf

    root_ll = likelihood[root]
    # Sum over root states (uniform prior 1/4)
    total = root_ll.sum(axis=1) * 0.25
    total = np.maximum(total, 1e-300)
    log_likelihood = np.sum(np.log(total))

    return log_likelihood


def _optimize_branch_length(nodes, root, aln_matrix, taxa_order, child_idx, parent_idx, current_bl):
    """Optimize branch length for edge child_idx -> parent_idx using golden section search."""
    best_bl = current_bl
    best_ll = -np.inf

    # Golden section search in [0.001, 3.0]
    a = 0.001
    b = 3.0
    gr = (math.sqrt(5.0) - 1.0) / 2.0

    x1 = b - gr * (b - a)
    x2 = a + gr * (b - a)

    bls = {}
    for bl in [current_bl, x1, x2]:
        bls[child_idx] = bl
        ll = _compute_jc_likelihood(nodes, root, aln_matrix, taxa_order, bls)
        bls.clear()
        if ll > best_ll:
            best_ll = ll
            best_bl = bl

    for _ in range(15):
        bls = {child_idx: x1}
        ll1 = _compute_jc_likelihood(nodes, root, aln_matrix, taxa_order, bls)
        bls.clear()
        bls = {child_idx: x2}
        ll2 = _compute_jc_likelihood(nodes, root, aln_matrix, taxa_order, bls)
        bls.clear()

        if ll1 < ll2:
            a = x1
            x1 = x2
            x2 = a + gr * (b - a)
            bls = {child_idx: x2}
            ll2 = _compute_jc_likelihood(nodes, root, aln_matrix, taxa_order, bls)
            bls.clear()
            if ll2 > best_ll:
                best_ll = ll2
                best_bl = x2
        else:
            b = x2
            x2 = x1
            x1 = b - gr * (b - a)
            bls = {child_idx: x1}
            ll1 = _compute_jc_likelihood(nodes, root, aln_matrix, taxa_order, bls)
            bls.clear()
            if ll1 > best_ll:
                best_ll = ll1
                best_bl = x1

    return best_bl, best_ll


def _nni_refine(newick, aln_matrix, deadline, max_rounds=50):
    """Refine tree using NNI moves under JC model with branch length optimization."""
    nodes, root = _parse_newick(newick)
    taxa_order = sorted(
        [nodes[i][1] for i in range(len(nodes)) if nodes[i][1] is not None],
        key=lambda x: int(x[1:])
    )

    current_ll = _compute_jc_likelihood(nodes, root, aln_matrix, taxa_order)
    best_newick = newick
    best_ll = current_ll

    n_taxa = len(taxa_order)

    for round_num in range(max_rounds):
        if time.time() > deadline - 15.0:
            break

        improved = False

        # Build parent map
        parent_map = {}
        for node_idx in range(len(nodes)):
            node = nodes[node_idx]
            if node[0] is not None:
                for child in node[0]:
                    if child is not None:
                        parent_map[child] = node_idx

        # Get internal nodes (non-root)
        internal_nodes = [i for i in range(len(nodes)) if nodes[i][0] is not None and i != root]

        for parent_idx in internal_nodes:
            if time.time() > deadline - 15.0:
                break

            node = nodes[parent_idx]
            children = node[0]
            if children is None or children[0] is None or children[1] is None:
                continue

            c1, c2 = children[0], children[1]

            # Find grandparent
            grandparent = parent_map.get(parent_idx, -1)
            if grandparent == -1 or grandparent == root:
                continue

            gp_node = nodes[grandparent]
            gp_children = gp_node[0]
            if gp_children is None or gp_children[0] is None or gp_children[1] is None:
                continue

            if gp_children[0] == parent_idx:
                other_gp_child = gp_children[1]
            else:
                other_gp_child = gp_children[0]

            # NNI: swap c2 with other_gp_child
            # Original:
            #       grandparent
            #      /          \
            #  parent_idx   other_gp_child
            #    /    \
            #   c1     c2
            #
            # After NNI:
            #       grandparent
            #      /          \
            #  parent_idx   c2
            #    /    \
            #   c1   other_gp_child

            new_nodes = [list(n) for n in nodes]
            new_nodes[parent_idx][0] = [c1, other_gp_child]
            if gp_children[0] == parent_idx:
                new_nodes[grandparent][0] = [parent_idx, c2]
            else:
                new_nodes[grandparent][0] = [c2, parent_idx]

            new_ll = _compute_jc_likelihood(new_nodes, root, aln_matrix, taxa_order)

            if new_ll > best_ll + 1e-8:
                best_ll = new_ll
                nodes = new_nodes
                best_newick = _tree_to_newick(nodes, root)
                improved = True

        if not improved:
            break

    return best_newick


def _nj_initial_tree(aln_matrix, labels):
    """Build NJ tree and return as parsed structure."""
    newick = _nj_tree(aln_matrix, labels)
    return _parse_newick(newick)


def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 255.0  # 5s safety margin before 260s hard limit

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"

    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    aln_list = []
    for lab in labels:
        seq = sequences[lab]
        aln_list.append([base_map[c] for c in seq])
    aln_matrix = np.array(aln_list, dtype=np.int8)

    # Step 1: Build initial tree using NJ
    nj_newick = _nj_tree(aln_matrix, labels)

    if time.time() > deadline - 20.0:
        return nj_newick

    # Step 2: ML NNI refinement
    refined_newick = _nni_refine(nj_newick, aln_matrix, deadline, max_rounds=30)

    return refined_newick
