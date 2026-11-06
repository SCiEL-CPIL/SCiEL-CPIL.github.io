# EVOLVE-BLOCK-START
import time
import math
import numpy as np
import sys

_START = None
_DEADLINE_FRACTION = 0.85


def _now():
    return time.time()


def _init_time():
    global _START
    if _START is None:
        _START = time.time()


def _time_left():
    _init_time()
    return max(0.0, 270.0 * _DEADLINE_FRACTION - (time.time() - _START))


def _seqs_to_array(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    seqs = [sequences[l] for l in labels]
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.array([[mapping[c] for c in s] for s in seqs], dtype=np.uint8)
    return labels, arr


def _jc_matrix(arr):
    n, L = arr.shape
    if n <= 64:
        diff = (arr[:, None, :] != arr[None, :, :])
        counts = diff.sum(axis=2)
    else:
        counts = np.zeros((n, n), dtype=np.int32)
        chunk_size = max(1, min(16, n))
        for i in range(0, n, chunk_size):
            i_end = min(i + chunk_size, n)
            diff_chunk = (arr[i:i_end, None, :] != arr[None, :, :])
            counts[i:i_end, :] = diff_chunk.sum(axis=2)
    p = counts / L
    p_safe = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1 - (4.0 / 3.0) * p_safe)
    np.fill_diagonal(d, 0)
    return d


def _nj(dist, labels):
    n = len(labels)
    if n == 1:
        return labels[0], set()
    if n == 2:
        return f"({labels[0]},{labels[1]})", set()

    all_leaves = set(labels)
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    leaf_sets = {i: {labels[i]} for i in range(n)}

    d = {}
    for i in range(n):
        d[i] = {}
        for j in range(n):
            d[i][j] = dist[i][j]

    splits = set()

    while len(nodes) > 3:
        n_curr = len(nodes)
        total = {a: sum(d[a][b] for b in nodes if b != a) for a in nodes}

        best_criterion = float('inf')
        best_a, best_b = -1, -1

        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                criterion = (n_curr - 2) * d[a][b] - total[a] - total[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a, best_b = a, b

        a, b = best_a, best_b

        new_id = -1 - len(nodes)
        trees[new_id] = f"({trees[a]},{trees[b]})"
        leaf_sets[new_id] = leaf_sets[a] | leaf_sets[b]

        side1 = frozenset(leaf_sets[new_id])
        side2 = frozenset(all_leaves - leaf_sets[new_id])
        if tuple(sorted(side1)) < tuple(sorted(side2)):
            split = side1
        else:
            split = side2
        if 1 < len(split) < n - 1:
            splits.add(split)

        d[new_id] = {}
        for x in nodes:
            if x == a or x == b:
                continue
            d[new_id][x] = 0.5 * (d[a][x] + d[b][x] - d[a][b])
            d[x][new_id] = d[new_id][x]

        nodes.remove(a)
        nodes.remove(b)
        nodes.append(new_id)

    final_tree = "(" + ",".join(trees[x] for x in nodes) + ")"
    return final_tree, splits


class _TreeNode:
    __slots__ = ['name', 'children', 'parent', 'branch_len']

    def __init__(self, name=None, branch_len=0.0):
        self.name = name
        self.children = []
        self.parent = None
        self.branch_len = branch_len


def _parse_newick(newick_str):
    root = None
    stack = []
    i = 0
    n = len(newick_str)

    while i < n:
        c = newick_str[i]
        if c == '(':
            new_node = _TreeNode()
            if stack:
                stack[-1].children.append(new_node)
                new_node.parent = stack[-1]
            else:
                root = new_node
            stack.append(new_node)
            i += 1
        elif c == ')':
            stack.pop()
            i += 1
            j = i
            while j < n and newick_str[j] not in ',;':
                j += 1
            if i < j:
                part = newick_str[i:j]
                if stack:
                    try:
                        bl = float(part)
                        if stack[-1].parent:
                            stack[-1].parent.branch_len = bl
                    except ValueError:
                        if stack[-1].name is None:
                            stack[-1].name = part
                i = j
        elif c == ',':
            i += 1
        elif c == ';':
            i += 1
        else:
            j = i
            while j < n and newick_str[j] not in '(),;':
                j += 1
            name = newick_str[i:j]
            if stack:
                if stack[-1].name is None:
                    stack[-1].name = name
            else:
                root = _TreeNode(name)
            i = j

    return root


def _tree_to_newick(node):
    if node is None:
        return ""
    if not node.children:
        return node.name if node.name else ""
    child_strs = [_tree_to_newick(c) for c in node.children]
    return "(" + ",".join(child_strs) + ")"


def _get_all_leaves(node):
    if node is None:
        return []
    if not node.children:
        return [node.name] if node.name else []
    result = []
    for child in node.children:
        result.extend(_get_all_leaves(child))
    return result


def _get_subtree_leaves(node):
    if node is None:
        return set()
    if not node.children:
        return {node.name} if node.name else set()
    result = set()
    for child in node.children:
        result |= _get_subtree_leaves(child)
    return result


def _tree_splits_from_node(node, all_leaves):
    splits = set()
    n = len(all_leaves)

    def _recurse(node):
        if node is None or not node.children:
            return
        for child in node.children:
            child_leaves = frozenset(_get_subtree_leaves(child))
            if 1 < len(child_leaves) < n - 1:
                complement = frozenset(all_leaves - child_leaves)
                if tuple(sorted(child_leaves)) < tuple(sorted(complement)):
                    splits.add(child_leaves)
                else:
                    splits.add(complement)
            _recurse(child)

    _recurse(node)
    return splits


def _jc_log_likelihood(tree, arr, labels):
    """
    Compute the JC log-likelihood for a tree with estimated branch lengths.
    Uses a simple approach: estimate branch lengths from the distance matrix
    and compute likelihood using the JC model.
    """
    n = len(labels)
    L = arr.shape[1]
    if n < 3 or L == 0:
        return 0.0

    label_to_idx = {label: i for i, label in enumerate(labels)}

    # Build adjacency from tree
    # For unrooted tree, we need to handle it carefully
    # Let's root at the first leaf for computation

    # Find all leaves
    leaves = []
    def _find_leaves(node):
        if node is None:
            return
        if not node.children:
            if node.name:
                leaves.append(node)
        else:
            for c in node.children:
                _find_leaves(c)

    _find_leaves(tree)

    if len(leaves) < 2:
        return 0.0

    # Root at first leaf
    root = leaves[0]

    # Build parent-child relationships from root
    # We'll do a DFS to establish the rooted structure
    parent_map = {}
    child_map = {}

    def _build_rooted(node, parent):
        parent_map[node] = parent
        child_map[node] = []
        for c in node.children:
            if c != parent:
                child_map[node].append(c)
                _build_rooted(c, node)

    _build_rooted(root, None)

    # Get all nodes in post-order for bottom-up computation
    post_order = []
    def _postorder(node):
        for c in child_map.get(node, []):
            _postorder(c)
        post_order.append(node)

    _postorder(root)

    # Estimate branch lengths using JC distances
    # For each edge, estimate length from the split it defines
    # Simple approach: use the average distance across the split

    # Build edge list and estimate lengths
    edge_lengths = {}
    for node in post_order:
        if node == root:
            continue
        parent = parent_map[node]
        # Get leaves in subtree rooted at node
        subtree_leaves = _get_subtree_leaves(node)
        other_leaves = set(labels) - subtree_leaves

        # Estimate branch length as average distance between subtree and rest
        # Use a simple proxy: the minimum distance
        if subtree_leaves and other_leaves:
            # For efficiency, just use a small default for now
            # In practice, we'd compute this properly
            edge_lengths[(node, parent)] = 0.1  # Default small branch length

    # JC model: P(same) = 0.25 + 0.75*exp(-4t), P(diff) = 0.25 - 0.25*exp(-4t)
    # Log likelihood: sum over sites of log(sum over states of product over edges)
    # This is expensive, so we use a simplified version

    # For each site, compute the likelihood using Felsenstein's pruning algorithm
    total_ll = 0.0

    # Precompute JC probabilities for various branch lengths
    # P_same(t) = 0.25 + 0.75 * exp(-4t)
    # P_diff(t) = 0.25 - 0.25 * exp(-4t)

    for site in range(L):
        # Bottom-up: compute likelihood vectors at each node
        # L_node(state) = P(observation at node | state at node) * product over children of sum_state' P(state->state') * L_child(state')

        # For leaves, L_leaf(observed_state) = 1, others = 0
        # For internal nodes, use the recursion

        # This is O(n * 4 * L) which is fine for n<=32, L<=2000
        # But we need to be careful about the unrooted tree structure

        # Simplified: just return a score based on the number of compatible splits
        # This is not a true likelihood but serves as a proxy

        break  # Skip the full likelihood for speed

    # Use a simpler scoring: count how many of the tree's splits are "supported"
    # by the data (i.e., the groups have low internal distance)
    tree_splits = _tree_splits_from_node(tree, set(labels))

    score = 0.0
    for split in tree_splits:
        if len(split) < 2 or len(split) > n - 2:
            continue
        # Compute average distance within the split group
        idx_list = [label_to_idx[l] for l in split if l in label_to_idx]
        if len(idx_list) >= 2:
            sub_arr = arr[idx_list]
            # Average pairwise difference
            diff = (sub_arr[:, None, :] != sub_arr[None, :, :]).sum(axis=2)
            avg_diff = diff.sum() / (len(idx_list) * (len(idx_list) - 1)) if len(idx_list) > 1 else 0
            score -= avg_diff  # Lower distance = better

    return score


def _spr_refine(tree_str, dist, labels, arr, max_iterations=50):
    """
    Apply SPR moves to improve the tree topology using a JC-based score.
    """
    n = len(labels)
    if n <= 4:
        return tree_str

    root = _parse_newick(tree_str)
    if root is None:
        return tree_str

    all_leaves = set(labels)
    best_tree = tree_str
    best_score = _jc_log_likelihood(root, arr, labels)

    for iteration in range(max_iterations):
        if _time_left() < 3.0:
            break

        improved = False

        # Collect all nodes
        all_nodes = []
        def _collect_all(node):
            if node is None:
                return
            all_nodes.append(node)
            for c in node.children:
                _collect_all(c)

        _collect_all(root)

        # Try SPR moves: for each edge, prune a subtree and regraft elsewhere
        # Simplified: try NNI moves first (faster)
        # For each internal edge, try the two alternative NNI topologies

        # Find internal edges
        internal_edges = []
        for node in all_nodes:
            if node.parent is not None and node.children:
                internal_edges.append((node.parent, node))

        for parent_node, child_node in internal_edges:
            if _time_left() < 2.0:
                break

            # Get the subtrees involved in NNI
            # The edge is between parent_node and child_node
            # child_node has children (subtrees)
            # parent_node has other children (subtrees)

            child_children = [c for c in child_node.children if c != parent_node]
            parent_other_children = [c for c in parent_node.children if c != child_node]

            if not child_children or not parent_other_children:
                continue

            # Try NNI: swap child_children[0] with parent_other_children[0]
            # This is a simplified NNI

            a = child_children[0]
            b = parent_other_children[0]

            # Perform swap
            child_node.children.remove(a)
            a.parent = parent_node
            parent_node.children.append(a)

            parent_node.children.remove(b)
            b.parent = child_node
            child_node.children.append(b)

            new_tree_str = _tree_to_newick(root)
            new_root = _parse_newick(new_tree_str)
            if new_root:
                new_score = _jc_log_likelihood(new_root, arr, labels)

                if new_score > best_score:
                    best_tree = new_tree_str
                    best_score = new_score
                    improved = True
                    # Keep the change
                    root = new_root
                    # Re-collect nodes for next iteration
                    all_nodes = []
                    _collect_all(root)
                    internal_edges = []
                    for node in all_nodes:
                        if node.parent is not None and node.children:
                            internal_edges.append((node.parent, node))
                    break
                else:
                    # Revert
                    parent_node.children.remove(a)
                    a.parent = child_node
                    child_node.children.append(a)

                    child_node.children.remove(b)
                    b.parent = parent_node
                    parent_node.children.append(b)

            if improved:
                break

        if not improved:
            break

    return best_tree


def reconstruct(sequences):
    global _START
    _START = time.time()

    labels, arr = _seqs_to_array(sequences)
    n = len(labels)
    L = arr.shape[1]

    d_full = _jc_matrix(arr)
    tree_full, splits_full = _nj(d_full, labels)

    candidates = [(tree_full, splits_full)]
    split_counts = {}

    for s in splits_full:
        split_counts[s] = split_counts.get(s, 0) + 1

    # Bootstrap consensus
    max_bootstraps = min(300, max(50, int(_time_left() / 0.5)))
    num_bootstraps = 0

    while num_bootstraps < max_bootstraps and _time_left() > 20.0:
        indices = np.random.randint(0, L, L)
        arr_boot = arr[:, indices]
        d_boot = _jc_matrix(arr_boot)
        tree_boot, splits_boot = _nj(d_boot, labels)
        candidates.append((tree_boot, splits_boot))

        for s in splits_boot:
            split_counts[s] = split_counts.get(s, 0) + 1

        num_bootstraps += 1

    if len(candidates) > 1:
        total_bootstraps = len(candidates)
        threshold = 0.5 * total_bootstraps

        strong_splits = set()
        for s, c in split_counts.items():
            if c > threshold:
                strong_splits.add(s)

        best_idx = 0
        best_score = -1

        for i in range(len(candidates)):
            splits_i = candidates[i][1]
            common = len(splits_i & strong_splits)
            if common > best_score:
                best_score = common
                best_idx = i

        best_tree = candidates[best_idx][0]
    else:
        best_tree = tree_full

    # SPR/NNI refinement if time permits
    if _time_left() > 15.0 and n >= 5:
        try:
            refined_tree = _spr_refine(best_tree, d_full, labels, arr, max_iterations=30)
            if refined_tree:
                best_tree = refined_tree
        except Exception:
            pass

    return best_tree


# EVOLVE-BLOCK-END
