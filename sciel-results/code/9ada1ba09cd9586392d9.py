# EVOLVE-BLOCK-START
import time
import math
import numpy as np
import sys
from itertools import combinations

_START = None
_DEADLINE_FRACTION = 0.82  # Safety margin


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


def _quartet_vote(dist, labels):
    """
    Build a tree using quartet-based voting.
    For each quartet (a,b,c,d), determine the most likely topology
    using the minimum sum of 3 pairwise distances (parsimony under JC).
    Then use a greedy/iterative algorithm to resolve the full tree.
    """
    n = len(labels)
    if n <= 3:
        if n == 1:
            return labels[0], set()
        elif n == 2:
            return f"({labels[0]},{labels[1]})", set()
        else:
            return f"({labels[0]},{labels[1]},{labels[2]})", set()

    # For each quartet, determine the best topology
    # Under JC, the best quartet topology is the one where the sum
    # of the 3 "internal" edge lengths is minimized, which corresponds
    # to the minimum of:
    #   d(a,b)+d(c,d), d(a,c)+d(b,d), d(a,d)+d(b,c)
    # The best topology pairs the two smallest of these three sums.

    # For efficiency with large n, we'll use a different approach:
    # Build the tree using a quartet-informed NJ variant.
    # Actually, let's use a proper quartet-based tree construction.

    # Strategy: Use the quartet signal to weight the NJ distance matrix
    # Or: Use a greedy quartet resolution algorithm

    # For n up to 32, O(n^4) quartets is at most ~32^4/24 ≈ 432k quartets
    # Each quartet evaluation is O(1), so this is feasible.

    # Let's use a different approach: build the tree by iteratively
    # adding taxa using quartet compatibility.

    # Actually, the most practical approach for n<=32:
    # 1. Compute all quartet votes
    # 2. Use a greedy algorithm to build a compatible tree

    # For the greedy algorithm, I'll use a variant of the "greedy quartet"
    # algorithm: start with a star tree, then repeatedly add quartet
    # constraints that are compatible with the current tree.

    # A simpler and more effective approach: use the quartet information
    # to create a weighted distance matrix, then run NJ.
    # The weight for each pair (i,j) is the number of quartets where
    # i and j are "close" (in the same clade).

    # Actually, let me just do a proper quartet-based tree construction:
    # Use the "greedy" algorithm where we start with all taxa as a star,
    # then for each quartet, if it's not compatible with the current tree,
    # modify the tree to accommodate it (prioritizing frequent quartets).

    # For simplicity and speed, let's use a hybrid:
    # 1. Compute quartet frequencies
    # 2. Use these to guide a modified NJ

    # Let me implement a proper quartet consensus approach:
    # Build the tree by finding the maximum weight compatible set of quartets.

    # For n<=32, let's use a simpler but effective method:
    # Weighted NJ where weights come from quartet support.

    # Compute quartet support for each pair of taxa
    # For each quartet (a,b,c,d), the best topology tells us which pairs are "siblings"
    # Count how many times each pair appears as siblings across all quartets

    pair_support = np.zeros((n, n), dtype=np.float64)

    if n <= 20:
        # Full quartet enumeration for small n
        for a, b, c, d in combinations(range(n), 4):
            if _time_left() < 5.0:
                break
            # Three possible topologies for quartet (a,b,c,d):
            # ((a,b),(c,d)): weight = d(a,b) + d(c,d)
            # ((a,c),(b,d)): weight = d(a,c) + d(b,d)
            # ((a,d),(b,c)): weight = d(a,d) + d(b,c)
            s1 = dist[a, b] + dist[c, d]
            s2 = dist[a, c] + dist[b, d]
            s3 = dist[a, d] + dist[b, c]

            # The best topology has the minimum sum
            min_s = min(s1, s2, s3)

            # Count support for the sibling pairs in the best topology
            # If multiple are tied, count all
            if s1 <= min_s + 1e-10:
                pair_support[a, b] += 1
                pair_support[b, a] += 1
                pair_support[c, d] += 1
                pair_support[d, c] += 1
            if s2 <= min_s + 1e-10:
                pair_support[a, c] += 1
                pair_support[c, a] += 1
                pair_support[b, d] += 1
                pair_support[d, b] += 1
            if s3 <= min_s + 1e-10:
                pair_support[a, d] += 1
                pair_support[d, a] += 1
                pair_support[b, c] += 1
                pair_support[c, b] += 1
    else:
        # For larger n, use a subset of quartets for speed
        # Sample quartets
        np.random.seed(42)
        num_quartets = min(50000, n * (n-1) * (n-2) * (n-3) // 24)
        for _ in range(num_quartets):
            if _time_left() < 5.0:
                break
            idx = np.random.choice(n, 4, replace=False)
            a, b, c, d = idx[0], idx[1], idx[2], idx[3]

            s1 = dist[a, b] + dist[c, d]
            s2 = dist[a, c] + dist[b, d]
            s3 = dist[a, d] + dist[b, c]

            min_s = min(s1, s2, s3)

            if s1 <= min_s + 1e-10:
                pair_support[a, b] += 1
                pair_support[b, a] += 1
                pair_support[c, d] += 1
                pair_support[d, c] += 1
            if s2 <= min_s + 1e-10:
                pair_support[a, c] += 1
                pair_support[c, a] += 1
                pair_support[b, d] += 1
                pair_support[d, b] += 1
            if s3 <= min_s + 1e-10:
                pair_support[a, d] += 1
                pair_support[d, a] += 1
                pair_support[b, c] += 1
                pair_support[c, b] += 1

    # Now use quartet-informed NJ
    # The quartet signal suggests that pairs with high support should be close
    # We can modify the distance matrix to incorporate this

    # Alternative: Just use the quartet information to select the best NJ tree
    # from bootstrap replicates

    return pair_support


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


def _parse_newick(newick_str):
    class Node:
        def __init__(self, name=None):
            self.name = name
            self.children = []
            self.parent = None

    root = None
    stack = []
    i = 0
    n = len(newick_str)

    while i < n:
        c = newick_str[i]
        if c == '(':
            new_node = Node()
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
                        float(part)
                    except ValueError:
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
                root = Node(name)
            i = j

    return root


def _get_leaves(node, leaves=None):
    if leaves is None:
        leaves = []
    if node is None:
        return leaves
    if not node.children:
        if node.name:
            leaves.append(node.name)
    else:
        for child in node.children:
            _get_leaves(child, leaves)
    return leaves


def _tree_splits(node, all_leaves):
    splits = set()
    n = len(all_leaves)

    def _recurse(node):
        if node is None or not node.children:
            return
        for child in node.children:
            child_leaves = frozenset(_get_leaves(child))
            if 1 < len(child_leaves) < n - 1:
                complement = frozenset(all_leaves - child_leaves)
                if tuple(sorted(child_leaves)) < tuple(sorted(complement)):
                    splits.add(child_leaves)
                else:
                    splits.add(complement)
            _recurse(child)

    _recurse(node)
    return splits


def _tree_to_newick_simple(node):
    if node is None:
        return ""
    if not node.children:
        return node.name if node.name else ""
    child_strs = [_tree_to_newick_simple(c) for c in node.children]
    return "(" + ",".join(child_strs) + ")"


def _nni_polish(tree_str, dist, labels, max_iterations=200):
    """
    Apply NNI moves to improve the tree topology using split-based scoring.
    """
    n = len(labels)
    if n <= 4:
        return tree_str

    root = _parse_newick(tree_str)
    if root is None:
        return tree_str

    all_leaves = set(labels)
    best_tree = tree_str
    best_splits = _tree_splits(root, all_leaves)
    best_score = len(best_splits)

    for iteration in range(max_iterations):
        if _time_left() < 2.0:
            break

        improved = False

        internal_nodes = []

        def _collect_internal(node, parent=None):
            if node is None:
                return
            if node.children:
                internal_nodes.append((node, parent))
                for child in node.children:
                    _collect_internal(child, node)

        _collect_internal(root)

        for node, parent in internal_nodes:
            if parent is None:
                continue
            if not node.children:
                continue
            if not parent.children:
                continue
            if len(node.children) < 2 or len(parent.children) < 2:
                continue

            node_children = [c for c in node.children if c != parent]
            if len(node_children) < 1:
                continue

            parent_children = [c for c in parent.children if c != node]
            if len(parent_children) < 1:
                continue

            for i in range(len(node_children)):
                for j in range(len(parent_children)):
                    if _time_left() < 1.0:
                        break

                    child_a = node_children[i]
                    child_b = parent_children[j]

                    # Perform the swap
                    node.children.remove(child_a)
                    child_a.parent = parent
                    parent.children.append(child_a)

                    parent.children.remove(child_b)
                    child_b.parent = node
                    node.children.append(child_b)

                    new_root = _parse_newick(_tree_to_newick_simple(root))
                    if new_root:
                        new_splits = _tree_splits(new_root, all_leaves)
                        new_score = len(new_splits)
                        common = len(new_splits & best_splits)

                        if new_score > best_score:
                            best_tree = _tree_to_newick_simple(root)
                            best_splits = new_splits
                            best_score = new_score
                            improved = True
                            root = _parse_newick(best_tree)
                            internal_nodes = []
                            _collect_internal(root)
                            break

                    # Revert the swap
                    parent.children.remove(child_a)
                    child_a.parent = node
                    node.children.append(child_a)

                    node.children.remove(child_b)
                    child_b.parent = parent
                    parent.children.append(child_b)

                    if improved:
                        break

                if improved:
                    break

            if improved:
                break

        if not improved:
            break

    return best_tree


def _quartet_based_tree(dist, labels):
    """
    Build a tree using quartet-based voting.
    For each quartet, determine the best topology using distance parsimony.
    Then use a greedy algorithm to build a compatible tree.
    """
    n = len(labels)
    if n <= 3:
        if n == 1:
            return labels[0], set()
        elif n == 2:
            return f"({labels[0]},{labels[1]})", set()
        else:
            return f"({labels[0]},{labels[1]},{labels[2]})", set()

    # For each quartet, determine the best topology
    # and count support for each split

    # A split is a partition of taxa into two non-empty sets
    # For quartet (a,b,c,d), the three possible splits are:
    #   {a,b}|{c,d}, {a,c}|{b,d}, {a,d}|{b,c}

    # Count support for each possible split
    # For n taxa, there are 2^(n-1) - 1 possible splits, but we only
    # need to consider those that appear in quartets.

    # For efficiency, let's use a different approach:
    # Build the tree using a quartet-informed NJ where we weight
    # the distance matrix based on quartet support.

    # Actually, let's implement a proper quartet-based tree construction:
    # Use the "greedy" algorithm:
    # 1. Start with all taxa as a star
    # 2. For each quartet (in order of support), try to add the
    #    corresponding split to the tree
    # 3. If the split is compatible with the current tree, add it
    # 4. Continue until all compatible splits are added

    # For n<=32, the number of quartets is at most ~432k, which is feasible.

    # Let's use a simpler but effective approach:
    # Use quartet voting to create a "consensus" distance matrix,
    # then run NJ on it.

    # The quartet signal suggests that if a and b are frequently
    # paired together in quartets, they should be close in the tree.
    # We can use this to adjust the distance matrix.

    # For each pair (i,j), compute the "quartet distance":
    # The average distance between i and j over all quartets where
    # i and j are in the same clade.

    # This is complex. Let's use a simpler approach:
    # Just run NJ on the original distance matrix, but use quartet
    # information to select the best tree from bootstrap replicates.

    # Actually, the most practical approach: use the quartet information
    # to weight the bootstrap consensus.

    return None  # Fall back to NJ + bootstrap


def reconstruct(sequences):
    global _START
    _START = time.time()

    labels, arr = _seqs_to_array(sequences)
    n = len(labels)

    d_full = _jc_matrix(arr)

    # Get quartet support matrix
    pair_support = _quartet_vote(d_full, labels)

    # Use quartet-informed NJ:
    # Modify the distance matrix to incorporate quartet support.
    # Pairs with high quartet support should have reduced distances.
    # This biases NJ toward quartet-supported topologies.

    # Normalize pair_support to [0, 1]
    max_support = pair_support.max() if pair_support.max() > 0 else 1.0
    norm_support = pair_support / max_support

    # Adjust distance matrix: reduce distance for well-supported pairs
    # d_adjusted[i,j] = d[i,j] * (1 - alpha * norm_support[i,j])
    # where alpha controls the strength of quartet influence
    alpha = 0.3  # Moderate quartet influence
    d_adjusted = d_full * (1.0 - alpha * norm_support)
    np.fill_diagonal(d_adjusted, 0)

    # Run NJ on the adjusted distance matrix
    tree_adj, splits_adj = _nj(d_adjusted, labels)

    # Also run NJ on the original distance matrix
    tree_orig, splits_orig = _nj(d_full, labels)

    candidates = [(tree_adj, splits_adj), (tree_orig, splits_orig)]
    split_counts = {}

    for s in splits_adj:
        split_counts[s] = split_counts.get(s, 0) + 1
    for s in splits_orig:
        split_counts[s] = split_counts.get(s, 0) + 1

    L = arr.shape[1]
    num_bootstraps = 0
    max_bootstraps = 500

    # Time budget: 270s total.
    while num_bootstraps < max_bootstraps and _time_left() > 15.0:
        indices = np.random.randint(0, L, L)
        arr_boot = arr[:, indices]
        d_boot = _jc_matrix(arr_boot)

        # Also adjust bootstrap distance with quartet support
        d_boot_adj = d_boot * (1.0 - alpha * norm_support)
        np.fill_diagonal(d_boot_adj, 0)

        # Run both NJ variants
        tree_boot_adj, splits_boot_adj = _nj(d_boot_adj, labels)
        tree_boot_orig, splits_boot_orig = _nj(d_boot, labels)

        candidates.append((tree_boot_adj, splits_boot_adj))
        candidates.append((tree_boot_orig, splits_boot_orig))

        for s in splits_boot_adj:
            split_counts[s] = split_counts.get(s, 0) + 1
        for s in splits_boot_orig:
            split_counts[s] = split_counts.get(s, 0) + 1

        num_bootstraps += 1

    if len(candidates) > 1:
        total_bootstraps = len(candidates)
        threshold = 0.4 * total_bootstraps

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
        best_tree = tree_adj

    # Apply NNI refinement if time permits
    if _time_left() > 10.0 and n >= 5:
        try:
            refined_tree = _nni_polish(best_tree, d_full, labels, max_iterations=100)
            if refined_tree:
                best_tree = refined_tree
        except Exception:
            pass

    return best_tree


# EVOLVE-BLOCK-END
