"""Quartet-scored Neighbor-Joining with NNI refinement and small-n exhaustive search."""
import math
import time
import itertools
import numpy as np

_START_TIME = None

def _check_time(deadline=255.0):
    """Return True if we're still within the time budget."""
    global _START_TIME
    if _START_TIME is None:
        return True
    return (time.time() - _START_TIME) < deadline


def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distances using NumPy."""
    n = len(labels)
    if n == 0:
        return np.zeros((0, 0))
    L = len(sequences[labels[0]])
    if L == 0:
        return np.zeros((n, n))

    char_map = np.array([{'A': 0, 'C': 1, 'G': 2, 'T': 3}[c] for c in 'ACGT'], dtype=np.int8)
    # Map characters to 0-3
    lookup = np.zeros(256, dtype=np.int8)
    lookup[ord('A')] = 0
    lookup[ord('C')] = 1
    lookup[ord('G')] = 2
    lookup[ord('T')] = 3

    numeric = np.zeros((n, L), dtype=np.int8)
    for i, label in enumerate(labels):
        seq = sequences[label].encode('ascii')
        numeric[i] = lookup[seq]

    # Pairwise mismatches via broadcasting (memory-efficient for reasonable sizes)
    if n * L <= 200000:  # For large cases, do it in chunks
        diff = (numeric[:, np.newaxis, :] != numeric[np.newaxis, :, :])
        mismatches = diff.sum(axis=2).astype(np.float64)
    else:
        # Chunked computation
        mismatches = np.zeros((n, n), dtype=np.float64)
        chunk = max(1, 200000 // n)
        for i in range(n):
            for j in range(i + 1, n):
                mismatches[i, j] = np.count_nonzero(numeric[i] != numeric[j])
                mismatches[j, i] = mismatches[i, j]

    p = mismatches / L
    p = np.clip(p, 0.0, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(d, 0.0)
    return d


def _nj_build_tree(d, labels):
    """Neighbour Joining algorithm. Returns (newick, splits).
    splits is a list of frozensets representing internal node leaf sets."""
    n = len(labels)
    if n <= 1:
        return (labels[0] if n == 1 else ""), []
    if n == 2:
        return f"({labels[0]},{labels[1]})", []

    dist = {}
    for i in range(n):
        for j in range(n):
            dist[(i, j)] = d[i, j]
            dist[(j, i)] = d[i, j]

    nodes = list(range(n))
    tree_parts = {i: labels[i] for i in range(n)}
    node_leaves = {i: frozenset({i}) for i in range(n)}
    next_node_id = n
    splits = []

    while len(nodes) > 3:
        m = len(nodes)
        row_sums = {node: 0.0 for node in nodes}
        for a in nodes:
            for b in nodes:
                if a != b:
                    row_sums[a] += dist[(a, b)]

        best_criterion = float('inf')
        best_a = best_b = None
        for i in range(m):
            for j in range(i + 1, m):
                a, b = nodes[i], nodes[j]
                criterion = (m - 2) * dist[(a, b)] - row_sums[a] - row_sums[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a, best_b = a, b

        a, b = best_a, best_b
        d_ab = dist[(a, b)]

        r_a = (row_sums[a] - row_sums[b] + d_ab) / (2.0 * m)
        r_b = d_ab - r_a
        r_a = max(0.0, r_a)
        r_b = max(0.0, r_b)

        new_node = next_node_id
        next_node_id += 1

        new_leaves = node_leaves[a] | node_leaves[b]
        if 0 < len(new_leaves) < n:
            splits.append(new_leaves)

        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            new_d = 0.5 * (dist[(a, x)] + dist[(b, x)] - d_ab)
            new_d = max(0.0, new_d)
            dist[(new_node, x)] = new_d
            dist[(x, new_node)] = new_d

        nodes = rest + [new_node]
        tree_parts[new_node] = f"({tree_parts[a]}:{r_a:.6f},{tree_parts[b]}:{r_b:.6f})"
        node_leaves[new_node] = new_leaves

    if len(nodes) == 3:
        a, b, c = nodes
        d_ab = dist[(a, b)]
        d_ac = dist[(a, c)]
        d_bc = dist[(b, c)]
        len_a = max(0.0, (d_ab + d_ac - d_bc) / 2.0)
        len_b = max(0.0, (d_ab + d_bc - d_ac) / 2.0)
        len_c = max(0.0, (d_ac + d_bc - d_ab) / 2.0)
        newick = f"({tree_parts[a]}:{len_a:.6f},{tree_parts[b]}:{len_b:.6f},{tree_parts[c]}:{len_c:.6f})"
    else:
        newick = "(" + ",".join(tree_parts[x] for x in nodes) + ")"

    return newick, splits


def _build_newick_from_splits(splits, n, labels):
    """Build a Newick string from a set of compatible splits.
    Each split is a frozenset of leaf indices. Splits must be compatible."""
    if n <= 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    # Sort splits by size (smaller first)
    sorted_splits = sorted(splits, key=len)

    # Build a hierarchy: repeatedly merge pairs of groups that form a split
    groups = {i: frozenset({i}) for i in range(n)}
    # Map from frozenset to group id (use the frozenset itself as key)
    # We'll build a tree by repeatedly finding cherries

    # Alternative: build using the split structure
    # For each split, create an internal node connecting the two sides

    # Use a simpler approach: recursive partition
    all_leaves = frozenset(range(n))

    def _build_recursive(leaf_set, available_splits):
        """Recursively build newick for the given leaf set using available splits."""
        ls = frozenset(leaf_set)
        if len(ls) == 1:
            return labels[list(ls)[0]]
        if len(ls) == 2:
            a, b = sorted(ls)
            return f"({labels[a]},{labels[b]})"
        if len(ls) == 3:
            a, b, c = sorted(ls)
            return f"({labels[a]},{labels[b]},{labels[c]})"

        # Find a split that partitions ls into two non-empty parts
        best_split = None
        for s in available_splits:
            left = s & ls
            right = ls - s
            if len(left) > 0 and len(right) > 0:
                if len(left) <= len(right):
                    best_split = s
                    break
                else:
                    # Check the complement
                    comp = all_leaves - s
                    left2 = comp & ls
                    right2 = ls - comp
                    if len(left2) > 0 and len(right2) > 0:
                        if len(left2) <= len(left) or best_split is None:
                            best_split = comp
                            break
                    else:
                        # s itself works but right is larger
                        if best_split is None:
                            best_split = s

        if best_split is None:
            # No valid split, make a star tree
            return "(" + ",".join(labels[i] for i in sorted(ls)) + ")"

        left = best_split & ls
        right = ls - best_split

        # Filter available splits for each side
        splits_left = [s for s in available_splits if (s & ls) and (s & ls) < ls]
        # Recurse
        newick_left = _build_recursive(left, available_splits)
        newick_right = _build_recursive(right, available_splits)

        if len(left) == 1 and len(right) == 1:
            return f"({newick_left},{newick_right})"
        return f"({newick_left},{newick_right})"

    return _build_recursive(all_leaves, list(splits))


def _nni_refine(splits, d, n, labels, max_passes=5, deadline=255.0):
    """Perform NNI refinement on splits to improve quartet score.
    Returns improved splits (list of frozensets)."""
    if n <= 4 or len(splits) < 1:
        return splits

    all_leaves = frozenset(range(n))

    def quartet_score(splits_set, quad):
        """Compute the quartet score for a 4-taxon subset given a split set.
        Returns the minimum score among quartet topologies consistent with the splits."""
        i, j, k, l = quad
        q1 = d[i, k] + d[j, l] - d[i, j] - d[k, l]  # (i,j)|(k,l)
        q2 = d[i, l] + d[j, k] - d[i, k] - d[j, l]  # (i,k)|(j,l)
        q3 = d[i, j] + d[k, l] - d[i, l] - d[j, k]  # (i,l)|(j,k)

        def is_consistent(a, b, c, dd, splits_list):
            """Check if (a,b)|(c,d) is consistent with all splits."""
            for split in splits_list:
                left = split
                right = all_leaves - split
                ab_together = (a in left and b in left) or (a in right and b in right)
                cd_together = (c in left and d in left) or (c in right and d in right)
                # In an unrooted tree, (a,b)|(c,d) means there's a split separating {a,b} from {c,d}
                # This is consistent if no split separates a from b while keeping c,d together
                # OR separates c from d while keeping a,b together
                # More precisely: a split {S} is compatible with (a,b)|(c,d) if
                # either a,b are on same side AND c,d are on same side
                if not (ab_together and cd_together):
                    # Check if the split actually contradicts
                    # (a,b)|(c,d) requires that a and b are on the same side of the quartet's
                    # defining split. A tree split is compatible if it doesn't "cut" a,b apart
                    # while keeping c,d together, or vice versa.
                    # Actually for NNI scoring, we just need: does this split set induce
                    # the quartet (a,b)|(c,d)? The quartet is induced if there exists
                    # a split in the tree that separates {a,b} from {c,d}.
                    # For consistency: the split set should NOT have a split that
                    # separates a from b (i.e., a in left, b in right) AND c,d on same side,
                    # because that would imply (a,c)|(b,d) or (a,d)|(b,c).
                    a_left = a in left
                    b_left = b in left
                    c_left = c in left
                    d_left = d in left

                    # Split separates a,b: a_left != b_left
                    # Split separates c,d: c_left != d_left
                    # If split separates a,b but not c,d: implies (a,c)|(b,d) or (a,d)|(b,c)
                    #   which contradicts (a,b)|(c,d)
                    # If split separates c,d but not a,b: similar contradiction
                    # If split separates both: could be consistent or not depending on arrangement
                    ab_sep = a_left != b_left
                    cd_sep = c_left != d_left

                    if ab_sep and not cd_sep:
                        return False
                    if cd_sep and not ab_sep:
                        return False
                    if ab_sep and cd_sep:
                        # Both separated - need to check the specific pattern
                        # (a,b)|(c,d) requires a,c on one side and b,d on the other,
                        # OR a,d on one side and b,c on the other? No.
                        # Actually if both are separated, the quartet induced depends on
                        # which elements are together.
                        # For (a,b)|(c,d): we need a and b together, c and d together.
                        # If a_left != b_left and c_left != d_left, then we have two possible
                        # groupings: {a,c},{b,d} or {a,d},{b,c}
                        # Neither is {a,b},{c,d}, so it's inconsistent.
                        return False
                    # If neither is separated, it's fine
            return True

        best = float('inf')
        if is_consistent(i, j, k, l, splits_set):
            best = min(best, q1)
        if is_consistent(i, k, j, l, splits_set):
            best = min(best, q2)
        if is_consistent(i, l, j, k, splits_set):
            best = min(best, q3)
        if best == float('inf'):
            best = min(q1, q2, q3)
        return best

    def total_score(splits_list):
        total = 0.0
        for quad in itertools.combinations(range(n), 4):
            total += quartet_score(splits_list, quad)
        return total

    current_splits = list(splits)
    current_score = total_score(current_splits)

    for pass_num in range(max_passes):
        if not _check_time(deadline):
            break

        improved = True
        while improved:
            if not _check_time(deadline):
                break
            improved = False
            for idx in range(len(current_splits)):
                if not _check_time(deadline):
                    break
                split = current_splits[idx]
                left = set(split)
                right = set(all_leaves) - left

                # NNI: for each pair (a,b) where a in left, b in right,
                # try the alternative splits
                # The current split separates left from right.
                # In NNI, we pick an internal edge. The two sides are left and right.
                # An NNI move at this edge picks a from left, b from right,
                # and creates new splits: left - {a} + {b} and right - {b} + {a}
                # Actually NNI on edge between L and R (|L|>=2, |R|>=2):
                # Pick a in L, b in R. New splits: L\{a} ∪ {b} and R\{b} ∪ {a}

                left_list = sorted(left)
                right_list = sorted(right)

                # Only consider if both sides have at least 2 elements (for unrooted binary)
                if len(left_list) < 2 or len(right_list) < 2:
                    continue

                # Sample a few pairs to save time (or all if small)
                pairs = list(itertools.product(left_list, right_list))
                if len(pairs) > 50:
                    # Random sample
                    import random
                    random.seed(42)
                    pairs = random.sample(pairs, 50)

                for a, b in pairs:
                    if not _check_time(deadline):
                        break
                    # New splits after NNI
                    new_left = frozenset((left - {a}) | {b})
                    new_right = frozenset((right - {b}) | {a})

                    if len(new_left) < 2 or len(new_right) < 2:
                        continue
                    if len(new_left) == len(left) and len(new_right) == len(right):
                        # Same split, skip
                        pass

                    # Check if new_left equals any existing split (would be duplicate)
                    # Build new split set
                    new_splits = list(current_splits)
                    new_splits[idx] = new_left

                    # Check compatibility (no crossing)
                    compatible = True
                    for si in range(len(new_splits)):
                        s1 = new_splits[si]
                        for sj in range(si + 1, len(new_splits)):
                            s2 = new_splits[sj]
                            # Check if s1 and s2 cross
                            a1 = s1
                            b1 = all_leaves - s1
                            a2 = s2
                            b2 = all_leaves - s2
                            if len(a1 & a2) > 0 and len(a1 & b2) > 0 and len(b1 & a2) > 0 and len(b1 & b2) > 0:
                                compatible = False
                                break
                        if not compatible:
                            break
                    if not compatible:
                        continue

                    new_score = total_score(new_splits)
                    if new_score < current_score - 1e-10:
                        current_splits = new_splits
                        current_score = new_score
                        improved = True
                        break  # Restart scanning from beginning

    return current_splits


def _small_n_exhaustive(d, n, labels):
    """For n <= 6, exhaustively search all unrooted binary trees using quartet scoring."""
    if n <= 3:
        return _build_newick_from_splits([], n, labels)

    leaves = list(range(n))
    all_leaves = frozenset(range(n))

    def crosses(s1, s2):
        a1 = s1
        b1 = all_leaves - s1
        a2 = s2
        b2 = all_leaves - s2
        return (len(a1 & a2) > 0 and len(a1 & b2) > 0 and
                len(b1 & a2) > 0 and len(b1 & b2) > 0)

    # Generate all possible 2-(n-2) splits (only consider the smaller side for uniqueness)
    all_splits = []
    for size in range(2, n // 2 + 1):
        if size == n - size:
            # To avoid duplicates, only consider subsets whose min element is in the subset
            for subset in itertools.combinations(range(n), size):
                if min(range(n)) in subset or min(subset) == 0:
                    all_splits.append(frozenset(subset))
        else:
            for subset in itertools.combinations(range(n), size):
                all_splits.append(frozenset(subset))

    # Need exactly n-3 splits for a binary tree on n leaves
    n_splits_needed = n - 3

    def quartet_score_for_splits(split_set, quad):
        i, j, k, l = quad
        q1 = d[i, k] + d[j, l] - d[i, j] - d[k, l]
        q2 = d[i, l] + d[j, k] - d[i, k] - d[j, l]
        q3 = d[i, j] + d[k, l] - d[i, l] - d[j, k]

        def consistent(a, b, c, dd, splits_list):
            for split in splits_list:
                left = split
                right = all_leaves - split
                a_l = a in left
                b_l = b in left
                c_l = c in left
                d_l = dd in left
                ab_sep = a_l != b_l
                cd_sep = c_l != d_l
                if ab_sep and not cd_sep:
                    return False
                if cd_sep and not ab_sep:
                    return False
                if ab_sep and cd_sep:
                    return False
            return True

        best = float('inf')
        if consistent(i, j, k, l, split_set):
            best = min(best, q1)
        if consistent(i, k, j, l, split_set):
            best = min(best, q2)
        if consistent(i, l, j, k, split_set):
            best = min(best, q3)
        if best == float('inf'):
            best = min(q1, q2, q3)
        return best

    def total_quartet_score(split_set):
        total = 0.0
        for quad in itertools.combinations(range(n), 4):
            total += quartet_score_for_splits(split_set, quad)
        return total

    best_score = float('inf')
    best_splits = None

    # For n=4: 3 possible trees
    if n == 4:
        candidates = [
            frozenset({0, 1}),  # (0,1)|(2,3)
            frozenset({0, 2}),  # (0,2)|(1,3)
            frozenset({0, 3}),  # (0,3)|(1,2)
        ]
        for s in candidates:
            score = total_quartet_score([s])
            if score < best_score:
                best_score = score
                best_splits = [s]

    elif n == 5:
        # 15 unrooted binary trees on 5 leaves
        # Each tree has 2 splits (each 2-3)
        for s1_idx in range(len(all_splits)):
            s1 = all_splits[s1_idx]
            for s2_idx in range(s1_idx + 1, len(all_splits)):
                s2 = all_splits[s2_idx]
                if crosses(s1, s2):
                    continue
                # Check compatibility (no crossing) - for 2 splits on 5 leaves,
                # they're compatible iff they don't cross
                score = total_quartet_score([s1, s2])
                if score < best_score:
                    best_score = score
                    best_splits = [s1, s2]

    elif n == 6:
        # 105 unrooted binary trees on 6 leaves
        # Each tree has 3 splits
        splits_by_size = {}
        for s in all_splits:
            size = len(s)
            if size not in splits_by_size:
                splits_by_size[size] = []
            splits_by_size[size].append(s)

        for s1 in all_splits:
            if not _check_time():
                break
            for s2 in all_splits:
                if s2 <= s1:
                    continue
                if crosses(s1, s2):
                    continue
                for s3 in all_splits:
                    if s3 <= s1 or s3 <= s2:
                        continue
                    if crosses(s1, s3) or crosses(s2, s3):
                        continue
                    score = total_quartet_score([s1, s2, s3])
                    if score < best_score:
                        best_score = score
                        best_splits = [s1, s2, s3]

    if best_splits is not None:
        return _build_newick_from_splits(best_splits, n, labels)

    return None


def reconstruct(sequences):
    """Reconstruct unrooted tree topology from DNA sequences."""
    global _START_TIME
    _START_TIME = time.time()
    DEADLINE = 250.0

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    # Compute distance matrix
    d = _jc_distance_matrix(sequences, labels)

    # For small n (<=6), use exhaustive quartet search
    if n <= 6:
        result = _small_n_exhaustive(d, n, labels)
        if result is not None:
            return result

    # For n > 6, use NJ then NNI refinement
    newick, splits = _nj_build_tree(d, labels)

    if splits and n > 4 and _check_time(DEADLINE):
        # NNI refinement
        refined_splits = _nni_refine(splits, d, n, labels, max_passes=3, deadline=DEADLINE)
        if refined_splits and refined_splits != splits:
            newick = _build_newick_from_splits(refined_splits, n, labels)

    return newick
