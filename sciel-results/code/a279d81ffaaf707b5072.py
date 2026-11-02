"""Kissing number in dimension 11: max independent set via Tomita B&B + local search."""
import numpy as np
import itertools
import time
import sys


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    D = 11

    # Generate candidate vectors: weight 1, 2, 3, 4
    candidates_list = []

    # Weight-4 vectors (norm^2=4): C(11,4)*16 = 5280
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Weight-3 vectors (norm^2=3): C(11,3)*8 = 1320
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Weight-2 vectors (norm^2=2): C(11,2)*4 = 220
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Weight-1 vectors (norm^2=1): 22
    for pos in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[pos] = s
            candidates_list.append(v)

    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)  # 6842

    # Precompute norms squared
    norms_sq = np.sum(candidates.astype(np.int32) ** 2, axis=1)
    max_norm_sq = int(norms_sq.max())  # 4

    # Precompute dot product matrix
    dot = candidates.astype(np.int32) @ candidates.T.astype(np.int32)

    # Constraint: ||v_i - v_j||^2 >= max_norm_sq
    # => v_i·v_j <= (||v_i||^2 + ||v_j||^2 - max_norm_sq) / 2
    threshold = (norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - max_norm_sq) / 2.0
    compat = (dot <= threshold).astype(np.uint8)
    np.fill_diagonal(compat, 1)

    # conflict[i,j] = 1 if i and j CANNOT both be selected
    conflict = (1 - compat).astype(np.uint8)
    np.fill_diagonal(conflict, 0)

    # Adjacency for B&B: adj[i,j] = 1 if i and j are compatible (can be in same set)
    # We want maximum independent set in conflict graph = maximum clique in compat graph
    # Tomita's algorithm finds maximum clique
    adj = compat.copy()
    np.fill_diagonal(adj, 0)

    # Degree for ordering
    deg = adj.sum(axis=1)

    # Tomita's maximum clique algorithm with branch and bound
    # Uses color sort for upper bound
    best_clique_size = [0]
    best_clique = [[]]

    def color_sort(P, order):
        """Color sort for upper bound. P is set of candidate vertices."""
        if len(P) == 0:
            return [], []

        # Greedy coloring
        colors = {}
        color_classes = []
        for v in P:
            placed = False
            for ci, cc in enumerate(color_classes):
                compatible_with_all_in_class = True
                for u in cc:
                    if not adj[v, u]:
                        compatible_with_all_in_class = False
                        break
                if compatible_with_all_in_class:
                    cc.append(v)
                    colors[v] = ci + 1
                    placed = True
                    break
            if not placed:
                color_classes.append([v])
                colors[v] = len(color_classes)

        # Sort by color (ascending) - vertices with higher color go later
        sorted_P = sorted(P, key=lambda v: colors[v])
        max_color = max(colors.values()) if colors else 0
        return sorted_P, max_color

    def expand(R, P, depth):
        """Tomita's expand: R is current clique, P is candidates."""
        if time.time() > deadline:
            return

        if len(P) == 0:
            if len(R) > best_clique_size[0]:
                best_clique_size[0] = len(R)
                best_clique[0] = list(R)
            return

        # Color sort
        sorted_P, max_color = color_sort(P, None)

        # Process in reverse order (highest color first)
        for i in range(len(sorted_P) - 1, -1, -1):
            if time.time() > deadline:
                return

            v = sorted_P[i]
            # Upper bound: current clique size + remaining vertices + max color
            if len(R) + i + 1 <= best_clique_size[0]:
                return

            # Include v
            R.append(v)
            # New P: vertices in sorted_P[0:i] that are adjacent to v
            new_P = [u for u in sorted_P[:i] if adj[v, u]]
            expand(R, new_P, depth + 1)
            R.pop()

    # Run Tomita's algorithm
    # Start with all vertices
    all_vertices = list(range(n))
    # Sort by degree descending for better initial ordering
    all_vertices.sort(key=lambda v: -deg[v])

    expand([], all_vertices, 0)

    # Local search improvement
    if best_clique[0] and time.time() < deadline - 5.0:
        selected = list(best_clique[0])
        sel_set = set(selected)

        # Try to add more compatible vectors
        compat_with_all = np.ones(n, dtype=np.uint8)
        for s in selected:
            compat_with_all = compat_with_all & compat[s]
        for s in selected:
            compat_with_all[s] = 0

        full_deg = conflict.sum(axis=1)
        addable = np.where(compat_with_all > 0)[0]
        for idx in addable[np.argsort(full_deg[addable])]:
            if time.time() > deadline - 5.0:
                break
            selected.append(idx)
            sel_set.add(idx)
            compat_with_all = compat_with_all & compat[idx]
            compat_with_all[idx] = 0

        # Swap local search
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            found_swap = False
            for si in range(len(selected)):
                if time.time() > deadline - 5.0:
                    break
                s = selected[si]
                # Find candidates compatible with all selected except s
                mask_except_s = np.ones(n, dtype=np.uint8)
                for s2 in selected:
                    if s2 != s:
                        mask_except_s = mask_except_s & compat[:, s2]
                mask_except_s[s] = 0
                mask_except_s[list(sel_set)] = 0

                candidates_for_swap = np.where(mask_except_s > 0)[0]
                if len(candidates_for_swap) < 2:
                    continue

                sorted_cands = candidates_for_swap[np.argsort(full_deg[candidates_for_swap])]
                found_pair = False
                for i in range(min(20, len(sorted_cands))):
                    c1 = sorted_cands[i]
                    for j in range(i + 1, min(40, len(sorted_cands))):
                        c2 = sorted_cands[j]
                        if compat[c1, c2]:
                            # Do the swap
                            selected.remove(s)
                            sel_set.discard(s)
                            selected.append(c1)
                            sel_set.add(c1)
                            selected.append(c2)
                            sel_set.add(c2)
                            found_swap = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                if found_swap:
                    break

            if not found_swap:
                break

        best_clique[0] = selected

    if not best_clique[0]:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)

    result = candidates[np.array(best_clique[0], dtype=np.int64)]
    return result.astype(np.int64)
