"""
Batched local search for two-colourings of [1,256] minimizing monochromatic 4-APs.
Key idea: represent colors as ±1 values, precompute a per-position contribution matrix,
and compute all 256 flip-deltas in one vectorized pass. Use multi-flip moves to escape
single-flip plateaus.
"""
import time
import random
import numpy as np


def construct():
    deadline = time.time() + 135  # Safety margin before 150s limit

    # Precompute all 4-term APs as 0-indexed tuples
    aps = []
    for d in range(1, 86):
        max_start = 256 - 3 * d
        for a in range(1, max_start + 1):
            i0 = a - 1
            aps.append((i0, i0 + d, i0 + 2 * d, i0 + 3 * d))
    aps = np.array(aps, dtype=np.int32)
    n_aps = len(aps)

    ap_i0 = aps[:, 0]
    ap_i1 = aps[:, 1]
    ap_i2 = aps[:, 2]
    ap_i3 = aps[:, 3]

    # Build per-position AP index arrays
    pos_aps = [[] for _ in range(256)]
    for idx in range(n_aps):
        pos_aps[ap_i0[idx]].append(idx)
        pos_aps[ap_i1[idx]].append(idx)
        pos_aps[ap_i2[idx]].append(idx)
        pos_aps[ap_i3[idx]].append(idx)

    pos_ap_indices = [np.array(pos_aps[p], dtype=np.int32) for p in range(256)]

    def count_mono(colors):
        """colors: array of 0/1, length 256"""
        c0 = colors[ap_i0]
        c1 = colors[ap_i1]
        c2 = colors[ap_i2]
        c3 = colors[ap_i3]
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(mono.sum())

    def compute_all_deltas(colors):
        """
        colors: array of 0/1, length 256.
        Returns delta array: delta[p] = change in mono count if we flip position p.
        """
        deltas = np.zeros(256, dtype=np.int32)
        for p in range(256):
            ap_idxs = pos_ap_indices[p]
            if len(ap_idxs) == 0:
                continue
            old = colors[p]
            new = 1 - old

            c0 = colors[ap_i0[ap_idxs]]
            c1 = colors[ap_i1[ap_idxs]]
            c2 = colors[ap_i2[ap_idxs]]
            c3 = colors[ap_i3[ap_idxs]]

            was_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)

            # After flipping p, the color at p becomes 'new'.
            # For each AP, check if all four would be 'new'.
            # The color at p is one of c0,c1,c2,c3 depending on role.
            # We need the other three to equal 'new'.
            # Build is_mono_new: for each AP, True if all four colors (with p flipped) are 'new'.
            is_mono_new = np.ones(len(ap_idxs), dtype=bool)
            for r in range(4):
                # Determine which position index corresponds to role r
                if r == 0:
                    role_idx = ap_i0[ap_idxs]
                elif r == 1:
                    role_idx = ap_i1[ap_idxs]
                elif r == 2:
                    role_idx = ap_i2[ap_idxs]
                else:
                    role_idx = ap_i3[ap_idxs]

                mask = (role_idx == p)
                if not np.any(mask):
                    continue
                # For these APs, p is at role r. The other three must be 'new'.
                # Get the three other color arrays
                if r == 0:
                    others = [(c1 == new), (c2 == new), (c3 == new)]
                elif r == 1:
                    others = [(c0 == new), (c2 == new), (c3 == new)]
                elif r == 2:
                    others = [(c0 == new), (c1 == new), (c3 == new)]
                else:
                    others = [(c0 == new), (c1 == new), (c2 == new)]

                all_true = np.ones(len(ap_idxs), dtype=bool)
                for ot in others:
                    all_true &= ot
                is_mono_new[mask] = all_true[mask]

            delta = int(np.sum(~was_mono & is_mono_new)) - int(np.sum(was_mono & ~is_mono_new))
            deltas[p] = delta
        return deltas

    def local_search(colors, current_count, max_rounds=200, k=1):
        """
        Greedy local search: repeatedly find the k most beneficial flips and apply them.
        """
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline - 5:
            deltas = compute_all_deltas(colors)
            neg_mask = deltas < 0
            if not np.any(neg_mask):
                break
            neg_positions = np.where(neg_mask)[0]
            neg_deltas = deltas[neg_positions]
            order = np.argsort(neg_deltas)
            neg_positions = neg_positions[order]
            neg_deltas = neg_deltas[order]

            n_flip = min(k, len(neg_positions))
            # Apply flips sequentially (approximation of simultaneous)
            for i in range(n_flip):
                pos = neg_positions[i]
                colors[pos] = 1 - colors[pos]
                current_count += neg_deltas[i]

            improved = True
            rounds += 1

        return colors, current_count

    def two_opt_local_search(colors, current_count, max_rounds=50):
        """
        Try all pairs of positions and flip both if it improves.
        This is O(256^2) per round but can escape single-flip plateaus.
        Only run a few rounds.
        """
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline - 5:
            best_delta = 0
            best_pair = None
            # Sample a subset of pairs to keep it fast
            # Try all pairs where both positions have negative single-flip delta
            deltas = compute_all_deltas(colors)
            neg_positions = np.where(deltas < 0)[0]
            if len(neg_positions) < 2:
                break

            # Try pairs among the top negative positions
            top_neg = neg_positions[np.argsort(deltas[neg_positions])][:20]
            for i in range(len(top_neg)):
                for j in range(i + 1, len(top_neg)):
                    p1, p2 = top_neg[i], top_neg[j]
                    # Compute delta for flipping both p1 and p2
                    old1, old2 = colors[p1], colors[p2]
                    new1, new2 = 1 - old1, 1 - old2

                    # Collect all APs involving p1 or p2
                    ap_set = set()
                    for idx in pos_ap_indices[p1]:
                        ap_set.add(idx)
                    for idx in pos_ap_indices[p2]:
                        ap_set.add(idx)
                    ap_list = list(ap_set)
                    if len(ap_list) == 0:
                        continue
                    ap_arr = np.array(ap_list, dtype=np.int32)

                    c0 = colors[ap_i0[ap_arr]]
                    c1 = colors[ap_i1[ap_arr]]
                    c2 = colors[ap_i2[ap_arr]]
                    c3 = colors[ap_i3[ap_arr]]

                    was_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)

                    # After flipping p1 and p2, compute new mono status
                    # For each AP, we need to check if all four positions (with flips applied) are equal
                    new_c0 = c0.copy()
                    new_c1 = c1.copy()
                    new_c2 = c2.copy()
                    new_c3 = c3.copy()

                    # Apply flip at p1
                    for r, arr in enumerate([ap_i0[ap_arr], ap_i1[ap_arr], ap_i2[ap_arr], ap_i3[ap_arr]]):
                        mask = (arr == p1)
                        if r == 0:
                            new_c0[mask] = new1
                        elif r == 1:
                            new_c1[mask] = new1
                        elif r == 2:
                            new_c2[mask] = new1
                        else:
                            new_c3[mask] = new1

                    # Apply flip at p2
                    for r, arr in enumerate([ap_i0[ap_arr], ap_i1[ap_arr], ap_i2[ap_arr], ap_i3[ap_arr]]):
                        mask = (arr == p2)
                        if r == 0:
                            new_c0[mask] = new2
                        elif r == 1:
                            new_c1[mask] = new2
                        elif r == 2:
                            new_c2[mask] = new2
                        else:
                            new_c3[mask] = new2

                    is_mono_new = (new_c0 == new_c1) & (new_c1 == new_c2) & (new_c2 == new_c3)

                    delta = int(np.sum(~was_mono & is_mono_new)) - int(np.sum(was_mono & ~is_mono_new))
                    if delta < best_delta:
                        best_delta = delta
                        best_pair = (p1, p2)

            if best_pair is not None:
                p1, p2 = best_pair
                colors[p1] = 1 - colors[p1]
                colors[p2] = 1 - colors[p2]
                current_count += best_delta
                improved = True
            else:
                improved = False
            rounds += 1

        return colors, current_count

    # Main search loop
    best_colors = None
    best_count = float('inf')

    n_restarts = 30
    for seed in range(n_restarts):
        if time.time() > deadline - 15:
            break
        random.seed(seed)
        colors = np.array([random.randint(0, 1) for _ in range(256)], dtype=np.int8)
        current_count = count_mono(colors)

        # Phase 1: single-flip local search
        colors, current_count = local_search(colors, current_count, max_rounds=100, k=1)

        # Phase 2: multi-flip (k=3)
        colors, current_count = local_search(colors, current_count, max_rounds=50, k=3)

        # Phase 3: more single-flip
        colors, current_count = local_search(colors, current_count, max_rounds=100, k=1)

        # Phase 4: two-opt
        if time.time() < deadline - 20:
            colors, current_count = two_opt_local_search(colors, current_count, max_rounds=20)
            # Follow up with single-flip
            colors, current_count = local_search(colors, current_count, max_rounds=50, k=1)

        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()

        # If good, refine further
        if best_count <= 40 and time.time() < deadline - 25:
            colors, current_count = local_search(colors, current_count, max_rounds=50, k=1)
            colors, current_count = local_search(colors, current_count, max_rounds=30, k=5)
            colors, current_count = local_search(colors, current_count, max_rounds=50, k=1)
            if current_count < best_count:
                best_count = current_count
                best_colors = colors.copy()

    # Final refinement
    if best_colors is not None and time.time() < deadline - 10:
        colors = best_colors.copy()
        current_count = best_count
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            colors, current_count = local_search(colors, current_count, max_rounds=50, k=1)
            colors, current_count = local_search(colors, current_count, max_rounds=20, k=3)
            colors, current_count = local_search(colors, current_count, max_rounds=50, k=1)
            if current_count < best_count:
                best_count = current_count
                best_colors = colors.copy()

    return [int(c) for c in best_colors]
