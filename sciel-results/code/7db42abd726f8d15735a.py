import time
import random
import numpy as np

def construct():
    """Return 256 colours in {0,1} for integers 1..256 minimizing monochromatic 4-APs."""
    n = 256
    deadline = time.time() + 140  # 10s buffer for 150s limit

    # Precompute all 4-term APs: a, a+d, a+2d, a+3d with 0-based indexing
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))

    num_aps = len(ap_list)

    # Build arrays for vectorized computation
    # For each AP, store the 4 positions
    ap_positions = np.array(ap_list, dtype=np.int32)  # shape (num_aps, 4)

    # For each position p, precompute:
    # - the indices of APs containing p
    # - for each such AP, the other 3 positions
    # This allows fast delta computation when flipping p

    # Precompute: for each position, list of (ap_idx, other_positions)
    pos_ap_data = [[] for _ in range(n)]
    for idx in range(num_aps):
        a, b, c, d_pos = ap_positions[idx]
        pos_ap_data[a].append((idx, b, c, d_pos))
        pos_ap_data[b].append((idx, a, c, d_pos))
        pos_ap_data[c].append((idx, a, b, d_pos))
        pos_ap_data[d_pos].append((idx, a, b, c))

    # Convert to numpy arrays for each position
    pos_ap_indices = []
    pos_other_positions = []
    for p in range(n):
        if pos_ap_data[p]:
            indices = [x[0] for x in pos_ap_data[p]]
            others = np.array([x[1:] for x in pos_ap_data[p]], dtype=np.int32)
            pos_ap_indices.append(np.array(indices, dtype=np.int32))
            pos_other_positions.append(others)
        else:
            pos_ap_indices.append(np.array([], dtype=np.int32))
            pos_other_positions.append(np.array([], dtype=np.int32))

    def count_mono(coloring):
        """Count monochromatic 4-APs using vectorized operations."""
        # coloring shape (n,), ap_positions shape (num_aps, 4)
        # Check if all 4 positions have the same color
        colors_at_aps = coloring[ap_positions]  # shape (num_aps, 4)
        # All same if all equal to first element
        mono = np.all(colors_at_aps == colors_at_aps[:, 0:1], axis=1)
        return int(np.sum(mono))

    def thue_morse():
        """Generate Thue-Morse sequence."""
        tm = np.zeros(n, dtype=np.int8)
        for i in range(1, n):
            tm[i] = tm[i >> 1] ^ (i & 1)
        return tm

    def alternating():
        """Generate alternating pattern."""
        return np.array([(i // 2) % 2 for i in range(n)], dtype=np.int8)

    def random_coloring():
        return np.random.randint(0, 2, size=n).astype(np.int8)

    # Initialize with several strategies
    candidates = []
    candidates.append(thue_morse())
    candidates.append(alternating())
    candidates.append(random_coloring())
    candidates.append(random_coloring())

    def tabu_search(initial, max_steps=100000, tabu_tenure=15):
        """Tabu search with aspiration criterion."""
        coloring = initial.copy()
        current_cost = count_mono(coloring)
        best_coloring = coloring.copy()
        best_cost = current_cost

        # Tabu list: for each position, the iteration at which it was last flipped
        last_flipped = np.full(n, -1000, dtype=np.int64)
        step = 0
        start_time = time.time()

        while step < max_steps and (time.time() - start_time) < 130:
            if time.time() > deadline:
                break

            # Compute delta for all positions simultaneously (vectorized)
            # For each position p, delta = sum over APs containing p of:
            #   +1 if flipping p creates a mono AP, -1 if it destroys one
            #
            # Before flip: AP is mono iff coloring[p] == all others
            # After flip: AP is mono iff (1-coloring[p]) == all others
            #
            # Let's compute this efficiently.
            # For a given position p and an AP containing p with other positions o1,o2,o3:
            #   before_mono = (coloring[p] == coloring[o1] == coloring[o2] == coloring[o3])
            #   after_mono = ((1-coloring[p]) == coloring[o1] == coloring[o2] == coloring[o3])
            #   delta contribution = after_mono - before_mono (as int: +1 or -1 or 0)
            #
            # Note: before_mono and after_mono cannot both be true (since coloring[p] != 1-coloring[p])
            # So delta is +1, -1, or 0.
            #
            # More precisely:
            #   If the 3 others are all same color c:
            #     before_mono = (coloring[p] == c)
            #     after_mono = (1-coloring[p] == c)
            #     So if coloring[p]==c: before=1, after=0, delta=-1
            #     If coloring[p]!=c: before=0, after=1, delta=+1
            #   If the 3 others are NOT all same:
            #     before_mono = 0, after_mono = 0, delta=0
            #
            # So for position p, delta = sum over APs containing p of:
            #   -1 if all 3 others have color == coloring[p]
            #   +1 if all 3 others have color == 1-coloring[p]
            #   0 otherwise
            #
            # Vectorized approach: for each position p, we need to check if the 3 others
            # are all equal to coloring[p] or all equal to 1-coloring[p].

            # Compute delta for all positions at once
            deltas = np.zeros(n, dtype=np.int32)

            for p in range(n):
                if len(pos_ap_indices[p]) == 0:
                    continue
                others = pos_other_positions[p]  # shape (k, 3)
                if len(others) == 0:
                    continue

                # Get colors of the other positions
                other_colors = coloring[others]  # shape (k, 3)

                # Check if all 3 others have the same color
                all_same = np.all(other_colors == other_colors[:, 0:1], axis=1)  # shape (k,)

                # If all same, what color?
                other_c = other_colors[:, 0]  # shape (k,)

                # delta contribution:
                # -1 if coloring[p] == other_c (and all same)
                # +1 if (1-coloring[p]) == other_c (and all same)
                # 0 otherwise

                cp = coloring[p]
                delta_p = np.where(all_same,
                                   np.where(other_c == cp, -1, 1),
                                   0).astype(np.int32)
                deltas[p] = int(np.sum(delta_p))

            # Find best non-tabu move
            # Tabu: position p is tabu if step - last_flipped[p] < tabu_tenure
            is_tabu = (step - last_flipped) < tabu_tenure

            # Score each move: prefer lower delta (more negative = better)
            # For tabu moves, apply aspiration: allow if it improves best_cost
            # Effective delta: for tabu moves without aspiration, treat as very bad
            effective_deltas = deltas.copy()
            for p in range(n):
                if is_tabu[p]:
                    # Aspiration: allow if new_cost < best_cost
                    new_cost = current_cost + deltas[p]
                    if new_cost >= best_cost:
                        effective_deltas[p] = 1000000  # very bad

            # Pick the position with minimum effective delta
            best_p = int(np.argmin(effective_deltas))
            best_delta = int(deltas[best_p])

            # Apply the flip
            coloring[best_p] = 1 - coloring[best_p]
            current_cost += best_delta
            last_flipped[best_p] = step

            if current_cost < best_cost:
                best_cost = current_cost
                best_coloring = coloring.copy()

            step += 1

        return best_coloring, best_cost

    # Run tabu search from each candidate
    best_overall = None
    best_overall_cost = float('inf')

    for candidate in candidates:
        result, cost = tabu_search(candidate)
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall = result

        if time.time() > deadline - 10:
            break

    # If we have time, try more random restarts
    if best_overall is not None and time.time() < deadline - 15:
        num_restarts = 0
        while time.time() < deadline - 5:
            restart = random_coloring()
            result, cost = tabu_search(restart, max_steps=30000, tabu_tenure=10)
            if cost < best_overall_cost:
                best_overall_cost = cost
                best_overall = result
            num_restarts += 1

    # Convert to list of ints
    return [int(c) for c in best_overall]
