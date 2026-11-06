"""Rewritten program: minimize monochromatic 4-AP count via local search."""
import time
import random
import numpy as np


def _count_mono_4ap(colors):
    """Count monochromatic 4-term APs in a 2-coloring of 1..256.
    colors is a list/tuple/array of 256 values in {0,1}, indexed 0..255 for integers 1..256.
    """
    n = 256
    count = 0
    c = np.asarray(colors, dtype=np.int8)
    for a in range(n):
        for d in range(1, n):
            a3 = a + 3 * d
            if a3 >= n:
                break
            if c[a] == c[a + d] == c[a + 2 * d] == c[a3]:
                count += 1
    return count


def construct():
    """
    Minimize the number of monochromatic 4-term arithmetic progressions
    in a 2-coloring of [1, 256].
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    N = 256
    rng = random.Random(42)

    # Precompute all 4-APs as (a, a+d, a+2d, a+3d) zero-indexed
    aps = []
    for a in range(N):
        for d in range(1, N):
            a3 = a + 3 * d
            if a3 >= N:
                break
            aps.append((a, a + d, a + 2 * d, a3))

    num_aps = len(aps)

    # For each position, precompute which APs it participates in
    # aps_by_pos[i] = list of (ap_index, role) where role in {0,1,2,3}
    aps_by_pos = [[] for _ in range(N)]
    for idx, (a, b, c, d) in enumerate(aps):
        aps_by_pos[a].append((idx, 0))
        aps_by_pos[b].append((idx, 1))
        aps_by_pos[c].append((idx, 2))
        aps_by_pos[d].append((idx, 3))

    def fast_count(colors):
        """Fast count using precomputed AP list."""
        count = 0
        for (a, b, c, d) in aps:
            if colors[a] == colors[b] == colors[c] == colors[d]:
                count += 1
        return count

    def delta_flip(colors, pos, ap_list):
        """
        Compute change in mono-4AP count if we flip colors[pos].
        Returns (delta, new_count) where new_count = old_count + delta.
        We need the current count to compute this; instead, compute directly.
        """
        # For each AP involving pos, check if flipping pos makes it mono or breaks it
        delta = 0
        for (ap_idx, role) in ap_list:
            a, b, c, d = aps[ap_idx]
            # Current state
            vals = [colors[a], colors[b], colors[c], colors[d]]
            cur_mono = (vals[0] == vals[1] == vals[2] == vals[3])
            # After flip
            vals[role] = 1 - vals[role]
            new_mono = (vals[0] == vals[1] == vals[2] == vals[3])
            if new_mono and not cur_mono:
                delta += 1
            elif cur_mono and not new_mono:
                delta -= 1
        return delta

    # Initial coloring: random
    best_colors = np.array([rng.randint(0, 1) for _ in range(N)], dtype=np.int8)
    best_count = fast_count(best_colors)

    # Local search: iterated local search with restarts
    current = best_colors.copy()
    current_count = best_count

    # Coordinate descent pass
    def coordinate_descent(colors, count, max_passes=50):
        for _ in range(max_passes):
            improved = False
            for pos in range(N):
                delta = delta_flip(colors, pos, aps_by_pos[pos])
                if delta < 0:
                    colors[pos] = 1 - colors[pos]
                    count += delta
                    improved = True
            if not improved:
                break
        return count

    # Iterated local search
    while time.time() < deadline:
        # Local search from current
        current_count = coordinate_descent(current, current_count, max_passes=200)

        if current_count < best_count:
            best_count = current_count
            best_colors = current.copy()

        # Random perturbation: flip a few random positions
        num_flips = rng.randint(1, 5)
        for _ in range(num_flips):
            pos = rng.randint(0, N - 1)
            current[pos] = 1 - current[pos]
            current_count = fast_count(current)  # Recompute after perturbation

        # Check time
        if time.time() > deadline - 5:
            break

    return [int(x) for x in best_colors]
