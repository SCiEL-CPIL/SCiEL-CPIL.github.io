"""Local search over 2-colorings of [1..256] to minimize monochromatic 4-APs."""
import time
import random
import numpy as np


def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Precompute all 4-term APs as index tuples (0-based)
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            if a + 3 * d <= N:
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))

    num_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)

    # For each position, list of (ap_index, other_positions) for fast delta
    # We'll use a numpy-based approach for speed.

    def count_mono(colors):
        """Count monochromatic 4-APs. colors is a numpy array of 0/1."""
        # For each AP, check if all 4 positions have same color
        c = colors[aps_arr]  # shape (num_aps, 4)
        # monochromatic if all equal: first == second == third == fourth
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return int(mono.sum())

    def thue_morse(n):
        """Thue-Morse sequence: parity of number of 1-bits."""
        return bin(n).count('1') % 2

    # Initial coloring: Thue-Morse
    colors = np.array([thue_morse(i) for i in range(N)], dtype=np.int8)
    best_count = count_mono(colors)
    best_colors = colors.copy()

    # Precompute for each position i, the list of AP indices that contain i
    pos_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)

    # Greedy local search: repeatedly find the flip that gives best improvement
    # Use numpy for speed in evaluating delta for a given flip
    def delta_for_flip(colors, pos):
        """Compute change in mono count if we flip position `pos`."""
        # Get all APs containing pos
        ap_indices = pos_to_aps[pos]
        if not ap_indices:
            return 0
        ap_indices = np.array(ap_indices, dtype=np.int32)
        c = colors[aps_arr[ap_indices]]  # shape (len, 4)
        # Current state: is this AP currently monochromatic?
        currently_mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        # After flip: the color at pos changes
        # For each AP, find which slot pos is in
        # c has the 4 positions; we need to flip the one corresponding to pos
        # Actually, let's just flip the relevant column
        # Determine which column in the AP tuple corresponds to pos
        # aps_arr[ap_indices] gives the 4 positions; find where pos is
        ap_positions = aps_arr[ap_indices]  # shape (len, 4)
        # Find column index where position == pos
        col_idx = np.where(ap_positions == pos, 1, 0)  # shape (len, 4), 1 where match
        # Flip the color at that position
        c_flipped = c.copy()
        c_flipped[np.arange(len(ap_indices)), np.argmax(col_idx, axis=1)] ^= 1
        # Check if flipped AP is monochromatic
        after_mono = (c_flipped[:, 0] == c_flipped[:, 1]) & (c_flipped[:, 1] == c_flipped[:, 2]) & (c_flipped[:, 2] == c_flipped[:, 3])
        # Delta = (new mono count for these APs) - (old mono count for these APs)
        return int(after_mono.sum()) - int(currently_mono.sum())

    # Greedy improvement loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_delta = 0
        best_pos = -1
        # Evaluate all positions
        for pos in range(N):
            d = delta_for_flip(colors, pos)
            if d < best_delta:
                best_delta = d
                best_pos = pos
        if best_pos >= 0 and best_delta < 0:
            colors[best_pos] ^= 1
            best_count += best_delta
            best_colors = colors.copy()
            improved = True

    # If we have time, try a few random restarts with perturbation
    if time.time() < deadline - 10:
        rng = random.Random(42)
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            # Start from best_colors with some random flips
            colors = best_colors.copy()
            num_flips = rng.randint(5, 20)
            flip_positions = rng.sample(range(N), num_flips)
            for p in flip_positions:
                colors[p] ^= 1
            cur_count = count_mono(colors)
            if cur_count < best_count:
                best_count = cur_count
                best_colors = colors.copy()
            # Local search from here
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                best_delta = 0
                best_pos = -1
                for pos in range(N):
                    d = delta_for_flip(colors, pos)
                    if d < best_delta:
                        best_delta = d
                        best_pos = pos
                if best_pos >= 0 and best_delta < 0:
                    colors[best_pos] ^= 1
                    cur_count += best_delta
                    if cur_count < best_count:
                        best_count = cur_count
                        best_colors = colors.copy()
                    improved = True

    return [int(c) for c in best_colors]
