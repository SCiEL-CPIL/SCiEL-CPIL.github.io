import time
import random
import math
import numpy as np


def construct():
    n = 256
    deadline = time.time() + 110  # Safety margin before 120s limit

    # Precompute all 4-term APs (0-indexed)
    aps = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 3 * d
        for a in range(max_a + 1):
            i0 = a - 1
            i1 = i0 + d
            i2 = i0 + 2 * d
            i3 = i0 + 3 * d
            aps.append((i0, i1, i2, i3))

    aps_arr = np.array(aps, dtype=np.int32)
    n_aps = len(aps_arr)

    # For each position, store which APs it belongs to and its role (0,1,2,3)
    pos_aps = [[] for _ in range(n)]
    for idx, (i0, i1, i2, i3) in enumerate(aps):
        pos_aps[i0].append((idx, 0))
        pos_aps[i1].append((idx, 1))
        pos_aps[i2].append((idx, 2))
        pos_aps[i3].append((idx, 3))

    def count_mono(colors):
        """Count monochromatic 4-APs. colors is a numpy array of 0s and 1s."""
        c0 = colors[aps_arr[:, 0]]
        c1 = colors[aps_arr[:, 1]]
        c2 = colors[aps_arr[:, 2]]
        c3 = colors[aps_arr[:, 3]]
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(mono.sum())

    def calc_delta(colors, pos):
        """
        Calculate the change in mono-AP count if we flip position 'pos'.
        colors is a numpy array.
        """
        old_color = colors[pos]
        new_color = 1 - old_color
        delta = 0
        for ap_idx, my_idx in pos_aps[pos]:
            row = aps_arr[ap_idx]
            # Get the other three positions
            other_indices = [row[j] for j in range(4) if j != my_idx]
            other_colors = [colors[oi] for oi in other_indices]

            # Check if currently mono (all four same)
            was_mono = all(c == old_color for c in other_colors)

            # Check if after flip it would be mono (all four same as new_color)
            is_mono_after = all(c == new_color for c in other_colors)

            if was_mono and not is_mono_after:
                delta -= 1
            elif not was_mono and is_mono_after:
                delta += 1

        return delta

    def greedy_local_search(colors, current_count, max_iters=200):
        """
        Greedy local search: repeatedly flip the position that gives the best improvement.
        Returns improved colors and count.
        """
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            best_delta = 0
            best_pos = -1

            # Try all positions, find the one with the most negative delta
            for pos in range(n):
                d = calc_delta(colors, pos)
                if d < best_delta:
                    best_delta = d
                    best_pos = pos

            if best_pos >= 0:
                colors[best_pos] = 1 - colors[best_pos]
                current_count += best_delta
                improved = True
                iters += 1

        return colors, current_count

    def thue_morse(n):
        return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

    def random_init(n):
        return np.array([random.randint(0, 1) for _ in range(n)], dtype=np.int8)

    def block_init(n, block_size):
        """Create a coloring with alternating blocks of 0s and 1s."""
        colors = np.zeros(n, dtype=np.int8)
        for i in range(n):
            colors[i] = (i // block_size) % 2
        return colors

    best_score = float('inf')
    best_colors = None

    # List of initialization strategies
    def generate_initializations():
        inits = []
        # Thue-Morse
        inits.append(thue_morse(n))
        # Random
        for _ in range(3):
            inits.append(random_init(n))
        # Block patterns with various sizes
        for bs in [4, 8, 16, 32, 64]:
            inits.append(block_init(n, bs))
        return inits

    # Process each initialization
    initializations = generate_initializations()
    random.shuffle(initializations)

    for init_colors in initializations:
        if time.time() > deadline - 10:
            break

        colors = init_colors.copy()
        current_count = count_mono(colors)

        # First, apply greedy local search
        colors, current_count = greedy_local_search(colors, current_count)

        if current_count < best_score:
            best_score = current_count
            best_colors = colors.copy()

        # Then, use SA to try to escape local minima
        # Multiple SA phases with decreasing temperature
        T = 3.0
        T_min = 0.05
        cooling_rate = 0.998

        while time.time() < deadline - 5 and T > T_min:
            pos = random.randint(0, n - 1)
            delta = calc_delta(colors, pos)

            if delta <= 0:
                accept = True
            else:
                prob = math.exp(-delta / T)
                accept = random.random() < prob

            if accept:
                colors[pos] = 1 - colors[pos]
                current_count += delta

                if current_count < best_score:
                    best_score = current_count
                    best_colors = colors.copy()

            T *= cooling_rate

            # Periodically run greedy to refine
            if random.random() < 0.01:
                colors, current_count = greedy_local_search(colors, current_count, max_iters=50)
                if current_count < best_score:
                    best_score = current_count
                    best_colors = colors.copy()

        # Final greedy pass
        colors, current_count = greedy_local_search(colors, current_count, max_iters=100)
        if current_count < best_score:
            best_score = current_count
            best_colors = colors.copy()

    # If we have time, do a few more random restarts with SA + greedy
    while time.time() < deadline - 10:
        colors = random_init(n)
        current_count = count_mono(colors)

        # Greedy first
        colors, current_count = greedy_local_search(colors, current_count, max_iters=100)

        if current_count < best_score:
            best_score = current_count
            best_colors = colors.copy()

        # Short SA phase
        T = 2.0
        T_min = 0.1
        cooling_rate = 0.995
        steps = 0
        while time.time() < deadline - 5 and T > T_min and steps < 5000:
            pos = random.randint(0, n - 1)
            delta = calc_delta(colors, pos)

            if delta <= 0 or random.random() < math.exp(-delta / T):
                colors[pos] = 1 - colors[pos]
                current_count += delta

                if current_count < best_score:
                    best_score = current_count
                    best_colors = colors.copy()

            T *= cooling_rate
            steps += 1

            # Occasional greedy refinement
            if steps % 1000 == 0:
                colors, current_count = greedy_local_search(colors, current_count, max_iters=30)
                if current_count < best_score:
                    best_score = current_count
                    best_colors = colors.copy()

    # Final greedy pass on best solution
    final_colors = best_colors.copy()
    final_count = count_mono(final_colors)
    final_colors, final_count = greedy_local_search(final_colors, final_count, max_iters=200)
    if final_count < best_score:
        best_score = final_count
        best_colors = final_colors

    return [int(c) for c in best_colors]
