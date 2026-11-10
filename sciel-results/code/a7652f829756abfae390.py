import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit

    # 1. Precompute all 4-term arithmetic progressions
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))

    aps_arr = np.array(aps, dtype=np.int32)
    num_aps = len(aps_arr)

    # 2. Precompute for each position, the list of (ap_index, position_in_ap)
    # This allows fast delta computation
    pos_aps = [[] for _ in range(N)]
    for ap_idx in range(num_aps):
        i1, i2, i3, i4 = aps_arr[ap_idx]
        pos_aps[i1].append((ap_idx, 0))
        pos_aps[i2].append((ap_idx, 1))
        pos_aps[i3].append((ap_idx, 2))
        pos_aps[i4].append((ap_idx, 3))

    # Convert to arrays for faster access
    pos_aps_arr = []
    for i in range(N):
        if pos_aps[i]:
            arr = np.array(pos_aps[i], dtype=np.int32)
            pos_aps_arr.append(arr)
        else:
            pos_aps_arr.append(np.empty((0, 2), dtype=np.int32))

    # Precompute the "other 3" indices for each position in each AP
    # For fast delta: when flipping position i, for each AP containing i,
    # check if the other 3 are all equal to old/new color
    other_indices = [[] for _ in range(N)]
    for i1, i2, i3, i4 in aps:
        other_indices[i1].append((i2, i3, i4))
        other_indices[i2].append((i1, i3, i4))
        other_indices[i3].append((i1, i2, i4))
        other_indices[i4].append((i1, i2, i3))

    other_indices_arr = []
    for i in range(N):
        if other_indices[i]:
            other_indices_arr.append(np.array(other_indices[i], dtype=np.int32))
        else:
            other_indices_arr.append(np.empty((0, 3), dtype=np.int32))

    def count_mono(colors):
        """Count monochromatic 4-APs using vectorized operations."""
        c1 = colors[aps_arr[:, 0]]
        c2 = colors[aps_arr[:, 1]]
        c3 = colors[aps_arr[:, 2]]
        c4 = colors[aps_arr[:, 3]]
        return int(np.sum((c1 == c2) & (c2 == c3) & (c3 == c4)))

    def delta_flip(colors, i):
        """Compute score change when flipping position i."""
        others = other_indices_arr[i]
        if len(others) == 0:
            return 0
        cols = colors[others]
        c0, c1, c2 = cols[:, 0], cols[:, 1], cols[:, 2]
        all_eq = (c0 == c1) & (c1 == c2)
        old_c = colors[i]
        new_c = 1 - old_c
        was_mono = np.sum(all_eq & (c0 == old_c))
        becomes_mono = np.sum(all_eq & (c0 == new_c))
        return int(becomes_mono - was_mono)

    def delta_multi_flip(colors, indices):
        """Compute score change when flipping multiple positions."""
        old_colors = colors[indices].copy()
        colors[indices] = 1 - old_colors
        new_score = count_mono(colors)
        old_score = count_mono(colors)  # We need old score before flip
        # Actually, let's compute differently
        colors[indices] = old_colors
        old_score = count_mono(colors)
        colors[indices] = 1 - old_colors
        new_score = count_mono(colors)
        colors[indices] = old_colors
        return new_score - old_score

    def polish(colors, score, time_limit=3.0):
        """Perform exact local search: only accept improving moves."""
        polish_start = time.time()
        improved = True
        while improved and time.time() < polish_start + time_limit:
            improved = False
            for i in range(N):
                if time.time() > polish_start + time_limit:
                    break
                delta = delta_flip(colors, i)
                if delta < 0:
                    colors[i] = 1 - colors[i]
                    score += delta
                    improved = True
        return score

    # 3. Generate diverse initial patterns
    # Use patterns known to reduce APs
    patterns = []

    # Block patterns with various sizes
    for block_size in [2, 3, 4, 5, 6, 8, 10, 12, 16]:
        patterns.append(np.array([(i // block_size) % 2 for i in range(N)], dtype=np.int8))
        patterns.append(np.array([1 - ((i // block_size) % 2) for i in range(N)], dtype=np.int8))

    # Radix-2 based (Thue-Morse-like)
    patterns.append(np.array([bin(i).count('1') % 2 for i in range(N)], dtype=np.int8))
    patterns.append(np.array([1 - (bin(i).count('1') % 2) for i in range(N)], dtype=np.int8))

    # Radix-4 based
    patterns.append(np.array([(i % 4) < 2 for i in range(N)], dtype=np.int8))
    patterns.append(np.array([(i % 4) >= 2 for i in range(N)], dtype=np.int8))

    # Hybrid patterns
    patterns.append(np.array([((i // 2) % 2) if (i // 8) % 2 == 0 else ((i // 3) % 2) for i in range(N)], dtype=np.int8))
    patterns.append(np.array([((i // 3) % 2) if (i // 16) % 2 == 0 else ((i // 4) % 2) for i in range(N)], dtype=np.int8))
    patterns.append(np.array([((i // 4) % 2) if (i % 8) < 4 else ((i // 2) % 2) for i in range(N)], dtype=np.int8))

    # Random patterns with structure
    for seed in [42, 123, 456, 789, 1024]:
        rng = np.random.RandomState(seed)
        # Generate block-based random pattern
        block_size = rng.randint(2, 8)
        pat = np.zeros(N, dtype=np.int8)
        for start in range(0, N, block_size):
            end = min(start + block_size, N)
            pat[start:end] = rng.randint(0, 2) - 1
        patterns.append(pat)

    # Compute initial scores for all patterns
    pattern_scores = []
    for pat in patterns:
        s = count_mono(pat)
        pattern_scores.append(s)

    # Sort patterns by score
    pattern_order = np.argsort(pattern_scores)

    best_score = float('inf')
    best_coloring = None

    # 4. Main search loop
    num_restarts = 0
    while time.time() < deadline - 15.0:
        # Choose a pattern (cycle through sorted patterns)
        base_idx = pattern_order[num_restarts % len(pattern_order)]
        colors = patterns[base_idx].copy()

        # Apply random perturbations to diversify
        num_perturb = random.randint(2, 8)
        perturb_indices = random.sample(range(N), num_perturb)
        for idx in perturb_indices:
            colors[idx] = 1 - colors[idx]

        current_score = count_mono(colors)

        if current_score < best_score:
            best_score = current_score
            best_coloring = colors.copy()

        # Simulated Annealing with adaptive parameters
        T = 0.5
        alpha = 0.999
        min_T = 0.01
        stagnation_count = 0
        last_improve_score = current_score

        # Time limit for this SA run
        remaining = deadline - 15.0 - time.time()
        sa_deadline = time.time() + max(3.0, min(10.0, remaining / 2.0))

        while T > min_T and time.time() < sa_deadline:
            # Decide move type based on stagnation
            if stagnation_count > 30:
                move_type = random.random()
                if move_type < 0.3:
                    move_type = 'block'
                elif move_type < 0.5:
                    move_type = 'multi'
                else:
                    move_type = 'flip'
            else:
                move_type = random.random()
                if move_type < 0.1:
                    move_type = 'block'
                elif move_type < 0.2:
                    move_type = 'multi'
                else:
                    move_type = 'flip'

            if move_type == 'flip':
                i = random.randint(0, N - 1)
                delta = delta_flip(colors, i)
                if delta < 0 or random.random() < np.exp(-delta / T):
                    colors[i] = 1 - colors[i]
                    current_score += delta
                    if current_score < last_improve_score:
                        last_improve_score = current_score
                        stagnation_count = 0
                        if current_score < best_score:
                            best_score = current_score
                            best_coloring = colors.copy()
                    else:
                        stagnation_count += 1
                else:
                    stagnation_count += 1

            elif move_type == 'multi':
                # Flip 2-3 random positions
                k = random.randint(2, 3)
                indices = random.sample(range(N), k)
                old_colors = colors[indices].copy()
                colors[indices] = 1 - old_colors
                new_score = count_mono(colors)
                delta = new_score - current_score
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_score = new_score
                    if current_score < last_improve_score:
                        last_improve_score = current_score
                        stagnation_count = 0
                        if current_score < best_score:
                            best_score = current_score
                            best_coloring = colors.copy()
                    else:
                        stagnation_count += 1
                else:
                    colors[indices] = old_colors
                    stagnation_count += 1

            else:  # block
                # Flip a contiguous block of positions
                block_len = random.randint(2, 6)
                start = random.randint(0, N - block_len)
                indices = list(range(start, start + block_len))
                old_colors = colors[indices].copy()
                colors[indices] = 1 - old_colors
                new_score = count_mono(colors)
                delta = new_score - current_score
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_score = new_score
                    if current_score < last_improve_score:
                        last_improve_score = current_score
                        stagnation_count = 0
                        if current_score < best_score:
                            best_score = current_score
                            best_coloring = colors.copy()
                    else:
                        stagnation_count += 1
                else:
                    colors[indices] = old_colors
                    stagnation_count += 1

            T *= alpha

        num_restarts += 1

    # 5. Final polish
    if best_coloring is not None:
        final_colors = best_coloring.copy()
        final_score = polish(final_colors, best_score, time_limit=8.0)
        if final_score < best_score:
            best_score = final_score
            best_coloring = final_colors.copy()

        # Try a few more polish passes from perturbed versions
        polish_deadline = time.time() + 10.0
        for _ in range(8):
            if time.time() > polish_deadline:
                break
            perturbed = best_coloring.copy()
            num_pert = random.randint(1, 4)
            pert_idx = random.sample(range(N), num_pert)
            for idx in pert_idx:
                perturbed[idx] = 1 - perturbed[idx]
            pert_score = count_mono(perturbed)
            polished = polish(perturbed, pert_score, time_limit=2.0)
            if polished < best_score:
                best_score = polished
                best_coloring = perturbed.copy()

    return best_coloring.tolist()
# EVOLVE-BLOCK-END
