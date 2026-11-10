import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Precompute all 4-term arithmetic progressions
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))

    total_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)

    # Precompute neighbors for fast delta calculation
    neighbors = [[] for _ in range(N)]
    for i1, i2, i3, i4 in aps:
        neighbors[i1].append((i2, i3, i4))
        neighbors[i2].append((i1, i3, i4))
        neighbors[i3].append((i1, i2, i4))
        neighbors[i4].append((i1, i2, i3))

    # Convert to numpy arrays for faster access
    neighbors_np = []
    for i in range(N):
        if neighbors[i]:
            neighbors_np.append(np.array(neighbors[i], dtype=np.int32))
        else:
            neighbors_np.append(np.empty((0, 3), dtype=np.int32))

    def count_mono(colors):
        """Count monochromatic APs using numpy vectorization"""
        c0 = colors[aps_arr[:, 0]]
        c1 = colors[aps_arr[:, 1]]
        c2 = colors[aps_arr[:, 2]]
        c3 = colors[aps_arr[:, 3]]
        return int(np.sum((c0 == c1) & (c0 == c2) & (c0 == c3)))

    def delta_flip(colors, i):
        """Compute the change in monochromatic AP count if we flip position i"""
        old_c = colors[i]
        new_c = 1 - old_c
        others = neighbors_np[i]
        if len(others) == 0:
            return 0
        cols = colors[others]
        c0, c1, c2 = cols[:, 0], cols[:, 1], cols[:, 2]
        all_eq = (c0 == c1) & (c1 == c2)
        val = c0
        was_mono = all_eq & (val == old_c)
        becomes_mono = all_eq & (val == new_c)
        return int(np.sum(becomes_mono) - np.sum(was_mono))

    # Generate diverse initializations
    def generate_initializations():
        inits = []
        # 1. Pure random
        inits.append(np.random.randint(0, 2, size=N))
        # 2. Block pattern with period 2
        inits.append(np.array([(i // 2) % 2 for i in range(N)], dtype=np.int32))
        # 3. Block pattern with period 4
        inits.append(np.array([(i // 4) % 2 for i in range(N)], dtype=np.int32))
        # 4. Striped pattern
        inits.append(np.array([i % 2 for i in range(N)], dtype=np.int32))
        # 5. Thue-Morse-like pattern
        tm = np.array([bin(i).count('1') % 2 for i in range(N)], dtype=np.int32)
        inits.append(tm)
        # 6. Block pattern with period 8
        inits.append(np.array([(i // 8) % 2 for i in range(N)], dtype=np.int32))
        # 7. Block pattern with period 16
        inits.append(np.array([(i // 16) % 2 for i in range(N)], dtype=np.int32))
        # 8. Another random
        inits.append(np.random.randint(0, 2, size=N))
        # 9. Random with bias
        inits.append(np.random.randint(0, 2, size=N, p=[0.6, 0.4]))
        return inits

    # Evaluate initializations and pick the best few
    init_scores = []
    for init in generate_initializations():
        score = count_mono(init)
        init_scores.append((score, init))

    # Sort by score, take top 5 for multi-start
    init_scores.sort(key=lambda x: x[0])
    best_inits = [x[1] for x in init_scores[:5]]

    best_score = min(x[0] for x in init_scores)
    best_coloring = best_inits[0].copy()

    # Multi-start simulated annealing
    for start_idx, initial_colors in enumerate(best_inits):
        if time.time() > deadline - 5.0:
            break

        colors = initial_colors.copy()
        current_score = count_mono(colors)

        if current_score < best_score:
            best_score = current_score
            best_coloring = colors.copy()

        # Simulated annealing parameters
        T = 3.0
        cooling_rate = 0.99995
        min_T = 0.05

        # Run SA until temperature drops or time runs out
        while time.time() < deadline - 5.0 and T > min_T:
            i = random.randint(0, N - 1)
            old_c = colors[i]
            new_c = 1 - old_c

            delta = delta_flip(colors, i)

            if delta < 0 or random.random() < np.exp(-delta / T):
                colors[i] = new_c
                current_score += delta

                if current_score < best_score:
                    best_score = current_score
                    best_coloring = colors.copy()

            T *= cooling_rate

    return best_coloring.tolist()
# EVOLVE-BLOCK-END
