import time
import random
import math
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    # Precompute all 4-APs (0-indexed)
    ap_list = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    AP = np.array(ap_list, dtype=np.int32)
    M = AP.shape[0]

    # Build per-position neighbor lists for fast delta computation
    neighbors = [[] for _ in range(n)]
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            idxs = [a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1]
            for i in idxs:
                others = tuple(j for j in idxs if j != i)
                neighbors[i].append(others)

    # Also build AP index lists per position for swap delta
    pos_ap_indices = [[] for _ in range(n)]
    for ap_idx in range(M):
        for pos in AP[ap_idx]:
            pos_ap_indices[pos].append(ap_idx)

    def count_mono(colors):
        c = np.asarray(colors, dtype=np.int8)
        c_ap = c[AP]
        eq = (c_ap[:, 0] == c_ap[:, 1]) & (c_ap[:, 1] == c_ap[:, 2]) & (c_ap[:, 2] == c_ap[:, 3])
        return int(np.count_nonzero(eq))

    def delta_flip(colors, i):
        old_val = colors[i]
        delta = 0
        for (j, k, l) in neighbors[i]:
            if colors[j] == colors[k] == colors[l]:
                if colors[j] == old_val:
                    delta -= 1
                else:
                    delta += 1
        return delta

    def delta_swap(colors, i, j):
        """Delta in mono count if we flip both i and j."""
        affected = set(pos_ap_indices[i]) | set(pos_ap_indices[j])
        delta = 0
        for ap_idx in affected:
            a0, a1, a2, a3 = AP[ap_idx]
            c0 = colors[a0]
            c1 = colors[a1]
            c2 = colors[a2]
            c3 = colors[a3]
            was_mono = (c0 == c1 == c2 == c3)
            if a0 == i or a0 == j: c0 = 1 - c0
            if a1 == i or a1 == j: c1 = 1 - c1
            if a2 == i or a2 == j: c2 = 1 - c2
            if a3 == i or a3 == j: c3 = 1 - c3
            is_mono = (c0 == c1 == c2 == c3)
            if was_mono and not is_mono:
                delta -= 1
            elif not was_mono and is_mono:
                delta += 1
        return delta

    def greedy_polish(colors, use_swap=False):
        """Greedy local search with optional 2-flip swaps."""
        colors = colors[:]
        cur = count_mono(colors)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            order = list(range(n))
            random.shuffle(order)
            for pos in order:
                if time.time() >= deadline - 5.0:
                    break
                d = delta_flip(colors, pos)
                if d < 0:
                    colors[pos] = 1 - colors[pos]
                    cur += d
                    improved = True
            if use_swap:
                for _ in range(20):
                    if time.time() >= deadline - 5.0:
                        break
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i == j:
                        continue
                    d = delta_swap(colors, i, j)
                    if d < 0:
                        colors[i] = 1 - colors[i]
                        colors[j] = 1 - colors[j]
                        cur += d
                        improved = True
        return colors, cur

    def thue_morse():
        return [bin(i).count('1') % 2 for i in range(n)]

    def block_pattern(block_size):
        return [(i // block_size) % 2 for i in range(n)]

    def random_init():
        return [random.randint(0, 1) for _ in range(n)]

    seeds = [
        thue_morse(),
        block_pattern(2),
        block_pattern(4),
        block_pattern(8),
        block_pattern(16),
        random_init(),
        random_init(),
        random_init(),
        random_init(),
    ]

    best_cost = float('inf')
    best_colors = None

    # Phase 1: Multi-start Simulated Annealing with swap moves
    sa_deadline = start_time + 80.0
    for colors in seeds:
        if time.time() >= sa_deadline:
            break

        current_cost = count_mono(colors)
        if current_cost < best_cost:
            best_cost = current_cost
            best_colors = colors[:]

        T = 3.0
        min_T = 0.005
        cool_rate = 0.998
        local_best_cost = current_cost
        local_best_colors = colors[:]
        no_improve = 0
        max_no_improve = 8000

        while time.time() < sa_deadline:
            # Batch of moves
            for _ in range(60):
                # 85% single flip, 15% swap
                if random.random() < 0.85:
                    i = random.randint(0, n - 1)
                    d_cost = delta_flip(colors, i)
                    if d_cost < 0:
                        colors[i] = 1 - colors[i]
                        current_cost += d_cost
                        no_improve = 0
                        if current_cost < local_best_cost:
                            local_best_cost = current_cost
                            local_best_colors = colors[:]
                    else:
                        if T > min_T:
                            arg = -d_cost / T
                            if arg < -50:
                                accept = False
                            else:
                                accept = random.random() < math.exp(arg)
                            if accept:
                                colors[i] = 1 - colors[i]
                                current_cost += d_cost
                                if current_cost < local_best_cost:
                                    local_best_cost = current_cost
                                    local_best_colors = colors[:]
                                    no_improve = 0
                                else:
                                    no_improve += 1
                            else:
                                no_improve += 1
                        else:
                            no_improve += 1
                else:
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i == j:
                        continue
                    d_cost = delta_swap(colors, i, j)
                    if d_cost < 0:
                        colors[i] = 1 - colors[i]
                        colors[j] = 1 - colors[j]
                        current_cost += d_cost
                        no_improve = 0
                        if current_cost < local_best_cost:
                            local_best_cost = current_cost
                            local_best_colors = colors[:]
                    else:
                        if T > min_T:
                            arg = -d_cost / T
                            if arg < -50:
                                accept = False
                            else:
                                accept = random.random() < math.exp(arg)
                            if accept:
                                colors[i] = 1 - colors[i]
                                colors[j] = 1 - colors[j]
                                current_cost += d_cost
                                if current_cost < local_best_cost:
                                    local_best_cost = current_cost
                                    local_best_colors = colors[:]
                                    no_improve = 0
                                else:
                                    no_improve += 1
                            else:
                                no_improve += 1
                        else:
                            no_improve += 1

            T *= cool_rate
            if T < min_T:
                T = min_T

            # Restart if stuck
            if no_improve > max_no_improve:
                colors = local_best_colors[:]
                current_cost = local_best_cost
                T = max(T * 3.0, 1.5)
                no_improve = 0
                # Perturb
                for _ in range(5):
                    p = random.randint(0, n - 1)
                    colors[p] = 1 - colors[p]
                current_cost = count_mono(colors)
                # Quick polish
                colors, current_cost = greedy_polish(colors, use_swap=False)
                if current_cost < local_best_cost:
                    local_best_cost = current_cost
                    local_best_colors = colors[:]

        if local_best_cost < best_cost:
            best_cost = local_best_cost
            best_colors = local_best_colors[:]

    # Phase 2: Coordinate descent polish with swaps
    polish_deadline = start_time + 110.0
    if best_colors is not None and time.time() < polish_deadline:
        colors = best_colors[:]
        current_cost = best_cost

        for pass_num in range(30):
            if time.time() >= polish_deadline:
                break
            improved = False
            if pass_num % 3 == 0:
                order = list(range(n))
            elif pass_num % 3 == 1:
                order = list(range(n - 1, -1, -1))
            else:
                order = list(range(n))
                random.shuffle(order)

            for i in order:
                if time.time() >= polish_deadline:
                    break
                d_cost = delta_flip(colors, i)
                if d_cost < 0:
                    colors[i] = 1 - colors[i]
                    current_cost += d_cost
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors[:]

            # Try swaps
            for _ in range(30):
                if time.time() >= polish_deadline:
                    break
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                d_cost = delta_swap(colors, i, j)
                if d_cost < 0:
                    colors[i] = 1 - colors[i]
                    colors[j] = 1 - colors[j]
                    current_cost += d_cost
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors[:]

            if not improved:
                break

    # Final verification
    if best_colors is not None:
        actual = count_mono(best_colors)
        if actual < best_cost:
            best_cost = actual

    return best_colors
