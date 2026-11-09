import time
import random
import math
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 110.0
    max_d = (n - 3) // 3

    # Precompute all 4-APs
    ap_list = []
    for d in range(1, max_d + 1):
        for a in range(n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    AP = np.array(ap_list, dtype=np.int32)
    M = AP.shape[0]

    # Precompute neighbors: for each index i, list of the other 3 indices in each AP containing i
    neighbors = [[] for _ in range(n)]
    for d in range(1, max_d + 1):
        for a in range(n - 3 * d):
            idxs = [a, a + d, a + 2 * d, a + 3 * d]
            for i in idxs:
                others = tuple(j for j in idxs if j != i)
                neighbors[i].append(others)

    def count_mono(colors):
        c = np.array(colors)
        c_ap = c[AP]
        eq = (c_ap[:, 0] == c_ap[:, 1]) & (c_ap[:, 1] == c_ap[:, 2]) & (c_ap[:, 2] == c_ap[:, 3])
        return int(np.count_nonzero(eq))

    def delta_flip(colors, i):
        """Compute change in mono-AP count if we flip color at position i."""
        old_val = colors[i]
        delta = 0
        for (j, k, l) in neighbors[i]:
            vj = colors[j]
            vk = colors[k]
            vl = colors[l]
            if vj == vk == vl:
                if vj == old_val:
                    delta -= 1
                else:
                    delta += 1
        return delta

    # Generate diverse initial colorings
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
    ]

    best_cost = float('inf')
    best_colors = None

    # Phase 1: Multi-start SA
    sa_deadline = time.time() + 80.0
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
            # Batch of moves to reduce overhead
            for _ in range(50):
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
                        # Use math.exp for speed; clamp argument to avoid underflow issues
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

            T *= cool_rate
            if T < min_T:
                T = min_T

            # Restart if stuck
            if no_improve > max_no_improve:
                T = max(T * 3.0, 1.5)
                no_improve = 0

        if local_best_cost < best_cost:
            best_cost = local_best_cost
            best_colors = local_best_colors[:]

    # Phase 2: Coordinate descent polish
    polish_deadline = time.time() + (110.0 - 80.0)
    if best_colors is not None and time.time() < polish_deadline:
        colors = best_colors[:]
        current_cost = best_cost

        for pass_num in range(20):
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

            if not improved:
                break

    return best_colors
