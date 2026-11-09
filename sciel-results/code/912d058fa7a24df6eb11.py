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

    # Precompute for each position i: the AP indices containing i, and the positions of the other 3 elements
    # ap_indices[i] = list of AP row indices where position i appears
    # ap_others[i] = for each AP containing i, the 3 other positions (as a flat array)
    ap_indices = [[] for _ in range(n)]
    ap_others_flat = [[] for _ in range(n)]
    for ap_idx in range(M):
        a, b, c, d = AP[ap_idx]
        for pos in (a, b, c, d):
            ap_indices[pos].append(ap_idx)
            others = [x for x in (a, b, c, d) if x != pos]
            ap_others_flat[pos].extend(others)

    # Convert to numpy arrays for fast access
    ap_indices_arr = [np.array(x, dtype=np.int32) if x else np.array([], dtype=np.int32) for x in ap_indices]
    ap_others_arr = [np.array(x, dtype=np.int32) if x else np.array([], dtype=np.int32) for x in ap_others_flat]

    def count_mono(colors):
        c = np.array(colors)
        c_ap = c[AP]
        eq = (c_ap[:, 0] == c_ap[:, 1]) & (c_ap[:, 1] == c_ap[:, 2]) & (c_ap[:, 2] == c_ap[:, 3])
        return int(np.count_nonzero(eq))

    def delta_flip_vec(c_arr, i):
        """Vectorized: compute change in mono-AP count if we flip color at position i.
        c_arr is a numpy array of colors (0/1)."""
        ap_idx = ap_indices_arr[i]
        if len(ap_idx) == 0:
            return 0
        others = ap_others_arr[i]
        # Get colors of the other 3 positions for each AP containing i
        # others is a flat array of length 3 * len(ap_idx), grouped in triples
        others_colors = c_arr[others]
        # Reshape into (num_aps, 3)
        oc = others_colors.reshape(-1, 3)
        old_val = c_arr[i]
        # Check if all 3 others are the same color
        all_same = (oc[:, 0] == oc[:, 1]) & (oc[:, 1] == oc[:, 2])
        # Count: if all_same and color == old_val, delta -= 1; if all_same and color != old_val, delta += 1
        same_as_old = all_same & (oc[:, 0] == old_val)
        diff_from_old = all_same & (oc[:, 0] != old_val)
        delta = -int(np.sum(same_as_old)) + int(np.sum(diff_from_old))
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

    # Phase 1: Multi-start SA with vectorized delta
    sa_deadline = time.time() + 85.0
    for seed_idx, colors in enumerate(seeds):
        if time.time() >= sa_deadline:
            break

        c_arr = np.array(colors, dtype=np.int8)
        current_cost = count_mono(c_arr)
        if current_cost < best_cost:
            best_cost = current_cost
            best_colors = colors[:]

        T = 3.0
        min_T = 0.005
        cool_rate = 0.997
        local_best_cost = current_cost
        local_best_colors = colors[:]
        no_improve = 0
        max_no_improve = 6000

        while time.time() < sa_deadline:
            # Batch of moves - larger batches since delta is fast
            batch_size = 200
            for _ in range(batch_size):
                i = random.randint(0, n - 1)
                d_cost = delta_flip_vec(c_arr, i)

                if d_cost < 0:
                    c_arr[i] = 1 - c_arr[i]
                    current_cost += d_cost
                    no_improve = 0
                    if current_cost < local_best_cost:
                        local_best_cost = current_cost
                        local_best_colors = c_arr.astype(int).tolist()
                else:
                    if T > min_T:
                        arg = -d_cost / T
                        if arg < -50:
                            accept = False
                        else:
                            accept = random.random() < math.exp(arg)
                        if accept:
                            c_arr[i] = 1 - c_arr[i]
                            current_cost += d_cost
                            if current_cost < local_best_cost:
                                local_best_cost = current_cost
                                local_best_colors = c_arr.astype(int).tolist()
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

    # Phase 2: Coordinate descent polish with vectorized delta
    polish_deadline = time.time() + (110.0 - 85.0)
    if best_colors is not None and time.time() < polish_deadline:
        c_arr = np.array(best_colors, dtype=np.int8)
        current_cost = best_cost

        for pass_num in range(50):
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
                d_cost = delta_flip_vec(c_arr, i)
                if d_cost < 0:
                    c_arr[i] = 1 - c_arr[i]
                    current_cost += d_cost
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = c_arr.astype(int).tolist()

            if not improved:
                break

    return best_colors
