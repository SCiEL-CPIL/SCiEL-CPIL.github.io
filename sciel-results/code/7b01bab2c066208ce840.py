import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80

    n = 128

    # Precompute all 4-APs
    aps = []
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))

    num_aps = len(aps)

    # For each position, precompute which APs it's in
    pos_to_aps = [[] for _ in range(n)]
    for i, (a, b, c, e) in enumerate(aps):
        pos_to_aps[a].append(i)
        pos_to_aps[b].append(i)
        pos_to_aps[c].append(i)
        pos_to_aps[e].append(i)

    def is_viol(ca, cb, cc, ce):
        return len({ca, cb, cc, ce}) < 3

    def compute_violations(colors):
        viol = [0] * num_aps
        count = 0
        for i, (a, b, c, e) in enumerate(aps):
            if is_viol(colors[a], colors[b], colors[c], colors[e]):
                viol[i] = 1
                count += 1
        return viol, count

    best_overall = None

    for num_colors in range(4, 9):
        if time.time() > deadline:
            break

        remaining = deadline - time.time()
        tc_deadline = min(deadline, time.time() + max(remaining * 0.7, 5))

        restart = 0
        while time.time() < tc_deadline - 0.5:
            random.seed(restart * 7919 + num_colors * 31)
            colors = [random.randint(0, num_colors - 1) for _ in range(n)]

            viol, viol_count = compute_violations(colors)
            best_viol_count = viol_count
            best_colors = colors[:]

            if viol_count == 0:
                return colors

            T = 1.0
            cooling = 0.99985

            while time.time() < tc_deadline - 0.3:
                pos = random.randint(0, n - 1)
                old_color = colors[pos]
                new_color = random.randint(0, num_colors - 1)
                if new_color == old_color:
                    continue

                colors[pos] = new_color

                delta = 0
                for ap_idx in pos_to_aps[pos]:
                    a, b, c, e = aps[ap_idx]
                    was = viol[ap_idx]
                    now = 1 if is_viol(colors[a], colors[b], colors[c], colors[e]) else 0
                    delta += now - was
                    viol[ap_idx] = now

                new_count = viol_count + delta

                if delta <= 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                    viol_count = new_count
                    if viol_count < best_viol_count:
                        best_viol_count = viol_count
                        best_colors = colors[:]
                        if viol_count == 0:
                            return colors
                else:
                    colors[pos] = old_color
                    for ap_idx in pos_to_aps[pos]:
                        a, b, c, e = aps[ap_idx]
                        viol[ap_idx] = 1 if is_viol(colors[a], colors[b], colors[c], colors[e]) else 0
                    viol_count -= delta

                T *= cooling

            if best_viol_count == 0:
                return best_colors

            if best_overall is None or best_viol_count < best_overall[0]:
                best_overall = (best_viol_count, best_colors, num_colors)

            restart += 1

    if best_overall is not None:
        return best_overall[1]

    # Fallback: greedy
    colors = []
    for x in range(1, 129):
        for color in range(128):
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                if x - 1 - 3 * d < 0:
                    break
                cs = {color, colors[x - 1 - d], colors[x - 1 - 2 * d], colors[x - 1 - 3 * d]}
                if len(cs) < 3:
                    valid = False
                    break
            if valid:
                colors.append(color)
                break
    return colors
