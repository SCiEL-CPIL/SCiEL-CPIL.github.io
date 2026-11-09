import numpy as np
import time
import random

def construct():
    n = 128
    deadline = time.time() + 55.0

    # Precompute all non-constant 4-APs
    ap_list = []
    for a in range(n):
        for d in range(1, (n - a) // 3 + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))

    num_aps = len(ap_list)
    ap_arr = np.array(ap_list, dtype=np.int32)

    # For each position, which APs does it appear in?
    pos_aps = [[] for _ in range(n)]
    for i, ap in enumerate(ap_list):
        for pos in ap:
            pos_aps[pos].append(i)
    pos_aps_arr = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in pos_aps]

    def distinct4(a, b, c, d):
        count = 1
        if b != a:
            count += 1
        if c != a and c != b:
            count += 1
        if d != a and d != b and d != c:
            count += 1
        return count

    def compute_bad_mask(colors):
        c0 = colors[ap_arr[:, 0]]
        c1 = colors[ap_arr[:, 1]]
        c2 = colors[ap_arr[:, 2]]
        c3 = colors[ap_arr[:, 3]]
        d1 = (c0 != c1)
        d2 = (c0 != c2) & (c1 != c2)
        d3 = (c0 != c3) & (c1 != c3) & (c2 != c3)
        distinct = 1 + d1.astype(np.int8) + d2.astype(np.int8) + d3.astype(np.int8)
        bad = distinct < 3
        return bad, np.sum(bad)

    # Try k colors
    for k in [3, 4, 5, 6]:
        if time.time() > deadline - 5.0:
            break

        for restart in range(30):
            if time.time() > deadline - 5.0:
                break

            # Initialize with modular pattern
            colors = np.array([i % k for i in range(n)], dtype=np.int32)
            bad_mask, bad_count = compute_bad_mask(colors)

            if bad_count == 0:
                return colors.tolist()

            # Local search with simulated annealing
            temp = 2.0
            cooling = 0.999
            max_iter = 6000
            best_local = bad_count
            best_local_colors = colors.copy()
            no_improve = 0

            for it in range(max_iter):
                if time.time() > deadline - 5.0:
                    break
                if bad_count == 0:
                    return colors.tolist()

                pos = random.randint(0, n - 1)
                old_color = int(colors[pos])
                new_color = random.randint(0, k - 1)
                if new_color == old_color:
                    continue

                # Compute delta in bad count
                delta = 0
                for ap_idx in pos_aps_arr[pos]:
                    ap = ap_arr[ap_idx]
                    c0 = int(colors[ap[0]])
                    c1 = int(colors[ap[1]])
                    c2 = int(colors[ap[2]])
                    c3 = int(colors[ap[3]])

                    if ap[0] == pos:
                        od = distinct4(old_color, c1, c2, c3)
                        nd = distinct4(new_color, c1, c2, c3)
                    elif ap[1] == pos:
                        od = distinct4(c0, old_color, c2, c3)
                        nd = distinct4(c0, new_color, c2, c3)
                    elif ap[2] == pos:
                        od = distinct4(c0, c1, old_color, c3)
                        nd = distinct4(c0, c1, new_color, c3)
                    else:
                        od = distinct4(c0, c1, c2, old_color)
                        nd = distinct4(c0, c1, c2, new_color)

                    ob = od < 3
                    nb = nd < 3

                    if ob and not nb:
                        delta -= 1
                    elif not ob and nb:
                        delta += 1

                nb_count = bad_count + delta

                if delta < 0 or random.random() < np.exp(-delta / temp):
                    colors[pos] = new_color
                    bad_count = nb_count
                    if bad_count < best_local:
                        best_local = bad_count
                        best_local_colors = colors.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                    temp *= cooling
                else:
                    no_improve += 1

                if no_improve > 1500:
                    break

            if bad_count == 0:
                return colors.tolist()

    # Fallback: return a valid coloring using all distinct colors
    return list(range(n))
