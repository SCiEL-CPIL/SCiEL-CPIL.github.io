import time
import random
import numpy as np

N = 128

def construct():
    start_time = time.time()
    deadline = start_time + 80

    # Precompute all 4-term APs
    aps = []
    pos_to_aps = [[] for _ in range(N)]

    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            p = (a, a + d, a + 2*d, a + 3*d)
            aps.append(p)
            for idx in p:
                pos_to_aps[idx].append(len(aps) - 1)

    num_aps = len(aps)

    def distinct4(c0, c1, c2, c3):
        return len({c0, c1, c2, c3})

    # Try K = 3, 4, 5
    for K in [3, 4, 5]:
        if time.time() > deadline - 3:
            break

        best_colors = None
        best_bad = float('inf')

        for trial in range(15):
            if time.time() > deadline - 3:
                break

            colors = np.random.randint(0, K, N)

            # Compute initial bad count
            bad_flags = np.zeros(num_aps, dtype=bool)
            for i in range(num_aps):
                p = aps[i]
                bad_flags[i] = distinct4(colors[p[0]], colors[p[1]], colors[p[2]], colors[p[3]]) < 3
            curr_bad = int(np.sum(bad_flags))

            if curr_bad < best_bad:
                best_bad = curr_bad
                best_colors = colors.copy()

            if best_bad == 0:
                break

            # Local search
            stuck = 0
            while stuck < 300 and time.time() < deadline - 3:
                pos = random.randint(0, N - 1)
                old_c = int(colors[pos])

                best_c = old_c
                best_local = float('inf')

                for c in range(K):
                    if c == old_c:
                        continue
                    colors[pos] = c
                    local_bad = 0
                    for ap_idx in pos_to_aps[pos]:
                        p = aps[ap_idx]
                        if distinct4(colors[p[0]], colors[p[1]], colors[p[2]], colors[p[3]]) < 3:
                            local_bad += 1
                    if local_bad < best_local:
                        best_local = local_bad
                        best_c = c
                    if best_local == 0:
                        break

                if best_c != old_c:
                    colors[pos] = best_c
                    stuck = 0
                    for ap_idx in pos_to_aps[pos]:
                        p = aps[ap_idx]
                        new_bad = distinct4(colors[p[0]], colors[p[1]], colors[p[2]], colors[p[3]]) < 3
                        if new_bad != bad_flags[ap_idx]:
                            curr_bad += 1 if new_bad else -1
                            bad_flags[ap_idx] = new_bad
                    if curr_bad < best_bad:
                        best_bad = curr_bad
                        best_colors = colors.copy()
                        if best_bad == 0:
                            break
                else:
                    stuck += 1

            if best_bad == 0:
                break

        if best_bad == 0:
            return list(best_colors)

    # Fallback: use 128 colors
    return list(range(N))
