import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 128

    aps = []
    for d in range(1, 43):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))

    pos_aps = [[] for _ in range(n)]
    for idx, (a, b, c, d) in enumerate(aps):
        pos_aps[a].append(idx)
        pos_aps[b].append(idx)
        pos_aps[c].append(idx)
        pos_aps[d].append(idx)

    for k in [3, 4, 5, 6, 7, 8]:
        if time.time() > deadline - 5:
            break
        result = local_search(n, aps, pos_aps, k, deadline)
        if result is not None:
            return result

    return list(range(128))

def local_search(n, aps, pos_aps, k, deadline):
    random.seed(42)

    for restart in range(2000):
        if time.time() > deadline - 3:
            break

        colors = [(i % k) for i in range(n)]
        is_viol = [False] * len(aps)
        total_v = 0

        for idx, (a, b, c, d) in enumerate(aps):
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                is_viol[idx] = True
                total_v += 1

        if total_v == 0:
            return colors

        for step in range(500000):
            if time.time() > deadline - 3:
                break
            if total_v == 0:
                return colors

            i = random.randint(0, n - 1)
            old_c = colors[i]
            new_c = random.randint(0, k - 1)
            if new_c == old_c:
                continue

            delta = 0
            for ap_idx in pos_aps[i]:
                a, b, c, d = aps[ap_idx]
                if i == a:
                    ns = {new_c, colors[b], colors[c], colors[d]}
                elif i == b:
                    ns = {colors[a], new_c, colors[c], colors[d]}
                elif i == c:
                    ns = {colors[a], colors[b], new_c, colors[d]}
                else:
                    ns = {colors[a], colors[b], colors[c], new_c}

                is_v = len(ns) < 3
                if is_viol[ap_idx]:
                    if not is_v:
                        delta -= 1
                else:
                    if is_v:
                        delta += 1
                is_viol[ap_idx] = is_v

            colors[i] = new_c
            total_v += delta

        if total_v == 0:
            return colors

    return None
