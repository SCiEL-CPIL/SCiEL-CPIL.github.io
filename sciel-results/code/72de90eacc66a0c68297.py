import time
import random

def construct():
    deadline = time.time() + 80
    n = 128

    # Precompute all 4-term APs (0-indexed)
    aps = []
    ap_end_pos = {}
    for a in range(n):
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            idx = len(aps)
            aps.append((a, b, c, e))
            if e not in ap_end_pos:
                ap_end_pos[e] = []
            ap_end_pos[e].append(idx)

    def count_violations_fast(colors):
        count = 0
        for (a, b, c, e) in aps:
            ca = colors[a]
            cb = colors[b]
            cc = colors[c]
            ce = colors[e]
            distinct = 1
            if cb != ca:
                distinct += 1
            if cc != ca and cc != cb:
                distinct += 1
            if ce != ca and ce != cb and ce != cc:
                distinct += 1
            if distinct < 3:
                count += 1
        return count

    def is_valid(colors):
        for (a, b, c, e) in aps:
            ca = colors[a]
            cb = colors[b]
            cc = colors[c]
            ce = colors[e]
            distinct = 1
            if cb != ca:
                distinct += 1
            if cc != ca and cc != cb:
                distinct += 1
            if ce != ca and ce != cb and ce != cc:
                distinct += 1
            if distinct < 3:
                return False
        return True

    # Greedy baseline
    def greedy_coloring():
        colors = []
        for x in range(n):
            for color in range(128):
                valid = True
                for idx in ap_end_pos.get(x, []):
                    a, b, c, e = aps[idx]
                    ca = colors[a]
                    cb = colors[b]
                    cc = colors[c]
                    distinct = 1
                    if cb != ca:
                        distinct += 1
                    if cc != ca and cc != cb:
                        distinct += 1
                    if color != ca and color != cb and color != cc:
                        distinct += 1
                    if distinct < 3:
                        valid = False
                        break
                if valid:
                    colors.append(color)
                    break
            else:
                colors.append(0)
        return colors

    baseline = greedy_coloring()
    best_colors = baseline
    best_num_colors = len(set(baseline))

    # Local search to find a valid 3-coloring
    for attempt in range(20):
        if time.time() > deadline - 15:
            break

        colors = [random.randint(0, 2) for _ in range(n)]

        for iteration in range(500):
            if time.time() > deadline - 15:
                break

            v = count_violations_fast(colors)
            if v == 0:
                nc = len(set(colors))
                if nc < best_num_colors:
                    best_num_colors = nc
                    best_colors = colors[:]
                break

            pos = random.randint(0, n - 1)
            old_color = colors[pos]

            best_new = old_color
            best_v = v

            for nc in range(3):
                if nc == old_color:
                    continue
                colors[pos] = nc
                v2 = count_violations_fast(colors)
                if v2 < best_v:
                    best_v = v2
                    best_new = nc
                colors[pos] = old_color

            if best_new != old_color:
                colors[pos] = best_new
            else:
                colors[pos] = random.randint(0, 2)

        if is_valid(colors):
            nc = len(set(colors))
            if nc < best_num_colors:
                best_num_colors = nc
                best_colors = colors[:]

    return best_colors
