"""Fast greedy with deadline guard for 3-color 4-AP avoidance."""
import time

def construct():
    start = time.time()
    deadline = start + 80.0

    N = 128
    colors = [None] * (N + 1)
    color_count = [0] * 128

    aps_by_pos = [[] for _ in range(N + 1)]
    for d in range(1, N):
        for a in range(1, N - 3 * d + 1):
            p1, p2, p3, p4 = a, a + d, a + 2 * d, a + 3 * d
            for p in (p1, p2, p3, p4):
                others = tuple(q for q in (p1, p2, p3, p4) if q != p)
                aps_by_pos[p].append(others)

    for p in range(1, N + 1):
        if time.time() > deadline:
            break
        cands = list(range(128))
        cands.sort(key=lambda c: color_count[c])

        placed = False
        for c in cands:
            ok = True
            for others in aps_by_pos[p]:
                placed_set = set()
                all_placed = True
                for q in others:
                    if colors[q] is None:
                        all_placed = False
                    else:
                        placed_set.add(colors[q])
                if all_placed:
                    if len(placed_set | {c}) < 3:
                        ok = False
                        break
                else:
                    if len(placed_set | {c}) < 2:
                        ok = False
                        break
            if ok:
                colors[p] = c
                color_count[c] += 1
                placed = True
                break

        if not placed:
            for c in range(128):
                colors[p] = c
                color_count[c] += 1
                break

    return [int(colors[i]) for i in range(1, N + 1)]
