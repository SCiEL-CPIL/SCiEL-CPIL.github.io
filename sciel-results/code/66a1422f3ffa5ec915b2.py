import random
import time
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 130

    n = 16
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_idx[(i, j)] = idx
            idx += 1

    k4_edges = []
    for a, b, c, d in itertools.combinations(range(n), 4):
        edges = [edge_idx[(a, b)], edge_idx[(a, c)], edge_idx[(a, d)],
                 edge_idx[(b, c)], edge_idx[(b, d)], edge_idx[(c, d)]]
        k4_edges.append(edges)

    num_k4 = len(k4_edges)
    edge_to_k4s = [[] for _ in range(120)]
    for k4i, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k4i)

    best_colors = None
    best_bad = float('inf')
    rng = random.Random(42)

    for attempt in range(400):
        if time.time() > deadline - 5:
            break

        palette_size = 13 if attempt < 250 else (14 if attempt < 350 else 15)

        colors = [rng.randint(0, palette_size - 1) for _ in range(120)]

        bad = 0
        k4_cnt = [None] * num_k4
        for k4i, edges in enumerate(k4_edges):
            cnt = {}
            for e in edges:
                c = colors[e]
                cnt[c] = cnt.get(c, 0) + 1
            k4_cnt[k4i] = cnt
            if len(cnt) <= 4:
                bad += 1

        if bad == 0:
            best_colors = colors[:]
            best_bad = 0
            break

        if bad < best_bad:
            best_bad = bad
            best_colors = colors[:]

        T = 0.5
        for step in range(4000):
            if time.time() > deadline - 2:
                break

            e = rng.randint(0, 119)
            old_c = colors[e]
            new_c = rng.randint(0, palette_size - 1)
            if new_c == old_c:
                continue

            delta = 0
            for k4i in edge_to_k4s[e]:
                cnt = k4_cnt[k4i]
                old_d = len(cnt)
                cnt[old_c] -= 1
                if cnt[old_c] == 0:
                    del cnt[old_c]
                cnt[new_c] = cnt.get(new_c, 0) + 1
                new_d = len(cnt)
                if old_d <= 4 and new_d > 4:
                    delta -= 1
                elif old_d > 4 and new_d <= 4:
                    delta += 1

            if delta <= 0:
                colors[e] = new_c
                bad += delta
                if bad < best_bad:
                    best_bad = bad
                    best_colors = colors[:]
                if bad == 0:
                    break
            elif rng.random() < math.exp(-delta / T):
                colors[e] = new_c
                bad += delta
            else:
                for k4i in edge_to_k4s[e]:
                    cnt = k4_cnt[k4i]
                    cnt[new_c] -= 1
                    if cnt[new_c] == 0:
                        del cnt[new_c]
                    cnt[old_c] = cnt.get(old_c, 0) + 1

        T *= 0.95
        if bad == 0:
            break

    if best_colors is not None:
        used = set(best_colors)
        mapping = {c: i for i, c in enumerate(sorted(used))}
        return [mapping[c] for c in best_colors]
    else:
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
