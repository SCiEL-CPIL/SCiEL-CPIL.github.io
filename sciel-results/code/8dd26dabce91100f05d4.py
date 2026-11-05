import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 105

    def elapsed():
        return time.time() - start_time

    def total_c6_count(colors, n):
        """Count all monochromatic C6s."""
        adj = [[False] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i][j] = True
                adj[c][j][i] = True
                idx += 1
        total = 0
        for c in range(4):
            A = adj[c]
            for v0 in range(n):
                for v1 in range(v0 + 1, n):
                    if not A[v0][v1]:
                        continue
                    for v2 in range(v1 + 1, n):
                        if not A[v1][v2]:
                            continue
                        for v3 in range(v2 + 1, n):
                            if not A[v2][v3]:
                                continue
                            for v4 in range(v3 + 1, n):
                                if not A[v3][v4]:
                                    continue
                                for v5 in range(v4 + 1, n):
                                    if not A[v4][v5]:
                                        continue
                                    if A[v5][v0]:
                                        total += 1
        return total

    best_n = 9
    n = 9
    best_colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]

    for target_n in range(10, 15):
        if elapsed() > DEADLINE:
            break
        found = False
        for attempt in range(20):
            if elapsed() > DEADLINE:
                break
            colors = [random.randint(0, 3) for _ in range(target_n * (target_n - 1) // 2)]
            cost = total_c6_count(colors, target_n)
            if cost == 0:
                best_n = target_n
                best_colors = colors
                found = True
                break
            for it in range(3000):
                if elapsed() > DEADLINE or cost == 0:
                    break
                ei = random.randint(0, len(colors) - 1)
                old_c = colors[ei]
                best_c = old_c
                best_cost = cost
                for nc in range(4):
                    if nc == old_c:
                        continue
                    colors[ei] = nc
                    nc_cost = total_c6_count(colors, target_n)
                    if nc_cost < best_cost:
                        best_cost = nc_cost
                        best_c = nc
                colors[ei] = best_c
                cost = best_cost
            if cost == 0:
                best_n = target_n
                best_colors = colors
                found = True
                break
        if found:
            break

    return {'n': best_n, 'colors': best_colors}
