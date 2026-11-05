import time

def construct():
    deadline = time.time() + 85.0
    N = 128
    last_aps = [[] for _ in range(N)]
    for i in range(N):
        for d in range(1, (i // 3) + 1):
            last_aps[i].append((i - 3 * d, i - 2 * d, i - d))

    import sys
    sys.setrecursionlimit(10000)

    def try_k(k):
        colors = [-1] * N

        def backtrack(i):
            if time.time() > deadline:
                return False
            if i == N:
                return True
            for c in range(k):
                ok = True
                for (a, b, cp) in last_aps[i]:
                    if len({colors[a], colors[b], colors[cp], c}) < 3:
                        ok = False
                        break
                if not ok:
                    continue
                colors[i] = c
                feasible = True
                for j in range(i + 1, N):
                    has_valid = False
                    for cc in range(k):
                        good = True
                        for (a, b, cp) in last_aps[j]:
                            if colors[a] != -1 or colors[b] != -1 or colors[cp] != -1:
                                if len({colors[a], colors[b], colors[cp], cc}) < 3:
                                    good = False
                                    break
                        if good:
                            has_valid = True
                            break
                    if not has_valid:
                        feasible = False
                        break
                if feasible and backtrack(i + 1):
                    return True
                colors[i] = -1
            return False

        if backtrack(0):
            return colors
        return None

    for k in [3, 4, 5, 6, 8, 10, 12, 16, 20, 32, 64, 128]:
        if time.time() > deadline:
            break
        res = try_k(k)
        if res is not None:
            return res

    # fallback greedy (valid)
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3
                    for d in range(1, (x - 1) // 3 + 1))
                    and all(any(colors[x - 1 - j * d] != color for j in (1, 2))
                            for d in range(1, (x - 1) // 2 + 1))):
                colors.append(color)
                break
    return colors
