import time
import random

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 118.0

    def check_color(c, v, u, edges, n):
        """Return True if adding edge (v,u) with color c creates a C6 in color c."""
        av = edges[v]
        au = edges[u]
        for w in range(n):
            if w == v or w == u:
                continue
            if (av[c] >> w) & 1 and (au[c] >> w) & 1:
                # shared neighbor w in color c -> C6 (v,w,?,u,...) via common 2-path
                return True
        return False

    def backtrack(idx, edges, n):
        nonlocal best_colors, best_n
        if idx == n:
            if n > best_n:
                best_n = n
                best_colors = list(colors)
            return True
        if time.time() > deadline:
            return False
        v = idx
        # try colors in a heuristic order
        for c in color_order(v, edges, n):
            ok = True
            for u in range(v):
                if check_color(c, v, u, edges, n):
                    ok = False
                    break
            if not ok:
                continue
            # place
            for u in range(v):
                edges[v][c] |= (1 << u)
                edges[u][c] |= (1 << v)
            colors.append(c)
            if backtrack(idx + 1, edges, n):
                return True
            # undo
            for u in range(v):
                edges[v][c] &= ~(1 << u)
                edges[u][c] &= ~(1 << v)
            colors.pop()
        return False

    def color_order(v, edges, n):
        # prefer colors that are already present (less new structure) and fewer conflicts
        order = []
        for c in range(4):
            # count existing color-c edges incident to v
            cnt = bin(edges[v][c]).count('1')
            order.append((cnt, c))
        order.sort(key=lambda x: (-x[0], x[1]))
        return [c for _, c in order]

    def try_n(n, rng):
        nonlocal best_colors, best_n
        # init edges
        edges = [[0, 0, 0, 0] for _ in range(n)]
        colors = []
        # shuffle initial color order bias for diversity
        base_order = list(range(4))
        rng.shuffle(base_order)
        global color_order
        _base = base_order
        def color_order(v, edges, n):
            order = []
            for c in _base:
                cnt = bin(edges[v][c]).count('1')
                order.append((cnt, c))
            order.sort(key=lambda x: (-x[0], x[1]))
            return [c for _, c in order]
        backtrack(0, edges, n)
        return best_n, list(best_colors)

    best_n = 0
    best_colors = []
    rng = random.Random(12345)

    # Try increasing n from 14 down to 10, a few restarts each
    for n in range(14, 9, -1):
        if time.time() > deadline:
            break
        for _ in range(3):
            if time.time() > deadline:
                break
            cur_n, cur_colors = try_n(n, rng)
            if cur_n > best_n:
                best_n = cur_n
                best_colors = cur_colors
            if best_n >= n:
                break
        if best_n >= n:
            break

    if best_n <= 9:
        # fallback to known valid n=9
        n = 9
        cols = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        return {'n': n, 'colors': cols}

    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
