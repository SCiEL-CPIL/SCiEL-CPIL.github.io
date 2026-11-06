import time
import itertools
import random

def _check_k4_fast(colors):
    """Check every K4 has at least 5 distinct colours. Returns True if valid."""
    n = 16
    edge = [[0]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            edge[i][j] = c
            edge[j][i] = c
            idx += 1

    for a in range(n):
        ea = edge[a]
        for b in range(a + 1, n):
            eb = edge[b]
            ab = ea[b]
            for c in range(b + 1, n):
                ac = ea[c]
                bc = eb[c]
                ec = edge[c]
                for d in range(c + 1, n):
                    mask = (1 << ab) | (1 << ac) | (1 << ea[d]) | (1 << bc) | (1 << eb[d]) | (1 << ec[d])
                    if mask.bit_count() < 5:
                        return False
    return True

def _num_colours(colors):
    return len(set(colors))

def _f4_slope_coloring():
    """Colour K16 edges by slope in F4^2 (affine plane AG(2,4))."""
    f4_add = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0]
    ]
    f4_mul = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]
    def f4_inv(x):
        if x == 0:
            return None
        for y in range(4):
            if f4_mul[x][y] == 1:
                return y
        return None

    n = 16
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = i // 4, i % 4
            x2, y2 = j // 4, j % 4
            dx = f4_add[x2][x1]  # x2 - x1 = x2 + x1 in char 2
            dy = f4_add[y2][y1]
            if dx == 0:
                colors.append(4)  # vertical
            else:
                dx_inv = f4_inv(dx)
                slope = f4_mul[dy][dx_inv]
                colors.append(slope)
    return colors

def _linear_coloring(a, b, p):
    n = 16
    return [(a * i + b * j) % p for i in range(n) for j in range(i + 1, n)]

def _local_search_reduce(colors, deadline):
    """Try to reduce colour count by recoloring individual edges.
    Returns a valid coloring with (hopefully) fewer colours."""
    n = 16
    cur = list(colors)
    cur_count = _num_colours(cur)

    # Precompute for speed: edge index -> (i, j)
    edge_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_pairs.append((i, j))

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to recolor each edge to an existing colour that reduces the palette
        # or to a new colour only if it doesn't increase the palette
        used = set(cur)
        # Order edges: prefer edges whose colour is used exactly once (easy to remove)
        from collections import Counter
        cnt = Counter(cur)
        # Sort: edges with rare colours first
        edge_order = sorted(range(len(cur)), key=lambda e: (cnt[cur[e]], e))
        for e in edge_order:
            if time.time() > deadline:
                break
            if cnt[cur[e]] != 1:
                # Only try to remove colours that appear once
                continue
            # Try recoloring edge e to each other used colour
            for new_c in used:
                if new_c == cur[e]:
                    continue
                old = cur[e]
                cur[e] = new_c
                if _check_k4_fast(cur):
                    # Check if this actually reduces the palette
                    new_used = set(cur)
                    if len(new_used) < cur_count:
                        cur_count = len(new_used)
                        used = new_used
                        cnt = Counter(cur)
                        improved = True
                        break
                    else:
                        cur[e] = old
                else:
                    cur[e] = old
    return cur, cur_count

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n = 16
    best_colors = None
    best_count = 999

    # Candidate generators
    candidates = []

    # 1. F4^2 slope coloring (6 colours)
    candidates.append(_f4_slope_coloring())

    # 2. Additive mod p
    for p in [13, 17, 19, 23, 29, 31]:
        candidates.append([(i + j) % p for i in range(n) for j in range(i + 1, n)])

    # 3. Linear forms (a*i + b*j) % p
    for p in [13, 17, 19, 23]:
        for a in range(1, p):
            for b in range(1, p):
                candidates.append(_linear_coloring(a, b, p))

    # 4. Quadratic forms (a*i + b*j + c*i*j) % p
    for p in [17, 19, 23]:
        for a in range(1, 4):
            for b in range(1, 4):
                for c in range(0, 3):
                    colors = [(a * i + b * j + c * i * j) % p for i in range(n) for j in range(i + 1, n)]
                    candidates.append(colors)

    # 5. Product forms (i*j) % p
    for p in [17, 19, 23]:
        candidates.append([(i * j) % p for i in range(n) for j in range(i + 1, n)])

    # Evaluate candidates, keeping the best valid one
    for colors in candidates:
        if time.time() > deadline - 30:
            break
        cnt = _num_colours(colors)
        if cnt < best_count:
            if _check_k4_fast(colors):
                best_colors = colors[:]
                best_count = cnt
                if best_count <= 13:
                    break

    # Fallback if nothing found
    if best_colors is None:
        best_colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
        best_count = _num_colours(best_colors)

    # Local search to reduce colour count
    if time.time() < deadline - 10:
        polished, polished_count = _local_search_reduce(best_colors, deadline - 10)
        if polished_count < best_count:
            best_colors = polished
            best_count = polished_count

    return best_colors
