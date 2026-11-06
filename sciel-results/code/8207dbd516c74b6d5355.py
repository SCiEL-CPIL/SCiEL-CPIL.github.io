import time
import random

def construct():
    start_time = time.time()
    hard_deadline = start_time + 115.0

    def verify(n, colors):
        """Verify that no color class contains a C6."""
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        for c in range(4):
            for i in range(n):
                for j in range(i + 1, n):
                    if not ((adj[c][i] >> j) & 1):
                        continue
                    # Check for path of length 5 from i to j in color c
                    # i - a - b - d - e - j
                    # Use iterative DFS
                    stack = [(i, 5, 1 << i)]
                    found = False
                    while stack:
                        cur, depth, visited = stack.pop()
                        if depth == 0:
                            if cur == j:
                                found = True
                                break
                            continue
                        if depth == 1:
                            if (adj[c][cur] >> j) & 1:
                                found = True
                                break
                            continue
                        mask = adj[c][cur] & ~visited
                        while mask:
                            lsb = mask & -mask
                            nxt = lsb.bit_length() - 1
                            stack.append((nxt, depth - 1, visited | lsb))
                            mask -= lsb
                    if found:
                        return False
        return True

    def small_fallback(n):
        """Generate a valid coloring for small n."""
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append((i * 3 + j * 2) % 4)
        return colors

    def attempt(n, time_budget):
        """
        Backtracking search with forward checking for 4-coloring of K_n
        without monochromatic C6.
        """
        local_deadline = time.time() + time_budget
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        colors = [-1] * num_edges
        adj = [[0] * n for _ in range(4)]

        def has_path_len_5(c, u, v):
            """Check for simple path of length 5 from u to v in color c."""
            stack = [(u, 5, 1 << u)]
            while stack:
                cur, depth, visited = stack.pop()
                if depth == 0:
                    if cur == v:
                        return True
                    continue
                if depth == 1:
                    if (adj[c][cur] >> v) & 1:
                        return True
                    continue
                mask = adj[c][cur] & ~visited
                while mask:
                    lsb = mask & -mask
                    nxt = lsb.bit_length() - 1
                    stack.append((nxt, depth - 1, visited | lsb))
                    mask -= lsb
            return False

        def is_valid(c, u, v):
            return not has_path_len_5(c, u, v)

        # Iterative backtracking with stack
        stack = [(0, 0)]  # (edge_index, colors_tried_mask)
        while stack:
            idx, tried_mask = stack[-1]

            if time.time() > local_deadline:
                return None

            if idx == num_edges:
                return colors[:]

            u, v = edges[idx]
            found = False
            # Try colors in random order for diversity
            color_order = [0, 1, 2, 3]
            random.shuffle(color_order)
            for c in color_order:
                cbit = 1 << c
                if tried_mask & cbit:
                    continue
                if is_valid(c, u, v):
                    colors[idx] = c
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    stack[-1] = (idx, tried_mask | cbit)
                    stack.append((idx + 1, 0))
                    found = True
                    break

            if not found:
                stack.pop()
                if idx > 0:
                    prev_idx = idx - 1
                    prev_u, prev_v = edges[prev_idx]
                    prev_c = colors[prev_idx]
                    colors[prev_idx] = -1
                    adj[prev_c][prev_u] &= ~(1 << prev_v)
                    adj[prev_c][prev_v] &= ~(1 << prev_u)

        return None

    # Try n=17 first with most time
    res = attempt(17, 55)
    if res:
        if verify(17, res):
            return {'n': 17, 'colors': res}

    # Try n=16
    if time.time() < hard_deadline - 35:
        res = attempt(16, 30)
        if res:
            if verify(16, res):
                return {'n': 16, 'colors': res}

    # Try n=15
    if time.time() < hard_deadline - 15:
        res = attempt(15, 12)
        if res:
            if verify(15, res):
                return {'n': 15, 'colors': res}

    # Try n=14
    if time.time() < hard_deadline - 8:
        res = attempt(14, 6)
        if res:
            if verify(14, res):
                return {'n': 14, 'colors': res}

    # Fallback to small n
    for n in [12, 10, 9, 8]:
        if time.time() < hard_deadline - 2:
            res = attempt(n, 3)
            if res:
                if verify(n, res):
                    return {'n': n, 'colors': res}

    # Absolute fallback
    n = 8
    colors = small_fallback(n)
    if verify(n, colors):
        return {'n': n, 'colors': colors}

    n = 7
    colors = small_fallback(n)
    return {'n': n, 'colors': colors}
