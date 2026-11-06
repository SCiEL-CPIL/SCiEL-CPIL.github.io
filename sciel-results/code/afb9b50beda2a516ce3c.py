import time

def construct():
    start_time = time.time()
    hard_deadline = start_time + 115.0

    def verify(n, colors):
        """Verify no color class contains a C6."""
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
                    if not (adj[c][i] >> j) & 1:
                        continue
                    # Check path i-a-b-d-e-j, all distinct
                    found = False
                    for a in range(n):
                        if a == i or a == j:
                            continue
                        if not (adj[c][i] >> a) & 1:
                            continue
                        for b in range(n):
                            if b in (i, j, a):
                                continue
                            if not (adj[c][a] >> b) & 1:
                                continue
                            for d in range(n):
                                if d in (i, j, a, b):
                                    continue
                                if not (adj[c][b] >> d) & 1:
                                    continue
                                for e in range(n):
                                    if e in (i, j, a, b, d):
                                        continue
                                    if not (adj[c][d] >> e) & 1:
                                        continue
                                    if (adj[c][e] >> j) & 1:
                                        found = True
                                        break
                                if found:
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        return False
        return True

    def attempt(n, time_budget):
        """Backtracking search for 4-coloring of K_n without monochromatic C6."""
        local_deadline = time.time() + time_budget
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        colors = [-1] * num_edges
        adj = [[0] * n for _ in range(4)]

        def has_path_len_5(c, u, v):
            """Check for simple path of length 5 from u to v in color c (excluding direct edge)."""
            # u - x1 - x2 - x3 - x4 - v, all distinct
            # Iterative DFS
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

        # Iterative backtracking with symmetry breaking
        stack = [(0, 0)]  # (edge_index, colors_tried_mask)
        while stack:
            idx, tried_mask = stack[-1]

            if time.time() > local_deadline:
                return None

            if idx == num_edges:
                return colors[:]

            u, v = edges[idx]
            found = False

            # Symmetry breaking: fix first few edges from vertex 0
            allowed = [0, 1, 2, 3]
            if u == 0:
                if v == 1 and n > 1:
                    allowed = [0]
                elif v == 2 and n > 2:
                    allowed = [1]
                elif v == 3 and n > 3:
                    allowed = [2]
                elif v == 4 and n > 4:
                    allowed = [3]

            for c in allowed:
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

    # Try n=17 first
    if time.time() < hard_deadline - 2:
        res = attempt(17, 55)
        if res:
            if verify(17, res):
                return {'n': 17, 'colors': res}

    # Try n=16
    if time.time() < hard_deadline - 30:
        res = attempt(16, 25)
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

    # Try n=13
    if time.time() < hard_deadline - 4:
        res = attempt(13, 3)
        if res:
            if verify(13, res):
                return {'n': 13, 'colors': res}

    # Fallback: for n <= 5, any coloring is valid (no C6 possible)
    # Use a simple pattern for n=9
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    if verify(n, colors):
        return {'n': n, 'colors': colors}

    # Absolute fallback n=5
    n = 5
    colors = [(i * 3 + j * 2) % 4 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}
