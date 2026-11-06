"""Rewritten program: backtracking C6-free 4-coloring of K_n, maximizing n."""
import time

_START = None


def _now():
    return time.time()


def _elapsed():
    return _now() - _START if _START is not None else 0.0


def construct():
    global _START
    _START = _now()
    deadline = _START + 115.0  # safety margin under 120s

    best = None
    for n in (18, 17, 16, 15, 14, 13, 12, 11, 10, 9):
        if _now() > deadline:
            break
        result = _solve(n, deadline)
        if result is not None:
            best = result
            break

    if best is None:
        n = 9
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
    return best


def _solve(n, deadline):
    m = n * (n - 1) // 2
    color = [-1] * m
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    adj = [[0] * n for _ in range(4)]

    def has_c6_using(i, j, c):
        """Check if adding edge (i,j) in color c creates a C6 in color c.
        Looks for a path i-a-b-d-e-j of length 5 in color c, all vertices distinct,
        not using the new edge (i,j)."""
        ai = adj[c][i]
        ai_bits = ai
        while ai_bits:
            lb = ai_bits & (-ai_bits)
            a = lb.bit_length() - 1
            ai_bits ^= lb
            if a == j:
                continue
            ab = adj[c][a] & ~lb
            ab_bits = ab
            while ab_bits:
                lb2 = ab_bits & (-ab_bits)
                b = lb2.bit_length() - 1
                ab_bits ^= lb2
                if b == i or b == j:
                    continue
                bd = adj[c][b] & ~lb2
                bd_bits = bd
                while bd_bits:
                    lb3 = bd_bits & (-bd_bits)
                    d = lb3.bit_length() - 1
                    bd_bits ^= lb3
                    if d == i or d == j or d == a:
                        continue
                    de = adj[c][d] & ~lb3
                    de_bits = de
                    while de_bits:
                        lb4 = de_bits & (-de_bits)
                        e = lb4.bit_length() - 1
                        de_bits ^= lb4
                        if e == i or e == j or e == a or e == b:
                            continue
                        if (adj[c][e] >> j) & 1:
                            return True
        return False

    def dfs2(edge_idx):
        if _now() > deadline:
            return None
        if edge_idx == m:
            return color[:]
        i, j = edges[edge_idx]
        for c in range(4):
            if has_c6_using(i, j, c):
                continue
            color[edge_idx] = c
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            res = dfs2(edge_idx + 1)
            if res is not None:
                return res
            adj[c][i] &= ~(1 << j)
            adj[c][j] &= ~(1 << i)
            color[edge_idx] = -1
        return None

    if n >= 4:
        # Symmetry breaking: (0,1)->0, (0,2)->{0,1}, (0,3)->{0,1,2}
        for c1 in (0, 1):
            for c2 in (0, 1, 2):
                if _now() > deadline:
                    break
                color[0] = 0
                adj[0][0] |= (1 << 1)
                adj[0][1] |= (1 << 0)
                color[1] = c1
                adj[c1][0] |= (1 << 2)
                adj[c1][2] |= (1 << 0)
                color[2] = c2
                adj[c2][0] |= (1 << 3)
                adj[c2][3] |= (1 << 0)
                res = dfs2(3)
                if res is not None:
                    return res
                adj[c2][0] &= ~(1 << 3)
                adj[c2][3] &= ~(1 << 0)
                adj[c1][0] &= ~(1 << 2)
                adj[c1][2] &= ~(1 << 0)
                adj[0][0] &= ~(1 << 1)
                adj[0][1] &= ~(1 << 0)
                color[0] = color[1] = color[2] = -1
    else:
        res = dfs2(0)
        if res is not None:
            return res
    return None
