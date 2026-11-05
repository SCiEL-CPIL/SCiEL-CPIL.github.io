"""Triangle-free graph with no independent 10-set, maximizing n (R(3,10))."""
import random
import time

# Hard wall-clock limit with safety margin (evaluator kills at 150s).
_DEADLINE = time.time() + 115.0


def _indep_size(g, n):
    """Exact maximum independent set size via recursive branch & bound.
    Returns n if an independent 10-set exists (early stop), else the true max."""
    # Greedy upper bound via maximal matching on the complement.
    match = 0
    seen = [0] * n
    for i in range(n):
        if seen[i]:
            continue
        seen[i] = 1
        found = False
        for j in range(i + 1, n):
            if seen[j] and not g[i][j]:
                seen[j] = 1
                match += 1
                found = True
                break
        if not found:
            match += 1
    if 2 * match < 10:
        return 2 * match  # proven < 10

    best = [0]
    stack = [0]
    while stack:
        cand = stack.pop()
        sz = cand.bit_length()
        if sz > best[0]:
            best[0] = sz
            if best[0] >= 10:
                return 10
        if not cand:
            continue
        # Upper bound: |cand| + matching(cand)
        c = cand
        m = 0
        while c:
            lsb = c & (-c)
            i = lsb.bit_length() - 1
            c &= c - 1
            nb = c & ~g[i]
            if nb:
                m += 1
                c &= ~nb
            else:
                m += 1
        if sz + m <= best[0]:
            continue
        lsb = cand & (-cand)
        i = lsb.bit_length() - 1
        stack.append(cand & ~g[i])
        stack.append(cand & ~((1 << i) | g[i]))
    return best[0]


def _build(n, seed, greedy=False, p=0.4):
    """Construct a triangle-free graph on n vertices.
    greedy=False -> random; greedy=True -> greedy triangle-free with prob p."""
    g = [[0] * n for _ in range(n)]
    rng = random.Random(seed)
    if greedy:
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    ok = True
                    for k in range(i):
                        if g[i][k] and g[j][k]:
                            ok = False
                            break
                    if ok:
                        g[i][j] = g[j][i] = 1
    else:
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    g[i][j] = g[j][i] = 1
    return g


def _has_triangle(g, n):
    for i in range(n):
        row = g[i]
        for j in range(i + 1, n):
            if row[j]:
                common = row & g[j]
                if common:
                    return True
    return False


def _local_search(g, n, rng, max_iter, deadline):
    """Flip edges to eliminate triangles, then maximize edges without creating
    triangles and without producing an independent 10-set."""
    def tri_count():
        c = 0
        for i in range(n):
            row = g[i]
            for j in range(i + 1, n):
                if row[j]:
                    c += (row & g[j]).bit_count()
        return c

    # Phase 1: remove triangles by flipping random edges.
    it = 0
    while it < max_iter and time.time() < deadline:
        it += 1
        if not _has_triangle(g, n):
            break
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j or not g[i][j]:
            continue
        g[i][j] = g[j][i] = 0

    # Phase 2: try adding edges that keep it triangle-free and independent-<10.
    for it in range(max_iter):
        if time.time() >= deadline:
            break
        progress = False
        for i in range(n):
            row = g[i]
            for j in range(i + 1, n):
                if row[j]:
                    continue
                common = row & g[j]
                if common:
                    continue  # would create a triangle
                g[i][j] = g[j][i] = 1
                if _indep_size(g, n) < 10:
                    progress = True
                    break
                g[i][j] = g[j][i] = 0
            if progress:
                break
        if not progress:
            break
    return g


def construct():
    start = time.time()
    deadline = start + 112.0
    best_n = 0
    best_edges = []

    for n in range(40, 25, -1):
        if time.time() >= deadline - 5:
            break
        found = False
        seed = 0
        while time.time() < deadline - 8 and not found:
            g = _build(n, seed)
            seed += 1
            if _has_triangle(g, n):
                g = _local_search(g, n, random.Random(seed * 7 + 1), 40000, deadline - 8)
            if _has_triangle(g, n):
                continue
            if _indep_size(g, n) < 10:
                best_n = n
                best_edges = [[i, j] for i in range(n) for j in range(i + 1, n) if g[i][j]]
                found = True
                break
        if found:
            break

    if best_n == 0:
        # Fallback: guaranteed valid Andrasfai graph And(9) on 26 vertices.
        best_n = 26
        best_edges = [[i, j] for i in range(26) for j in range(i + 1, 26)
                      if (j - i) % 26 % 3 == 1]
    return {'n': best_n, 'edges': best_edges}
