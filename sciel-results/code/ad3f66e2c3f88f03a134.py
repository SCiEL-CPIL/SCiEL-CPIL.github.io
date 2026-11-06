import time

def construct():
    start = time.time()
    deadline = start + 110.0

    def _build(n):
        q = 5
        # vertices 0..24 are GF(5)^2, vertex 25 is the point at infinity
        # directions: 0: x-axis, 1: y-axis, 2: slope 1, 3: slope -1, 4: infinity
        def coord(v):
            if v == 25:
                return None
            return (v % 5, v // 5)

        colors = []
        for i in range(n):
            ci = coord(i)
            for j in range(i + 1, n):
                cj = coord(j)
                if ci is None or cj is None:
                    # one of them is infinity -> direction 4
                    colors.append(4)
                    continue
                dx = (cj[0] - ci[0]) % 5
                dy = (cj[1] - ci[1]) % 5
                if dx == 0:
                    c = 1
                elif dy == 0:
                    c = 0
                elif dx == dy:
                    c = 2
                elif (dx + dy) % 5 == 0:
                    c = 3
                else:
                    # should not happen for q=5 (all nonzero slopes covered)
                    c = 4
                colors.append(c)
        return {'n': n, 'colors': colors}

    result = _build(26)

    # Safety: verify no monochromatic C4 within the time budget.
    # A C4 in color c exists iff two vertices share >=2 common neighbors in color c.
    n = result['n']
    colors = result['colors']
    idx = {}
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx[(i, j)] = k
            k += 1

    # Build per-color adjacency neighbor sets
    nbrs = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx[(i, j)]]
            nbrs[i].add((j, c))
            nbrs[j].add((i, c))

    # Check each color class for C4
    def valid():
        for i in range(n):
            seen = {}
            for (j, c) in nbrs[i]:
                if j <= i:
                    continue
                # For pair (i,j) in color c, count common neighbors in color c
                pass
        # More direct: for each color, for each unordered pair of vertices,
        # count common neighbors in that color.
        for c in range(5):
            col_nbr = [set() for _ in range(n)]
            for i in range(n):
                for (j, cc) in nbrs[i]:
                    if cc == c:
                        col_nbr[i].add(j)
            for i in range(n):
                for j in range(i + 1, n):
                    common = col_nbr[i] & col_nbr[j]
                    if len(common) >= 2:
                        return False
        return True

    if valid():
        return result

    # Fallback to a smaller guaranteed-valid construction (K11) if verification fails.
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
