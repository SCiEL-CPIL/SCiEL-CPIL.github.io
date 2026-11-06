import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    def edge_index(a, b, n):
        """O(1) edge index lookup."""
        if a > b:
            a, b = b, a
        return a * n - a * (a + 1) // 2 + (b - a - 1)

    def build_adj_bitmasks(n, colors):
        adj0 = [0] * n
        adj1 = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                else:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
                idx += 1
        return adj0, adj1

    def _find_k222(adj):
        """Find a K2,2,2 subgraph in adj (bitmask adjacency). Returns (u,v,w,x,y,z) or None."""
        n = len(adj)
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    continue
                common = adj[u] & adj[v]
                verts = []
                mask = common
                while mask:
                    lsb = mask & -mask
                    idx = lsb.bit_length() - 1
                    verts.append(idx)
                    mask ^= lsb
                for i in range(len(verts)):
                    w = verts[i]
                    for j in range(i + 1, len(verts)):
                        x = verts[j]
                        if adj[w] & (1 << x):
                            continue
                        cand = common & adj[w] & adj[x]
                        cverts = []
                        cmask = cand
                        while cmask:
                            lsb = cmask & -cmask
                            idx = lsb.bit_length() - 1
                            cverts.append(idx)
                            cmask ^= lsb
                        for k in range(len(cverts)):
                            y = cverts[k]
                            for l in range(k + 1, len(cverts)):
                                z = cverts[l]
                                if not (adj[y] & (1 << z)):
                                    return (u, v, w, x, y, z)
        return None

    def _has_k222(adj):
        return _find_k222(adj) is not None

    def check_coloring(n, colors):
        adj0, adj1 = build_adj_bitmasks(n, colors)
        return not (_has_k222(adj0) or _has_k222(adj1))

    def paley_with_prime(n, p):
        """Paley coloring using prime p >= n."""
        if p < n:
            return None
        if p < 2:
            return None
        if p % 2 == 0:
            return None
        for i in range(3, int(p ** 0.5) + 1, 2):
            if p % i == 0:
                return None
        squares = {x * x % p for x in range(1, p)}
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % p in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    def local_search_violation(n, colors, max_time=10.0):
        """Local search: find K2,2,2 violations and flip edges to break them."""
        start = time.time()
        colors = list(colors)
        for iteration in range(20000):
            if time.time() - start > max_time:
                break
            adj0, adj1 = build_adj_bitmasks(n, colors)
            k222 = _find_k222(adj0)
            if k222:
                u, v, w, x, y, z = k222
                cross_edges = [(u, w), (u, x), (u, y), (u, z), (v, w), (v, x), (v, y), (v, z),
                               (w, y), (w, z), (x, y), (x, z)]
                edge = random.choice(cross_edges)
                a, b = edge
                if a > b:
                    a, b = b, a
                idx = edge_index(a, b, n)
                colors[idx] = 1 - colors[idx]
                continue
            k222 = _find_k222(adj1)
            if k222:
                u, v, w, x, y, z = k222
                cross_edges = [(u, w), (u, x), (u, y), (u, z), (v, w), (v, x), (v, y), (v, z),
                               (w, y), (w, z), (x, y), (x, z)]
                edge = random.choice(cross_edges)
                a, b = edge
                if a > b:
                    a, b = b, a
                idx = edge_index(a, b, n)
                colors[idx] = 1 - colors[idx]
                continue
            return colors
        return colors

    def count_k222_numba_like(n, colors):
        """Count K2,2,2 subgraphs in both color classes using vectorized approach."""
        # Build adjacency matrices as numpy arrays
        m = n * (n - 1) // 2
        adj1 = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[idx] == 1:
                    adj1[i, j] = 1
                    adj1[j, i] = 1
                idx += 1
        adj0 = 1 - adj1
        np.fill_diagonal(adj0, 0)
        np.fill_diagonal(adj1, 0)

        count = 0
        # For each pair (u,v) not connected in color 1
        for u in range(n):
            for v in range(u + 1, n):
                if adj1[u, v]:
                    continue
                # Common neighbors of u and v in color 1
                common = adj1[u] & adj1[v]
                common_idx = np.where(common)[0]
                if len(common_idx) < 4:
                    continue
                # For each pair (w,x) in common not connected in color 1
                for wi in range(len(common_idx)):
                    w = common_idx[wi]
                    for xi in range(wi + 1, len(common_idx)):
                        x = common_idx[xi]
                        if adj1[w, x]:
                            continue
                        # Candidates y,z must be common neighbors of w,x in color 1
                        cand = common & adj1[w] & adj1[x]
                        cand_idx = np.where(cand)[0]
                        if len(cand_idx) < 2:
                            continue
                        # Count pairs (y,z) not connected in color 1
                        for yi in range(len(cand_idx)):
                            y = cand_idx[yi]
                            for zi in range(yi + 1, len(cand_idx)):
                                z = cand_idx[zi]
                                if not adj1[y, z]:
                                    count += 1
        return count

    # Try Paley with various primes, descending n
    primes = [31, 29, 23, 19, 17]

    for n in range(31, 16, -1):
        if time.time() > deadline - 10:
            break
        for p in primes:
            if time.time() > deadline - 10:
                break
            if p < n:
                continue
            colors = paley_with_prime(n, p)
            if colors is None:
                continue
            if check_coloring(n, colors):
                return {'n': n, 'colors': colors}

    # Local search from Paley-like and random starts
    for n in range(31, 16, -1):
        if time.time() > deadline - 15:
            break
        attempts = 0
        while time.time() < deadline - 15 and attempts < 8:
            attempts += 1
            # Try different initializations
            if attempts % 2 == 0:
                # Paley with largest prime >= n
                p = next((pr for pr in primes if pr >= n), 31)
                colors = paley_with_prime(n, p)
                if colors is None:
                    colors = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
            else:
                colors = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]

            remaining = deadline - 15 - time.time()
            if remaining < 3:
                break
            colors = local_search_violation(n, colors, max_time=min(12.0, remaining))
            if check_coloring(n, colors):
                return {'n': n, 'colors': colors}

    # Fallback to known valid n=17 Paley
    squares = {x * x % 17 for x in range(1, 17)}
    colors17 = [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]
    return {'n': 17, 'colors': colors17}
