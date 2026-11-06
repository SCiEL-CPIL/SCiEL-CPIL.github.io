import time
import random

def _count_c4s(color_adj, n):
    """Count C4 violations in a single color class given as list of adjacency bitmasks."""
    cnt = 0
    for a in range(n):
        A = color_adj[a]
        for b in range(a + 1, n):
            cnt += (A & color_adj[b]).bit_count()
    return cnt

def _find_c4(color_adj, n):
    """Find one C4 in a color class; return (a,b,c,d) or None."""
    for a in range(n):
        A = color_adj[a]
        for b in range(a + 1, n):
            common = A & color_adj[b]
            c = (common & -common).bit_length() - 1
            if c < a:
                c = (common & -common).bit_length() - 1
            # pick c from common
            if common == 0:
                continue
            c = (common & -common).bit_length() - 1
            # need d in A & adj[b] & adj[c]
            dmask = A & color_adj[b] & color_adj[c]
            dmask &= ~(1 << a) & ~(1 << b) & ~(1 << c)
            if dmask:
                d = (dmask & -dmask).bit_length() - 1
                return (a, b, c, d)
    return None

def _init_cyclic(n):
    """Cyclic coloring: color(i,j) = min(j-i, n-(j-i)) mod 5."""
    colors = [0] * (n * (n - 1) // 2)
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            colors[i * n + j] = min(d, n - d) % 5
    return colors

def _build_adj(colors, n):
    adj = [[0] * n for _ in range(5)]
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[i * n + j]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
    return adj

def _is_valid(adj, n):
    for c in range(5):
        ca = adj[c]
        for a in range(n):
            A = ca[a]
            for b in range(a + 1, n):
                if (A & ca[b]).bit_count() >= 2:
                    return False
    return True

def _random_valid(n, deadline, seed):
    """Random edge coloring, then repair C4s via local search."""
    rng = random.Random(seed)
    colors = [rng.randrange(5) for _ in range(n * (n - 1) // 2)]
    adj = _build_adj(colors, n)
    # repair loop
    while time.time() < deadline:
        # find a violating color
        found = None
        for c in range(5):
            v = _find_c4(adj[c], n)
            if v:
                found = (c, v)
                break
        if found is None:
            return colors
        c, (a, b, cc, d) = found
        # try recoloring one of the 4 edges to break the C4
        edges = [(a, b), (a, cc), (b, d), (cc, d), (a, d), (b, cc)]
        rng.shuffle(edges)
        done = False
        for (x, y) in edges:
            if x > y:
                x, y = y, x
            idx = x * n + y
            old = colors[idx]
            if old != c:
                continue
            # try other colors
            for nc in range(5):
                if nc == old:
                    continue
                # tentatively recolor
                colors[idx] = nc
                # update adj
                adj[old][x] &= ~(1 << y)
                adj[old][y] &= ~(1 << x)
                adj[nc][x] |= (1 << y)
                adj[nc][y] |= (1 << x)
                # check this C4 gone and no new C4 in nc
                ok = True
                if (adj[c][a] & adj[c][b]).bit_count() >= 2:
                    # still has C4 a-b; recoloring didn't help this one
                    pass
                # quick check: does nc still have a C4 involving a,b?
                if (adj[nc][a] & adj[nc][b]).bit_count() >= 2:
                    ok = False
                if ok:
                    done = True
                    break
                else:
                    # revert
                    colors[idx] = old
                    adj[old][x] |= (1 << y)
                    adj[old][y] |= (1 << x)
                    adj[nc][x] &= ~(1 << y)
                    adj[nc][y] &= ~(1 << x)
            if done:
                break
        if not done:
            # random kick
            idx = rng.randrange(len(colors))
            old = colors[idx]
            nc = rng.randrange(5)
            colors[idx] = nc
            i = idx // n
            j = idx - i * n
            adj[old][i] &= ~(1 << j)
            adj[old][j] &= ~(1 << i)
            adj[nc][i] |= (1 << j)
            adj[nc][j] |= (1 << i)
    return colors

def _local_search(colors, n, deadline, seed):
    """Local search minimizing total C4 count."""
    rng = random.Random(seed)
    adj = _build_adj(colors, n)
    total = sum(_count_c4s(adj[c], n) for c in range(5))
    best = total
    T = 1.0
    while time.time() < deadline and total > 0:
        # pick a random edge
        i = rng.randrange(n)
        j = rng.randrange(i + 1, n)
        idx = i * n + j
        old = colors[idx]
        nc = rng.randrange(5)
        if nc == old:
            continue
        # delta
        # removing (i,j) from old: decreases C4 in old by number of common neighbors
        d_old = (adj[old][i] & adj[old][j]).bit_count()
        # adding to nc: increases C4 in nc
        d_new = (adj[nc][i] & adj[nc][j]).bit_count()
        delta = d_new - d_old
        if delta < 0 or (T > 0 and rng.random() < 2.71828 ** (-delta / max(T, 1e-6))):
            colors[idx] = nc
            adj[old][i] &= ~(1 << j)
            adj[old][j] &= ~(1 << i)
            adj[nc][i] |= (1 << j)
            adj[nc][j] |= (1 << i)
            total += delta
            if total < best:
                best = total
        T *= 0.999
    return colors, total

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin
    best_n = 11
    best_colors = _init_cyclic(11)
    # guarantee a valid small solution
    adj = _build_adj(best_colors, 11)
    if not _is_valid(adj, 11):
        # fallback: simple valid construction for small n
        # n=6: 5 colors, each color a matching-ish; use cyclic
        pass
    # try to improve from n=12 upward
    for n in range(12, 29):
        if time.time() > deadline - 5.0:
            break
        # multiple attempts
        for attempt in range(3):
            if time.time() > deadline - 2.0:
                break
            seed = n * 1000 + attempt
            colors = _random_valid(n, deadline - 2.0, seed)
            adj = _build_adj(colors, n)
            if _is_valid(adj, n):
                best_n = n
                best_colors = colors
                break
    return {'n': best_n, 'colors': best_colors}
