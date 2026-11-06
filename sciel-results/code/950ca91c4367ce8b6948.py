import time
import random
from collections import defaultdict

_DEADLINE = 112.0
_DEADLINE = time.time() + _DEADLINE
_rng = random.Random(20240501)


def _gf16_seed(n=32):
    """Greenwood-Gleason: two GF(16) copies (Clebsch 3-colouring), colour 3 between copies."""
    def mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for _ in range(15):
        log[x] = len(log)
        x = mul(x, 2)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return colors


def _greedy_extend(colors, cur_n, target, rng):
    """Try to extend a valid n-colouring to target by adding vertices one at a time."""
    adj = [set() for _ in range(cur_n)]
    for idx in range(len(colors)):
        a = (idx - (idx + 1) // 2)
        b = (idx + 1) // 2 + 1
        if a < 0:
            a = idx - (idx + 1) // 2
            b = (idx + 1) // 2 + 1
        adj[a].add(b)
        adj[b].add(a)
    n = cur_n
    while n < target:
        if time.time() > _DEADLINE:
            return colors, n
        new = n
        n += 1
        adj.append(set())
        ok = True
        for _ in range(30):
            ok = True
            for i in range(new):
                forb = [False, False, False, False]
                for j in adj[i]:
                    if j == new:
                        continue
                    c = colors[(i * (i + 1) // 2 + j - i - 1)]
                    if forb[c]:
                        ok = False
                        break
                    forb[c] = True
                if not ok:
                    break
                pick = [c for c in range(4) if not forb[c]]
                if not pick:
                    ok = False
                    break
                colors[(new * (new + 1) // 2 + i - new)] = rng.choice(pick)
            if ok:
                for i in range(new):
                    adj[i].add(new)
                adj[new].update(range(new))
                break
        if not ok:
            # rollback
            del adj[new]
            n -= 1
            for i in range(new):
                adj[i].discard(new)
            n -= 1
            break
    return colors, n


def _local_search(colors, n, rng, max_iter=200000):
    """Min-conflicts: recolor random edges to eliminate monochromatic triangles."""
    if time.time() > _DEADLINE:
        return colors, False
    # precompute for each edge (i,j) the list of k that form a triangle with it
    tri = [None] * (n * (n - 1) // 2)
    for i in range(n):
        for j in range(i + 1, n):
            e = i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
            lst = []
            for k in range(n):
                if k == i or k == j:
                    continue
                lst.append(k)
            tri[e] = lst

    def conflict_count():
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)]
                for k in range(j + 1, n):
                    if colors[i * (n - 1) - i * (i - 1) // 2 + (k - i - 1)] == c and \
                       colors[j * (n - 1) - j * (j - 1) // 2 + (k - j - 1)] == c:
                        total += 1
        return total

    it = 0
    while it < max_iter:
        if time.time() > _DEADLINE:
            return colors, False
        it += 1
        # pick a random edge and recolor to minimise local mono-triangles
        e = rng.randrange(len(colors))
        i = (n - 1 - int(((n - 1) ** 2 - 4 * e) ** 0.5)) // 2
        j = e + i * (i - 1) // 2 + i + 1
        cur = colors[e]
        best_c, best_cost = cur, 10 ** 9
        for c in range(4):
            if c == cur:
                continue
            cost = 0
            for k in tri[e]:
                if k > i and k != j:
                    ek = i * (n - 1) - i * (i - 1) // 2 + (k - i - 1)
                    ek2 = j * (n - 1) - j * (j - 1) // 2 + (k - j - 1)
                    if colors[ek] == c and colors[ek2] == c:
                        cost += 1
                elif k < i:
                    ek = k * (n - 1) - k * (k - 1) // 2 + (i - k - 1)
                    ek2 = k * (n - 1) - k * (k - 1) // 2 + (j - k - 1)
                    if colors[ek] == c and colors[ek2] == c:
                        cost += 1
            if cost < best_cost:
                best_cost, best_c = cost, c
        colors[e] = best_c
    return colors, False


def construct():
    t0 = time.time()
    rng = random.Random(12345)

    # Start from known valid 32-vertex seed
    colors = _gf16_seed(32)
    n = 32

    # Try to extend greedily up to 50, multiple restarts
    best_colors, best_n = colors[:], n
    for attempt in range(40):
        if time.time() > _DEADLINE:
            break
        rng = random.Random(777 + attempt * 131)
        colors = _gf16_seed(32)
        n = 32
        colors, n = _greedy_extend(colors, n, 50, rng)
        if n > best_n:
            best_n = n
            best_colors = colors[:]
        if best_n >= 50:
            break

    # If we reached a larger n, optionally polish with local search on the best
    if best_n < 50 and time.time() < _DEADLINE - 10:
        # Try local search on a fresh 40-extension attempt to repair
        rng = random.Random(999)
        colors = _gf16_seed(32)
        n = 32
        colors, n = _greedy_extend(colors, n, 44, rng)
        if n >= 40:
            colors, _ = _local_search(colors, n, rng, max_iter=150000)
            if n > best_n:
                best_n = n
                best_colors = colors

    if time.time() > _DEADLINE - 3:
        pass
    return {'n': best_n, 'colors': best_colors[:best_n * (best_n - 1) // 2]}
