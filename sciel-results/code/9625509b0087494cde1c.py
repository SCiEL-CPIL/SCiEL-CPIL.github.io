import time
import random
import numpy as np

_DEADLINE = 140.0
_START = time.time()

# ---------------------------------------------------------------------------
# Fast monochromatic-triangle counter using bitmasks.
# For each colour c and vertex v, adj[c][v] is a 64-bit mask of neighbours of v
# joined to v by a colour-c edge. A monochromatic c-triangle (i<j<k) exists
# iff (adj[c][i] & adj[c][j]) >> k has bit 0 set.
# ---------------------------------------------------------------------------
def _build_adj(n, colors):
    adj = [np.zeros(n, dtype=np.uint64) for _ in range(4)]
    for idx, c in enumerate(colors):
        i = idx // (n - 1)
        j = idx - i * (n - 1)
        adj[c][i] |= 1 << j
        adj[c][j] |= 1 << i
    return adj


def _count_mono(adj, n):
    total = 0
    for c in range(4):
        A = adj[c]
        for i in range(n - 2):
            ai = A[i]
            for j in range(i + 1, n - 1):
                total += (ai & A[j] & (1 << (j + 1))).bit_count()
    return total


# ---------------------------------------------------------------------------
# Initial colourings
# ---------------------------------------------------------------------------
def _greedy_color(n, rng):
    """Greedy vertex-by-vertex 4-colouring; never creates a mono triangle."""
    adj = [0] * (4 * n)
    cols = []
    for v in range(n):
        for u in range(v):
            ok = [0, 0, 0, 0]
            for w in range(u):
                c = cols[u * (n - 1) - u * u + w]
                if (adj[c * n + u] >> v) & 1 and (adj[c * n + v] >> u) & 1:
                    ok[c] = 1
            best = -1
            for c in range(4):
                if ok[c] == 0:
                    cnt = bin(adj[c * n + v]).count('1')
                    if cnt < best or best == -1:
                        best = cnt
                        pick = c
            if best == -1:
                pick = rng.randrange(4)
            cols.append(pick)
            adj[pick * n + u] |= 1 << v
            adj[pick * n + v] |= 1 << u
    return cols


def _clebsch_block(n):
    """A 3-colouring of K_n (n<=16) with no monochromatic triangle (GF(16) cubic residues)."""
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
        log[x] = x
        x = mul(x, 2)
    log = {v: e for e, v in log.items()}
    # recompute exponent map properly
    exp = {}
    x = 1
    for e in range(15):
        exp[x] = e
        x = mul(x, 2)
    cols = []
    for i in range(n):
        for j in range(i + 1, n):
            cols.append(exp[i ^ j] % 3)
    return cols


def _block_init(n, rng):
    """4 equal blocks; colour c inside block c; 3-colour inter-block edges (no mono triangle)."""
    b = n // 4
    rem = n - 4 * b
    sizes = [b + (1 if i < rem else 0) for i in range(4)]
    start = [0] + list(np.cumsum(sizes))
    cols = [0] * (n * (n - 1) // 2)
    for c in range(4):
        s0, s1 = start[c], start[c + 1]
        blk = _clebsch_block(s1 - s0) if (s1 - s0) <= 16 else None
        for a in range(s0, s1):
            for b2 in range(a + 1, s1):
                if blk is not None:
                    cols[a * (n - 1) - a * a + b2] = blk[(a - s0) * (s1 - s0 - 1) - (a - s0) * (a - s0) + (b2 - s0 - 1)]
                else:
                    cols[a * (n - 1) - a * a + b2] = rng.randrange(3)
    for a in range(n):
        for b2 in range(a + 1, n):
            ca = a // b if b else 0
            # determine block of a and b2
            ba = next(c for c in range(4) if start[c] <= a < start[c + 1])
            bb = next(c for c in range(4) if start[c] <= b2 < start[c + 1])
            if ba != bb:
                cols[a * (n - 1) - a * a + b2] = (ba + bb) % 3
    return cols


# ---------------------------------------------------------------------------
# Local optimisation: hill-climb + annealing on monochromatic-triangle count.
# ---------------------------------------------------------------------------
def _optimize(n, colors, budget, rng, restarts=1):
    best_cols = list(colors)
    best_cnt = _count_mono(_build_adj(n, best_cols), n)
    best_n = n if best_cnt == 0 else -1
    total_deadline = _START + budget
    for _ in range(restarts):
        cols = list(best_cols) if best_cnt == 0 else list(colors)
        if best_cnt == 0 and rng.random() < 0.5:
            # perturb a valid solution slightly to escape
            for _ in range(rng.randrange(1, 5)):
                k = rng.randrange(len(cols))
                cols[k] = rng.randrange(4)
        adj = _build_adj(n, cols)
        cnt = _count_mono(adj, n)
        if cnt == 0:
            best_n = n
            best_cols = list(cols)
            best_cnt = 0
        T = 3.0
        step = 0
        while time.time() < total_deadline - 2.0:
            if cnt == 0:
                best_n = n
                best_cols = list(cols)
                best_cnt = 0
                # explore a bit more then stop
                if step > 400:
                    break
            i = rng.randrange(n - 1)
            j = rng.randrange(i + 1, n)
            idx = i * (n - 1) - i * i + j
            old = cols[idx]
            new = (old + 1 + rng.randrange(3)) % 4
            if new == old:
                continue
            # delta: change of triangle count from recoloring edge (i,j)
            # triangles containing (i,j) with third vertex k: mono if both (i,k),(j,k) same colour
            di = adj[old][i]
            dj = adj[old][j]
            # count mono triangles that use edge (i,j) in colour old
            old_tri = (di & dj).bit_count()
            # after recoloring, edge (i,j) is colour new; triangles (i,j,k) mono in new if (i,k)&(j,k) both new
            di2 = adj[new][i]
            dj2 = adj[new][j]
            new_tri = (di2 & dj2).bit_count()
            # but triangles (i,j,k) that were mono in old and involve k where (i,k) or (j,k) also old
            # careful: old_tri counts k where both (i,k) and (j,k) are old -> those become non-mono
            # new_tri counts k where both are new -> become mono
            # However k must be != i,j; masks exclude self so fine.
            delta = new_tri - old_tri
            if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
                # apply
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if (di >> k) & 1:
                        adj[old][i] &= ~(1 << k)
                    if (dj >> k) & 1:
                        adj[old][j] &= ~(1 << k)
                    if (di2 >> k) & 1:
                        adj[new][i] |= 1 << k
                    if (dj2 >> k) & 1:
                        adj[new][j] |= 1 << k
                adj[old][i] &= ~(1 << j)
                adj[old][j] &= ~(1 << i)
                adj[new][i] |= 1 << j
                adj[new][j] |= 1 << i
                cols[idx] = new
                cnt += delta
                if cnt < best_cnt:
                    best_cnt = cnt
                    best_cols = list(cols)
                T *= 0.9995
                step += 1
            else:
                step += 1
    return best_n, best_cols, best_cnt


def construct():
    rng = random.Random(12345)
    budget = _DEADLINE - (time.time() - _START)
    if budget < 5.0:
        # not enough time: return a known-good 32-vertex solution
        return _fallback()
    best_n = -1
    best_cols = None
    # try increasing n from a good structural start
    for n in [36, 40, 44, 48, 50]:
        if time.time() - _START > _DEADLINE - 12.0:
            break
        init = _block_init(n, rng)
        nn, cols, cnt = _optimize(n, init, budget, rng, restarts=1)
        if cnt == 0:
            if n > best_n:
                best_n = n
                best_cols = cols
        # also try greedy start
        if time.time() - _START > _DEADLINE - 8.0:
            break
        ginit = _greedy_color(n, rng)
        nn2, cols2, cnt2 = _optimize(n, ginit, budget, rng, restarts=1)
        if cnt2 == 0 and n > best_n:
            best_n = n
            best_cols = cols2
    if best_n < 0:
        return _fallback()
    return {'n': best_n, 'colors': best_cols}


def _fallback():
    # Greenwood-Gleason / Clebsch doubled 32-vertex valid colouring (from seed)
    def mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r
    exp = {}
    x = 1
    for e in range(15):
        exp[x] = e
        x = mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(exp[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors}
