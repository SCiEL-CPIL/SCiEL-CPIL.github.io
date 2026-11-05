"""Fresh design: greedy vertex-extension + simulated annealing with O(1) bitmask
triangle counting to push the triangle-free 4-colouring of K_n past 32."""
import time
import random
import math

DEADLINE = time.time() + 140.0  # safety margin under the 150 s kill


# ---------------------------------------------------------------------------
# Fast monochromatic-triangle counting using per-vertex, per-colour bitmasks.
# For each vertex v and colour c, mask[v][c] has bit u set iff edge (v,u) has
# colour c. A monochromatic triangle of colour c exists for pair (i,j) (i<j,
# colour c) iff mask[i][c] & mask[j][c] has any bit set (the common neighbour).
# ---------------------------------------------------------------------------
def build_masks(n, colors, idx):
    masks = [[0, 0, 0, 0] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx[i][j]]
            masks[i][c] |= 1 << j
            masks[j][c] |= 1 << i
    return masks


def count_triangles(masks, n):
    total = 0
    for i in range(n):
        mi = masks[i]
        for j in range(i + 1, n):
            c = (mi[0] >> j) & 1
            if c:
                total += (mi[0] & masks[j][0]).bit_count()
            else:
                c = (mi[1] >> j) & 1
                if c:
                    total += (mi[1] & masks[j][1]).bit_count()
                else:
                    c = (mi[2] >> j) & 1
                    if c:
                        total += (mi[2] & masks[j][2]).bit_count()
                    else:
                        total += (mi[3] & masks[j][3]).bit_count()
    return total


def make_idx(n):
    return [[0] * n for _ in range(n)]


# ---------------------------------------------------------------------------
# Seed: Greenwood-Gleason (GF(16) Clebsch) doubled with a bipartite colour-3.
# ---------------------------------------------------------------------------
def _mul(a, b):
    r = 0
    for i in range(4):
        if (b >> i) & 1:
            r ^= a << i
    for i in range(7, 3, -1):
        if (r >> i) & 1:
            r ^= 0b10011 << (i - 4)
    return r


def seed_coloring():
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = _mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return colors


def extend(n, colors, idx, masks):
    """Try to add vertex n (0..n-1 already present). Returns (ok, colors, idx, masks)."""
    if time.time() > DEADLINE:
        return False, colors, idx, masks
    best = None
    best_cost = None
    for c in range(4):
        cost = 0
        for j in range(n):
            if (masks[j][c] >> n) & 1:
                cost += (masks[j][c] & (1 << n)).bit_count()  # placeholder, recomputed below
        # Recompute properly: triangles formed with new vertex n and pair (j, k<n)
        # where edge (n,j) colour c and edge (j,k) colour c, i.e. k in masks[j][c] with k<n.
        # That's exactly popcount(masks[j][c] & ((1<<n)-1)) for each j with edge(n,j)=c.
        cost = 0
        for j in range(n):
            if (masks[j][c] >> n) & 1:
                cost += (masks[j][c] & ((1 << n) - 1)).bit_count()
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best = c
    c = best
    # append edges (j, n)
    for j in range(n):
        idx[j][n] = len(colors)
        idx[n][j] = len(colors)
        colors.append(c)
    # update masks
    row = [0, 0, 0, 0]
    for j in range(n):
        if (masks[j][c] >> n) & 1:
            row[c] |= 1 << j
            masks[j][c] |= 1 << n
    masks.append(row)
    return True, colors, idx, masks


def anneal(n, colors, idx, masks, budget):
    """Reduce monochromatic triangles by flipping random edges. Returns (colors, idx, masks)."""
    cur = count_triangles(masks, n)
    best = cur
    best_colors = colors[:]
    best_idx = [r[:] for r in idx]
    best_masks = [[m[0], m[1], m[2], m[3]] for m in masks]

    def flip(i, j, newc):
        oldc = colors[idx[i][j]]
        if oldc == newc:
            return
        colors[idx[i][j]] = newc
        # remove old edge from masks
        masks[i][oldc] &= ~(1 << j)
        masks[j][oldc] &= ~(1 << i)
        # add new edge
        masks[i][newc] |= 1 << j
        masks[j][newc] |= 1 << i

    t = 0.0
    tmin = 0.001
    steps = 0
    while time.time() < budget:
        i = random.randrange(n)
        j = random.randrange(n)
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        newc = random.randrange(4)
        oldc = colors[idx[a][b]]
        if newc == oldc:
            continue
        # cost delta of flipping edge (a,b) from oldc to newc
        # triangles destroyed: pairs (a,b,x) where x has edge(a,x)=oldc and edge(b,x)=oldc
        destroyed = (masks[a][oldc] & masks[b][oldc]).bit_count()
        # triangles created: x has edge(a,x)=newc and edge(b,x)=newc
        created = (masks[a][newc] & masks[b][newc]).bit_count()
        d = created - destroyed
        if d <= 0 or random.random() < math.exp(-d / max(t, 1e-6)):
            flip(a, b, newc)
            cur += d
            if cur < best:
                best = cur
                best_colors = colors[:]
                best_idx = [r[:] for r in idx]
                best_masks = [[m[0], m[1], m[2], m[3]] for m in masks]
        t *= 0.9999
        if t < tmin:
            t = tmin
        steps += 1
        if steps % 4096 == 0:
            # occasional reheat to escape
            if best == 0:
                break
    # restore best
    colors[:] = best_colors
    for r in range(n):
        idx[r] = best_idx[r][:]
    for r in range(n):
        for c in range(4):
            masks[r][c] = best_masks[r][c]
    return colors, idx, masks


def construct():
    colors = seed_coloring()
    n = 32
    idx = make_idx(n)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx[i][j] = k
            idx[j][i] = k
            k += 1
    masks = build_masks(n, colors, idx)
    base = count_triangles(masks, n)  # should be 0

    # Greedy extend with anneal-in-between to break plateaus.
    for _ in range(40):
        if time.time() > DEADLINE - 30:
            break
        ok, colors, idx, masks = extend(n, colors, idx, masks)
        if not ok:
            break
        n += 1
        cur = count_triangles(masks, n)
        if cur > 0:
            # anneal to try to clear triangles before continuing
            colors, idx, masks = anneal(n, colors, idx, masks, time.time() + 25)
            if count_triangles(masks, n) == 0:
                continue
            else:
                # could not clear; stop extending
                break
        # keep going (cur == 0)

    # Final polish: try to extend more while valid, and anneal to keep improving n.
    while time.time() < DEADLINE - 20:
        ok, colors, idx, masks = extend(n, colors, idx, masks)
        if not ok:
            break
        n += 1
        cur = count_triangles(masks, n)
        if cur > 0:
            colors, idx, masks = anneal(n, colors, idx, masks, time.time() + 20)
            if count_triangles(masks, n) == 0:
                continue
            else:
                break

    # If we have a valid coloring, return it. Otherwise fall back to the seed.
    final = count_triangles(masks, n)
    if final != 0:
        # revert to the known-valid 32-vertex seed
        colors = seed_coloring()
        n = 32
        return {'n': n, 'colors': colors}
    return {'n': n, 'colors': colors}
