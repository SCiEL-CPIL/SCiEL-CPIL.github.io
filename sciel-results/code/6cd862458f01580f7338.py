"""Combine: K32 seed floor + greedy-init simulated annealing (inspiration refinement)."""
import time
import random
import math

_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin under the 150 s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _seed_32():
    """Greenwood-Gleason colouring of K32 (guaranteed valid floor)."""
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return colors

def _compute_index(n, i, j):
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

def _build_adj(colors, n):
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj

def _count_mono_triangles(adj, n):
    count = 0
    for c in range(4):
        for i in range(n):
            neigh = adj[c][i] & ~((1 << (i + 1)) - 1)
            j_mask = neigh
            while j_mask:
                j_bit = j_mask & (-j_mask)
                j = j_bit.bit_length() - 1
                j_mask ^= j_bit
                common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                count += common.bit_count()
    return count

def _greedy_init(n):
    """Build vertex-by-vertex, choosing the color minimizing immediate triangles."""
    m = n * (n - 1) // 2
    colors = [0] * m
    adj = [[0] * n for _ in range(4)]
    for v in range(1, n):
        for u in range(v):
            best_c = 0
            min_cost = float('inf')
            for c in range(4):
                # number of neighbours of u already in color c (would form triangle)
                cost = (adj[c][u] & ((1 << u) - 1)).bit_count()
                if cost < min_cost:
                    min_cost = cost
                    best_c = c
            idx = _compute_index(n, u, v)
            colors[idx] = best_c
            adj[best_c][u] |= (1 << v)
            adj[best_c][v] |= (1 << u)
    return colors

def _sa_search(n, time_budget):
    """Simulated annealing from greedy init. Returns (colors, tri_count) of best found."""
    m = n * (n - 1) // 2
    start_time = time.time()

    idx_to_edge = [None] * m
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx_to_edge[idx] = (i, j)
            idx += 1

    best_tri = float('inf')
    best_result = None

    while time.time() - start_time < time_budget and _time_left() > 2.0:
        colors = _greedy_init(n)
        adj = _build_adj(colors, n)
        tri_count = _count_mono_triangles(adj, n)

        if tri_count < best_tri:
            best_tri = tri_count
            best_result = colors[:]
        if tri_count == 0:
            return colors, 0

        T = 5.0
        T_min = 0.01
        cooling = 0.9995
        accepted = 0
        total_steps = 0

        while _time_left() > 2.0 and time.time() - start_time < time_budget:
            ri = random.randint(0, m - 1)
            old_c = colors[ri]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue

            u, v = idx_to_edge[ri]
            removed = (adj[old_c][u] & adj[old_c][v]).bit_count()
            added = (adj[new_c][u] & adj[new_c][v]).bit_count()
            delta = added - removed
            total_steps += 1

            if delta <= 0:
                colors[ri] = new_c
                adj[old_c][u] &= ~(1 << v)
                adj[old_c][v] &= ~(1 << u)
                adj[new_c][u] |= (1 << v)
                adj[new_c][v] |= (1 << u)
                tri_count += delta
                accepted += 1
                if tri_count < best_tri:
                    best_tri = tri_count
                    best_result = colors[:]
                    if tri_count == 0:
                        return colors, 0
            else:
                if T > T_min:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        colors[ri] = new_c
                        adj[old_c][u] &= ~(1 << v)
                        adj[old_c][v] &= ~(1 << u)
                        adj[new_c][u] |= (1 << v)
                        adj[new_c][v] |= (1 << u)
                        tri_count += delta
                        accepted += 1
            T *= cooling
            if total_steps > 1000 and accepted < total_steps * 0.01:
                break

    return best_result, best_tri

def construct():
    # Guaranteed floor: K32 valid coloring.
    floor_colors = _seed_32()
    floor_n = 32

    # Try to improve via SA on larger n.
    for n in (50, 48, 46, 44):
        if _time_left() < 15.0:
            break
        colors, tri = _sa_search(n, time_budget=_time_left() - 15.0)
        if colors is not None and tri == 0:
            return {'n': n, 'colors': colors}

    return {'n': floor_n, 'colors': floor_colors}
