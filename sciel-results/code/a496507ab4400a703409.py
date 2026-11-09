"""Gallai coloring of K_16: recursive partition with monochromatic cut edges."""
import time
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

_CYCLES = []
for a, b, c, d in combinations(range(N), 4):
    _CYCLES.append((_edge_idx(a,b), _edge_idx(b,c), _edge_idx(c,d), _edge_idx(d,a)))
    _CYCLES.append((_edge_idx(a,b), _edge_idx(b,d), _edge_idx(d,c), _edge_idx(c,a)))
    _CYCLES.append((_edge_idx(a,c), _edge_idx(c,b), _edge_idx(b,d), _edge_idx(d,a)))

# EVOLVE-BLOCK-START

def construct():
    t0 = time.time()
    hard_deadline = t0 + 58.0

    colors = [0] * NUM_EDGES
    ctr = [0]

    def nxt():
        v = ctr[0]
        ctr[0] += 1
        return v

    def fill_distinct(vs):
        for x in range(len(vs)):
            for y in range(x + 1, len(vs)):
                colors[_edge_idx(vs[x], vs[y])] = nxt()

    def fill_cut(A, B):
        c = nxt()
        for u in A:
            for v in B:
                colors[_edge_idx(u, v)] = c

    # Six leaves: sizes 3,3,3,3,3,1 summing to 16
    L = (
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (9, 10, 11),
        (12, 13, 14),
        (15,),
    )

    for leaf in L:
        if len(leaf) >= 2:
            fill_distinct(leaf)

    # Balanced binary tree over the six leaves:
    #   internal node 0: L0 vs L1  -> S01  = (0..5)
    #   internal node 1: L2 vs L3  -> S23  = (6..11)
    #   internal node 2: S01 vs S23 -> S0123 = (0..11)
    #   internal node 3: L4 vs L5  -> S45  = (12..15)
    #   root:            S0123 vs S45 -> (0..15)
    if time.time() > hard_deadline:
        return colors

    fill_cut(L[0], L[1])            # node 0
    fill_cut(L[2], L[3])            # node 1

    s0123 = list(L[0]) + list(L[1]) + list(L[2]) + list(L[3])
    fill_cut(s0123, list(L[4]) + list(L[5]))  # root

    s01 = list(L[0]) + list(L[1])
    s23 = list(L[2]) + list(L[3])
    fill_cut(s01, s23)              # node 2

    fill_cut(L[4], L[5])            # node 3

    if time.time() > hard_deadline:
        return colors

    # Quick validity check
    for cyc in _CYCLES:
        if len({colors[i] for i in cyc}) == 4:
            break

    return colors

# EVOLVE-BLOCK-END
