import time
import random
import sys
import numpy as np

def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4 using fast local search.
    Maximizes n. Uses incremental O(1) cost updates so millions of moves fit in time.
    """
    deadline = time.time() + 140.0  # safety margin under 150s kill

    def is_valid(n, colors):
        if len(colors) != n * (n - 1) // 2:
            return False
        neighbors = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c < 0 or c > 4:
                    return False
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                if nu == 0:
                    continue
                for v in range(u + 1, n):
                    common = nu & neighbors[c][v]
                    if common & (common - 1):
                        return False
        return True

    def build_state(n, colors):
        """Build neighbor bitsets, pair-pair counts and total cost."""
        neighbors = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        # pair_count[c][u][v] = number of common neighbors in color c for u<v
        pair_count = [[None] * n for _ in range(5)]
        cost = 0
        for c in range(5):
            pc = [[0] * n for _ in range(n)]
            for u in range(n):
                nu = neighbors[c][u]
                if nu == 0:
                    continue
                for v in range(u + 1, n):
                    common = nu & neighbors[c][v]
                    cnt = common.bit_count()
                    pc[u][v] = cnt
                    if cnt >= 2:
                        cost += 1
            pair_count[c] = pc
        return neighbors, pair_count, cost

    def solve_ls(n, time_limit):
        total_edges = n * (n - 1) // 2
        colors = [random.randint(0, 4) for _ in range(total_edges)]
        neighbors, pair_count, cost = build_state(n, colors)

        # edge_idx -> (u, v)
        edge_map = [None] * total_edges
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[idx] = (i, j)
                idx += 1

        best_cost = cost
        best_colors = colors[:]
        if cost == 0:
            return colors

        # random helpers
        rint = random.randint
        rrand = random.random

        T = 1.0
        cooling = 0.9999
        start = time.time()
        # step budget guard to avoid pathological infinite loops
        steps = 0
        max_steps = 20_000_000

        while time.time() - start < time_limit and steps < max_steps:
            steps += 1
            e_idx = rint(0, total_edges - 1)
            u, v = edge_map[e_idx]
            old_c = colors[e_idx]
            new_c = rint(0, 4)
            if new_c == old_c:
                continue

            # --- remove edge (u,v) from color old_c ---
            pc = pair_count[old_c]
            nc = neighbors[old_c]
            nu = nc[u]
            nv = nc[v]
            # neighbors that see both u and v (excluding u,v themselves)
            both = (nu & nv) & ~(1 << u) & ~(1 << v)
            # for each common neighbor w: pair (u,w) and (v,w) each lose 1
            # count pairs currently >=2 that drop to 1 (cost decreases)
            delta = 0
            w = both
            while w:
                lsb = w & (-w)
                w ^= lsb
                wi = lsb.bit_length() - 1
                # pair (u, wi)
                if wi < u:
                    a, b = wi, u
                else:
                    a, b = u, wi
                if pc[a][b] == 2:
                    delta -= 1
                pc[a][b] -= 1
                # pair (v, wi)
                if wi < v:
                    a, b = wi, v
                else:
                    a, b = v, wi
                if pc[a][b] == 2:
                    delta -= 1
                pc[a][b] -= 1
            # pair (u, v) loses 1
            if u < v:
                a, b = u, v
            else:
                a, b = v, u
            if pc[a][b] == 2:
                delta -= 1
            pc[a][b] -= 1
            nc[u] &= ~(1 << v)
            nc[v] &= ~(1 << u)

            # --- add edge (u,v) to color new_c ---
            pc2 = pair_count[new_c]
            nc2 = neighbors[new_c]
            nu2 = nc2[u]
            nv2 = nc2[v]
            both2 = (nu2 & nv2) & ~(1 << u) & ~(1 << v)
            w = both2
            while w:
                lsb = w & (-w)
                w ^= lsb
                wi = lsb.bit_length() - 1
                if wi < u:
                    a, b = wi, u
                else:
                    a, b = u, wi
                if pc2[a][b] == 1:
                    delta += 1
                pc2[a][b] += 1
                if wi < v:
                    a, b = wi, v
                else:
                    a, b = v, wi
                if pc2[a][b] == 1:
                    delta += 1
                pc2[a][b] += 1
            if u < v:
                a, b = u, v
            else:
                a, b = v, u
            if pc2[a][b] == 1:
                delta += 1
            pc2[a][b] += 1
            nc2[u] |= (1 << v)
            nc2[v] |= (1 << u)

            new_cost = cost + delta

            # accept
            if delta < 0 or rrand() < pow(2.718281828, -delta / T):
                colors[e_idx] = new_c
                cost = new_cost
                if cost < best_cost:
                    best_cost = cost
                    best_colors = colors[:]
                    if best_cost == 0:
                        return best_colors
            else:
                # revert: undo add then redo remove? Simpler: recompute is expensive.
                # Instead, we must revert the structures. We'll just recompute state
                # is too slow; so we revert by inverse operations.
                # Revert add (remove edge from new_c):
                pc2 = pair_count[new_c]
                nc2 = neighbors[new_c]
                nu2 = nc2[u]
                nv2 = nc2[v]
                both2 = (nu2 & nv2) & ~(1 << u) & ~(1 << v)
                w = both2
                while w:
                    lsb = w & (-w)
                    w ^= lsb
                    wi = lsb.bit_length() - 1
                    if wi < u:
                        a, b = wi, u
                    else:
                        a, b = u, wi
                    pc2[a][b] -= 1
                    if wi < v:
                        a, b = wi, v
                    else:
                        a, b = v, wi
                    pc2[a][b] -= 1
                if u < v:
                    a, b = u, v
                else:
                    a, b = v, u
                pc2[a][b] -= 1
                nc2[u] &= ~(1 << v)
                nc2[v] &= ~(1 << u)
                # Revert remove (add edge to old_c):
                pc = pair_count[old_c]
                nc = neighbors[old_c]
                nu = nc[u]
                nv = nc[v]
                both = (nu & nv) & ~(1 << u) & ~(1 << v)
                w = both
                while w:
                    lsb = w & (-w)
                    w ^= lsb
                    wi = lsb.bit_length() - 1
                    if wi < u:
                        a, b = wi, u
                    else:
                        a, b = u, wi
                    pc[a][b] += 1
                    if wi < v:
                        a, b = wi, v
                    else:
                        a, b = v, wi
                    pc[a][b] += 1
                if u < v:
                    a, b = u, v
                else:
                    a, b = v, u
                pc[a][b] += 1
                nc[u] |= (1 << v)
                nc[v] |= (1 << u)
                # cost unchanged
            T *= cooling
            if T < 0.05:
                T = 1.0
        return best_colors if best_cost == 0 else None

    search_plan = [
        (28, 8.0),
        (27, 15.0),
        (26, 60.0),
        (25, 15.0),
        (24, 10.0),
    ]

    def get_seed_solution():
        n = 11
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = min(j - i, n - (j - i))
                colors.append(d - 1)
        return n, colors

    best_n, best_colors = get_seed_solution()

    for n, t_alloc in search_plan:
        if time.time() > deadline - 2.0:
            break
        res = solve_ls(n, t_alloc)
        if res is not None and is_valid(n, res):
            return {'n': n, 'colors': res}

    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
