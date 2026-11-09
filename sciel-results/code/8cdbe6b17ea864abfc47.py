import time
import random

N = 16
DEADLINE = 75.0  # safety margin before 90s kill

# Precomputed optimal colors g(n) for n=1..16
# These values are derived from the maximum number of colors in K_n 
# without a rainbow C4, using a recursive splitting strategy.
G = [0, 0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20]

# Precomputed best split k for n=4..16
SPLIT = {
    4: 3,
    5: 3,
    6: 3,
    7: 3,
    8: 4,
    9: 3,
    10: 4,
    11: 5,
    12: 6,
    13: 6,
    14: 7,
    15: 6,
    16: 7
}

def construct():
    """
    Constructs a Gallai coloring of K_16, then attempts to increase the number
    of distinct colors via local search (edge recoloring and small swaps) while
    maintaining the no-rainbow-C4 constraint.
    """
    start_time = time.time()

    # --- Build initial Gallai coloring ---
    colors = [[0] * N for _ in range(N)]
    next_color = [0]

    def assign_colors(vertices):
        n = len(vertices)
        if n <= 1:
            return next_color[0]
        if n == 2:
            u, v = vertices
            c = next_color[0]
            colors[u][v] = c
            colors[v][u] = c
            next_color[0] += 1
            return next_color[0]
        if n == 3:
            u, v, w = vertices
            c1 = next_color[0]
            c2 = next_color[0] + 1
            c3 = next_color[0] + 2
            colors[u][v] = c1; colors[v][u] = c1
            colors[v][w] = c2; colors[w][v] = c2
            colors[u][w] = c3; colors[w][u] = c3
            next_color[0] += 3
            return next_color[0]
        k = SPLIT[n]
        v1 = vertices[:k]
        v2 = vertices[k:]
        cross_color = next_color[0]
        for u in v1:
            for v in v2:
                colors[u][v] = cross_color
                colors[v][u] = cross_color
        next_color[0] += 1
        assign_colors(v1)
        assign_colors(v2)
        return next_color[0]

    assign_colors(list(range(N)))

    # --- Precompute all 4-vertex subsets and their 3 four-cycles ---
    # A 4-cycle on vertices {a,b,c,d} in K4 has 3 possible cycles:
    # (a,b,c,d), (a,b,d,c), (a,c,b,d) — i.e., the 3 perfect-matchings complement
    # Actually in K4, the three 4-cycles are the three ways to arrange 4 vertices
    # in a cycle up to rotation/reflection. They are:
    # cycle1: a-b-c-d-a  (edges ab, bc, cd, da)
    # cycle2: a-b-d-c-a  (edges ab, bd, dc, ca)
    # cycle3: a-c-b-d-a  (edges ac, cb, bd, da)
    all_quads = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Three 4-cycles on {a,b,c,d}:
                    # C1: a-b-c-d-a
                    # C2: a-b-d-c-a
                    # C3: a-c-b-d-a
                    cycles = [
                        [(a, b), (b, c), (c, d), (d, a)],
                        [(a, b), (b, d), (d, c), (c, a)],
                        [(a, c), (c, b), (b, d), (d, a)],
                    ]
                    all_quads.append(cycles)

    def is_valid(colors):
        """Check no 4-cycle has four distinct colors."""
        for cycles in all_quads:
            for cyc in cycles:
                cs = set()
                for (u, v) in cyc:
                    cs.add(colors[u][v])
                    if len(cs) == 4:
                        return False
        return True

    def count_colors(colors):
        seen = set()
        for i in range(N):
            for j in range(i + 1, N):
                seen.add(colors[i][j])
        return len(seen)

    best_colors = [row[:] for row in colors]
    best_count = count_colors(best_colors)

    # --- Local search: try recoloring individual edges ---
    rng = random.Random(42)
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]

    def try_recolor(colors, edge, new_color):
        """Try recoloring `edge` to `new_color`; return True if valid."""
        u, v = edge
        old_color = colors[u][v]
        if old_color == new_color:
            return False
        colors[u][v] = new_color
        colors[v][u] = new_color
        # Check only 4-cycles containing this edge
        # An edge (u,v) appears in 4-cycles with any pair (w,x) where
        # {w,x} is disjoint from {u,v} and the cycle uses edge (u,v).
        # For speed, check all quads containing both u and v.
        for cycles in all_quads:
            # Check if this quad contains both u and v
            # We stored cycles as lists of 4 edges; check if (u,v) or (v,u) is in any cycle
            for cyc in cycles:
                if (u, v) in cyc or (v, u) in cyc:
                    cs = set()
                    for (a, b) in cyc:
                        cs.add(colors[a][b])
                        if len(cs) == 4:
                            colors[u][v] = old_color
                            colors[v][u] = old_color
                            return False
        return True

    # Iterate: try to introduce new colors or recolor edges to unused colors
    max_iter = 5000
    for it in range(max_iter):
        if time.time() - start_time > DEADLINE:
            break
        # Pick a random edge
        edge = rng.choice(all_edges)
        # Try a random new color in 0..119
        new_color = rng.randint(0, 119)
        if try_recolor(colors, edge, new_color):
            new_count = count_colors(colors)
            if new_count > best_count:
                best_count = new_count
                best_colors = [row[:] for row in colors]
        else:
            # Revert (try_recolor already reverted on failure)
            pass

    # Also try: for edges with the most common color, try splitting them
    # (this is implicitly covered by random recoloring)

    result = []
    for i in range(N):
        for j in range(i + 1, N):
            result.append(best_colors[i][j])
    return result

if __name__ == "__main__":
    # Simple self-check
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Number of colors used: {len(set(res))}")
    print(f"First 10 colors: {res[:10]}")
    print(f"Time: {t1 - t0:.2f}s")
