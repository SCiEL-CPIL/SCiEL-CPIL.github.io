import time
import random
import itertools
from collections import defaultdict

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit

    # Edge index for all pairs (i, j) with 0 <= i < j < N
    edge_idx = [[-1] * N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1

    # Precompute all 4-cycles (3 per 4-vertex subset) as tuples of 4 edge indices
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        cycles.append((edge_idx[a][b], edge_idx[b][c], edge_idx[c][d], edge_idx[d][a]))
        cycles.append((edge_idx[a][b], edge_idx[b][d], edge_idx[d][c], edge_idx[c][a]))
        cycles.append((edge_idx[a][c], edge_idx[c][b], edge_idx[b][d], edge_idx[d][a]))

    # For each edge, the "other three" edge indices of every 4-cycle containing it.
    edge_others = [[] for _ in range(NUM_EDGES)]
    for c in cycles:
        for k in range(4):
            others = [c[(k + 1) % 4], c[(k + 2) % 4], c[(k + 3) % 4]]
            edge_others[c[k]].append(others)

    def is_bad(e1, e2, e3, e4, colors):
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        return c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4

    def is_valid(colors):
        for c in cycles:
            if is_bad(c[0], c[1], c[2], c[3], colors):
                return False
        return True

    # ---- Build candidate starting colorings ----
    candidates = []

    # 1) Recursive bisection (Gallai coloring) -> 15 colors, guaranteed valid
    def make_bisection():
        cols = [0] * NUM_EDGES
        cid = 0
        def rec(vertices):
            nonlocal cid
            if len(vertices) <= 1:
                return
            mid = len(vertices) // 2
            left, right = vertices[:mid], vertices[mid:]
            for u in left:
                for v in right:
                    cols[edge_idx[u][v]] = cid
            cid += 1
            rec(left)
            rec(right)
        rec(list(range(N)))
        return cols
    candidates.append(make_bisection())

    # 2) Modular arithmetic colorings c(i,j) = (a*i + b*j) mod m
    for a, b, m in [(1, 0, 19), (0, 1, 19), (1, 1, 19), (1, 2, 19), (2, 1, 19),
                    (1, 3, 19), (3, 1, 19), (1, 4, 19), (4, 1, 19),
                    (1, 1, 17), (1, 2, 17), (2, 1, 17), (1, 3, 17), (3, 1, 17),
                    (1, 1, 13), (1, 2, 13), (2, 1, 13), (1, 3, 13), (3, 1, 13),
                    (1, 1, 11), (1, 2, 11), (2, 1, 11), (1, 3, 11), (3, 1, 11),
                    (1, 1, 7), (1, 2, 7), (2, 1, 7), (1, 3, 7), (3, 1, 7),
                    (1, 1, 5), (1, 2, 5), (2, 1, 5)]:
        cols = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                cols[edge_idx[i][j]] = (a * i + b * j) % m
        candidates.append(cols)

    # Keep only valid candidates, ranked by number of colors used (desc)
    valid_cands = []
    for cols in candidates:
        if is_valid(cols):
            valid_cands.append((len(set(cols)), cols))
    valid_cands.sort(key=lambda t: -t[0])

    if not valid_cands:
        cols = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                cols[edge_idx[i][j]] = i % 7
        return cols

    # ---- Structural change: Simulated Annealing with color-introduction bias ----
    random.seed(42)
    best_colors = None
    best_num = -1

    for seed_num, seed_cols in valid_cands:
        if time.time() > deadline - 8:
            break
        if seed_num + 1 <= best_num:
            continue

        colors = seed_cols[:]
        freq = [0] * MAX_COLORS
        for c in colors:
            freq[c] += 1
        cur_num = len(set(colors))
        cur_temp = 1.0

        if cur_num > best_num:
            best_num = cur_num
            best_colors = colors[:]

        local_deadline = deadline - 8
        no_improve_steps = 0
        max_no_improve = 2000

        while time.time() < local_deadline:
            # Pick random edge
            e = random.randint(0, NUM_EDGES - 1)
            old = colors[e]

            # Build allowed color set
            allowed = None
            for others in edge_others[e]:
                o1, o2, o3 = others
                c1, c2, c3 = colors[o1], colors[o2], colors[o3]
                s = {c1, c2, c3}
                if allowed is None:
                    allowed = set(s)
                else:
                    allowed &= s
                if not allowed:
                    break

            if allowed is None or not allowed:
                continue

            # Choose new color with bias toward new colors
            unused_allowed = [x for x in allowed if freq[x] == 0]
            if unused_allowed:
                # Strong bias: 80% chance pick unused, 20% pick any allowed
                if random.random() < 0.8:
                    new = random.choice(unused_allowed)
                else:
                    new = random.choice(list(allowed))
            else:
                new = random.choice(list(allowed))

            if new == old:
                continue

            # Calculate delta in number of distinct colors
            old_freq = freq[old]
            new_freq = freq[new]
            delta_num = 0
            if old_freq == 1:
                delta_num -= 1
            if new_freq == 0:
                delta_num += 1

            # Accept or reject based on simulated annealing
            if delta_num >= 0:
                accept = True
            else:
                # delta_num < 0: reject with probability exp(delta_num / temp)
                if random.random() > (delta_num / cur_temp):
                    accept = False
                else:
                    accept = True

            if accept:
                colors[e] = new
                freq[old] -= 1
                freq[new] += 1
                cur_num += delta_num
                no_improve_steps = 0

                if cur_num > best_num:
                    best_num = cur_num
                    best_colors = colors[:]
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1

                # Cool down
                cur_temp *= 0.9995
                if cur_temp < 0.01:
                    cur_temp = 0.01

                if cur_num >= MAX_COLORS:
                    break
            else:
                no_improve_steps += 1

            # Restart if stuck for too long
            if no_improve_steps > max_no_improve:
                no_improve_steps = 0
                cur_temp = min(cur_temp * 3, 2.0)

    if best_colors is None:
        best_colors = valid_cands[0][1][:]
    return best_colors
# EVOLVE-BLOCK-END
