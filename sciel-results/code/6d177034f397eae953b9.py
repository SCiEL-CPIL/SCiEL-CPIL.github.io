import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    # Edge index table
    edge_idx = [[-1] * N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1

    # Precompute the three 4-cycles per 4-set, and edge -> cycles map
    cycles = []
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]

    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = edge_idx[a][b]
        e2 = edge_idx[b][c]
        e3 = edge_idx[c][d]
        e4 = edge_idx[d][a]
        cid = len(cycles)
        cycles.append((e1, e2, e3, e4))
        edge_to_cycles[e1].append(cid)
        edge_to_cycles[e2].append(cid)
        edge_to_cycles[e3].append(cid)
        edge_to_cycles[e4].append(cid)

        # Cycle 2: a-b-d-c-a
        e1 = edge_idx[a][b]
        e2 = edge_idx[b][d]
        e3 = edge_idx[d][c]
        e4 = edge_idx[c][a]
        cid = len(cycles)
        cycles.append((e1, e2, e3, e4))
        edge_to_cycles[e1].append(cid)
        edge_to_cycles[e2].append(cid)
        edge_to_cycles[e3].append(cid)
        edge_to_cycles[e4].append(cid)

        # Cycle 3: a-c-b-d-a
        e1 = edge_idx[a][c]
        e2 = edge_idx[c][b]
        e3 = edge_idx[b][d]
        e4 = edge_idx[d][a]
        cid = len(cycles)
        cycles.append((e1, e2, e3, e4))
        edge_to_cycles[e1].append(cid)
        edge_to_cycles[e2].append(cid)
        edge_to_cycles[e3].append(cid)
        edge_to_cycles[e4].append(cid)

    def is_bad_cycle(colors, c_idx):
        e1, e2, e3, e4 = c_idx
        c1 = colors[e1]
        c2 = colors[e2]
        c3 = colors[e3]
        c4 = colors[e4]
        return c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4

    # Recursive bisection initial coloring (always Gallai, no rainbow 4-cycle)
    def make_bisection():
        colors = [0] * NUM_EDGES
        color_id = [0]
        def process_partition(vertices):
            if len(vertices) <= 1:
                return
            mid = len(vertices) // 2
            left = vertices[:mid]
            right = vertices[mid:]
            for u in left:
                for v in right:
                    colors[edge_idx[u][v]] = color_id[0]
            color_id[0] += 1
            process_partition(left)
            process_partition(right)
        process_partition(list(range(N)))
        return colors

    def count_bad(colors):
        return sum(1 for c in cycles if is_bad_cycle(colors, c))

    best_score = -1
    best_colors = None
    random.seed(42)

    while time.time() < deadline:
        colors = make_bisection()
        freq = [0] * MAX_COLORS
        for c in colors:
            freq[c] += 1
        bad_count = count_bad(colors)  # expected 0

        cur_score = len(set(colors))
        if cur_score > best_score:
            best_score = cur_score
            best_colors = colors.copy()

        # Local search: single-edge recolors with delta accounting
        # + occasional swap to free a color, then recolor.
        no_improve = 0
        while time.time() < deadline - 1.0:
            e = random.randint(0, NUM_EDGES - 1)
            old_color = colors[e]

            # Choose a candidate new color, biased toward unused colors
            if random.random() < 0.6:
                unused = [i for i in range(MAX_COLORS) if freq[i] == 0]
                if unused:
                    new_color = random.choice(unused)
                else:
                    new_color = random.randint(0, MAX_COLORS - 1)
            else:
                new_color = random.randint(0, MAX_COLORS - 1)

            if new_color == old_color:
                continue

            # Delta in bad_count if we set colors[e] = new_color
            delta_bad = 0
            for cid in edge_to_cycles[e]:
                c_idx = cycles[cid]
                was_bad = is_bad_cycle(colors, c_idx)
                colors[e] = new_color
                is_bad_now = is_bad_cycle(colors, c_idx)
                colors[e] = old_color
                if was_bad and not is_bad_now:
                    delta_bad -= 1
                elif (not was_bad) and is_bad_now:
                    delta_bad += 1

            if delta_bad <= 0:
                colors[e] = new_color
                freq[old_color] -= 1
                freq[new_color] += 1
                bad_count += delta_bad
                cur_score = len(set(colors))
                if cur_score > best_score:
                    best_score = cur_score
                    best_colors = colors.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Try a swap to free room: swap e with e2, then try new color
                if random.random() < 0.3:
                    e2 = random.randint(0, NUM_EDGES - 1)
                    if e2 == e:
                        continue
                    c1, c2 = colors[e], colors[e2]
                    if c1 == c2:
                        continue
                    colors[e], colors[e2] = c2, c1
                    # Now try to set e to a fresh unused color
                    unused = [i for i in range(MAX_COLORS) if freq[i] == 0]
                    if unused:
                        nc = random.choice(unused)
                    else:
                        nc = random.randint(0, MAX_COLORS - 1)
                    if nc == colors[e]:
                        colors[e], colors[e2] = c1, c2
                        continue
                    # delta for setting e (already swapped) to nc
                    db = 0
                    for cid in edge_to_cycles[e]:
                        c_idx = cycles[cid]
                        was_bad = is_bad_cycle(colors, c_idx)
                        colors[e] = nc
                        is_bad_now = is_bad_cycle(colors, c_idx)
                        colors[e] = c2  # restore swapped state
                        if was_bad and not is_bad_now:
                            db -= 1
                        elif (not was_bad) and is_bad_now:
                            db += 1
                    if db <= 0:
                        colors[e] = nc
                        freq[c2] -= 1
                        freq[nc] += 1
                        bad_count += db
                        cur_score = len(set(colors))
                        if cur_score > best_score:
                            best_score = cur_score
                            best_colors = colors.copy()
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        colors[e], colors[e2] = c1, c2

            if no_improve > 400:
                break

    return best_colors

if __name__ == "__main__":
    c = construct()
    print(len(set(c)))
