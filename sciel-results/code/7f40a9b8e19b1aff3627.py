"""Structural change: quartet-based topology inference using distance-based quartet scoring."""
import math
import time
from itertools import combinations

# EVOLVE-BLOCK-START

_DEADLINE_FRACTION = 0.85

def _jc_distance(a, b):
    n = len(a)
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _build_distance_matrix(sequences, labels):
    n = len(labels)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i][j] = d
            dist[j][i] = d
    return dist


def _quartet_best_split(dist, i, j, k, l):
    s12_34 = dist[i][j] + dist[k][l]
    s13_24 = dist[i][k] + dist[j][l]
    s14_23 = dist[i][l] + dist[j][k]
    if s12_34 <= s13_24 and s12_34 <= s14_23:
        return frozenset({i, j})
    elif s13_24 <= s12_34 and s13_24 <= s14_23:
        return frozenset({i, k})
    else:
        return frozenset({i, l})


def _nj_newick(sequences, labels, dist):
    n = len(labels)
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    D = [row[:] for row in dist]

    next_id = n
    while len(nodes) > 3:
        m = len(nodes)
        total = [0.0] * m
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if i != j:
                    total[i] += D[a][b]

        best_criterion = float('inf')
        best_pair = (0, 1)
        for i in range(m):
            for j in range(i + 1, m):
                a, b = nodes[i], nodes[j]
                criterion = (m - 2) * D[a][b] - total[i] - total[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_pair = (i, j)

        i, j = best_pair
        a, b = nodes[i], nodes[j]

        r_a = (total[i] - total[j]) / (2 * (m - 2))
        len_a = 0.5 * D[a][b] + r_a
        len_b = 0.5 * D[a][b] - r_a

        joined = f"({trees[a]}:{len_a:.6f},{trees[b]}:{len_b:.6f})"
        new_id = next_id
        next_id += 1

        rest = [x for k, x in enumerate(nodes) if k not in (i, j)]
        for x in rest:
            D[new_id][x] = D[x][new_id] = 0.5 * (D[a][x] + D[b][x] - D[a][b])

        nodes = rest + [new_id]
        trees[new_id] = joined

    a, b, c = nodes
    return f"({trees[a]},{trees[b]},{trees[c]})"


def _exhaustive_n4(dist, labels):
    i, j, k, l = 0, 1, 2, 3
    split = _quartet_best_split(dist, i, j, k, l)
    if split == frozenset({0, 1}):
        return f"(({labels[0]},{labels[1]}),({labels[2]},{labels[3]}))"
    elif split == frozenset({0, 2}):
        return f"(({labels[0]},{labels[2]}),({labels[1]},{labels[3]}))"
    else:
        return f"(({labels[0]},{labels[3]}),({labels[1]},{labels[2]}))"


def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 270 * _DEADLINE_FRACTION

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    dist = _build_distance_matrix(sequences, labels)

    if n == 4:
        return _exhaustive_n4(dist, labels)

    return _nj_newick(sequences, labels, dist)

# EVOLVE-BLOCK-END
