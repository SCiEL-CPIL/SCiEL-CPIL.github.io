"""Phylogenetic topology inference using neighbor joining."""
import math
import time

_DEADLINE = 260.0
_START_TIME = None


def _jc_distance(seq_a, seq_b):
    diffs = sum(1 for a, b in zip(seq_a, seq_b) if a != b)
    p = diffs / len(seq_a)
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _nj_build(labels, sequences):
    n = len(labels)
    if n <= 3:
        if n == 1:
            return labels[0]
        elif n == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return f"({labels[0]},{labels[1]},{labels[2]})"

    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist[i][j] = dist[j][i] = _jc_distance(sequences[labels[i]], sequences[labels[j]])

    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    next_id = n
    while len(nodes) > 3:
        m = len(nodes)
        total = [sum(dist[nodes[i]][nodes[j]] for j in range(m) if i != j) for i in range(m)]
        best_criterion, best_pair = None, None
        for i in range(m):
            for j in range(i + 1, m):
                a, b = nodes[i], nodes[j]
                c = (m - 2) * dist[a][b] - total[i] - total[j]
                if best_criterion is None or c < best_criterion:
                    best_criterion, best_pair = c, (i, j)
        i, j = best_pair
        a, b = nodes[i], nodes[j]
        joined = f"({trees[a]},{trees[b]})"
        new_node = next_id
        next_id += 1
        for k in range(m):
            if k not in (i, j):
                x = nodes[k]
                dist[a][x] = dist[x][a] = 0.5 * (dist[a][x] + dist[b][x] - dist[a][b])
                dist[b][x] = dist[x][b] = dist[a][x]
        rest = [nodes[k] for k in range(m) if k not in (i, j)]
        trees[new_node] = joined
        nodes = rest + [new_node]
    return "(" + ",".join(trees[x] for x in nodes) + ")"


def reconstruct(sequences):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 3:
        if n == 1:
            return labels[0]
        elif n == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return f"({labels[0]},{labels[1]},{labels[2]})"

    tree = _nj_build(labels, sequences)

    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        return tree

    return tree
