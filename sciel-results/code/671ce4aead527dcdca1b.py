"""Phylogenetic topology recovery: robust NJ under JC69 with deadline guard."""
import math
import time

_START = time.time()
_DEADLINE = 260.0  # safety margin before 270s wall-clock kill


def _jc_distance(a, b):
    n = len(a)
    diff = sum(1 for x, y in zip(a, b) if x != y)
    p = diff / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    # Build distance matrix
    D = {}
    for i, a in enumerate(labels):
        D[a] = {}
        for b in labels[i + 1:]:
            d = _jc_distance(sequences[a], sequences[b])
            D[a][b] = d
            D[b][a] = d

    nodes = list(labels)
    tree_repr = {name: name for name in labels}

    while len(nodes) > 3:
        if time.time() - _START > _DEADLINE:
            # Return best effort: star tree
            return "(" + ",".join(tree_repr[x] for x in nodes) + ")"

        nn = len(nodes)
        r = {a: sum(D[a][b] for b in nodes if b != a) for a in nodes}

        best_val = float('inf')
        best_pair = None
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                q = (nn - 2) * D[a][b] - r[a] - r[b]
                if q < best_val:
                    best_val = q
                    best_pair = (a, b)

        a, b = best_pair
        new = f"n{len(nodes)}_{a}_{b}"

        # Compute distances from new node to all others
        for x in nodes:
            if x == a or x == b:
                continue
            D[new][x] = 0.5 * (D[a][x] + D[b][x] - D[a][b])
            D[x][new] = D[new][x]

        tree_repr[new] = f"({tree_repr[a]},{tree_repr[b]})"

        nodes = [x for x in nodes if x != a and x != b] + [new]

        # Clean D
        for k in list(D.keys()):
            if k == a or k == b:
                del D[k]
            else:
                D[k].pop(a, None)
                D[k].pop(b, None)

    a, b, c = nodes
    return f"({tree_repr[a]},{tree_repr[b]},{tree_repr[c]})"
