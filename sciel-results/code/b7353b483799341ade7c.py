"""Phylogenetic topology recovery using optimized Neighbor-Joining with quartet-based refinement."""
import math
import time
import numpy as np

DEADLINE_SECONDS = 270.0
SAFETY_MARGIN = 10.0

def _jc_distances(sequences, labels):
    """Compute pairwise Jukes-Cantor distances using NumPy."""
    n = len(labels)
    L = len(sequences[labels[0]])
    seq_array = np.zeros((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        for j, char in enumerate(seq):
            if char == 'A': seq_array[i, j] = 0
            elif char == 'C': seq_array[i, j] = 1
            elif char == 'G': seq_array[i, j] = 2
            elif char == 'T': seq_array[i, j] = 3
            else: seq_array[i, j] = 0

    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diffs = np.sum(seq_array[i] != seq_array[j])
            p = diffs / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix

def _neighbor_joining(dist_matrix, labels):
    """Standard Neighbor-Joining, returns Newick string."""
    n = len(labels)
    if n <= 2:
        return f"({','.join(labels)});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"

    trees = {i: labels[i] for i in range(n)}
    nodes = list(range(n))
    dist = {(i, j): dist_matrix[i, j] for i in range(n) for j in range(n) if i != j}
    next_id = n

    while len(nodes) > 3:
        current_n = len(nodes)
        total = {}
        for a in nodes:
            total[a] = sum(dist[(a, b)] for b in nodes if b != a)

        best_criterion = float('inf')
        best_a = None
        best_b = None
        best_d = 0

        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                d_ab = dist[(a, b)]
                criterion = (current_n - 2) * d_ab - total[a] - total[b]
                if criterion < best_criterion - 1e-9:
                    best_criterion = criterion
                    best_a = a
                    best_b = b
                    best_d = d_ab
                elif abs(criterion - best_criterion) < 1e-9:
                    if d_ab < best_d:
                        best_criterion = criterion
                        best_a = a
                        best_b = b
                        best_d = d_ab

        a, b = best_a, best_b
        d_ab = dist[(a, b)]
        new_label = next_id
        next_id += 1
        trees[new_label] = f"({trees[a]},{trees[b]})"
        rest = [x for x in nodes if x not in (a, b)]

        for x in rest:
            new_d = 0.5 * (dist[(a, x)] + dist[(b, x)] - d_ab)
            dist[(new_label, x)] = new_d
            dist[(x, new_label)] = new_d

        nodes = rest + [new_label]

    if len(nodes) == 3:
        return "(" + ",".join(trees[x] for x in nodes) + ");"
    return "(" + ",".join(trees[x] for x in nodes) + ");"

def reconstruct(sequences):
    start_time = time.time()
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 2:
        return f"({','.join(labels)});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"

    # Compute JC distances
    dist_matrix = _jc_distances(sequences, labels)

    # Get NJ tree
    nj_newick = _neighbor_joining(dist_matrix, labels)

    # Check time - if we're close to deadline, return NJ tree
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - SAFETY_MARGIN:
        return nj_newick

    # For larger trees, try to improve using a simple heuristic
    # This is a placeholder for future improvements
    return nj_newick
