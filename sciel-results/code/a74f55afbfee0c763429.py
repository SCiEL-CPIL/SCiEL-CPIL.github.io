"""Heuristic matrix for ant colony optimization of capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree


def heuristics(distance_matrix, coordinates, demands, capacity):
    n = distance_matrix.shape[0]
    coords = np.asarray(coordinates, dtype=float)
    demands = np.asarray(demacts if False else demands, dtype=float)
    cap = float(capacity)

    # --- Greedy insertion tour (2-opt-ish local improvement) to get a sensible
    # --- visit order, used only to estimate residual vehicle capacity per node.
    order = np.arange(1, n)
    tour = [0]
    remaining = set(order.tolist())
    while remaining:
        last = tour[-1]
        best_j = None
        best_cost = np.inf
        for j in remaining:
            c = distance_matrix[last, j]
            if c < best_cost:
                best_cost = c
                best_j = j
        tour.append(best_j)
        remaining.remove(best_j)
    tour = np.asarray(tour)

    # 2-opt local improvement on the tour (depot fixed at both ends)
    improved = True
    while improved:
        improved = False
        m = len(tour)
        for i in range(1, m - 2):
            for k in range(i + 1, m - 1):
                a, b, c, d = tour[i - 1], tour[i], tour[k], tour[k + 1]
                delta = (distance_matrix[a, c] + distance_matrix[b, d]) - \
                        (distance_matrix[a, b] + distance_matrix[c, d])
                if delta < -1e-12:
                    tour[i:k + 1] = tour[i:k + 1][::-1]
                    improved = True

    # --- Visit position -> residual vehicle capacity estimate.
    # Walk the tour from the depot, accumulating load; whenever the load would
    # exceed capacity, start a new vehicle. residual[pos] = capacity - load
    # just before visiting that node (i.e. free space when arriving).
    pos_of = np.zeros(n, dtype=int)
    residual = np.zeros(n, dtype=float)
    load = 0.0
    for idx, node in enumerate(tour):
        if load + demands[node] > cap:
            load = 0.0
        pos_of[node] = idx
        residual[node] = cap - load
        load += demands[node]

    # For the depot, use the full capacity (a fresh vehicle departs).
    residual[0] = cap

    # --- Edge attractiveness.
    # Base: inverse distance (closer = more attractive).
    base = 1.0 / (distance_matrix + 1e-9)

    # Capacity-potential factor: ratio of destination demand to the residual
    # capacity available when the destination is visited. Small ratio -> safe
    # (lots of slack) -> boost; large ratio (near/over capacity) -> penalize.
    # Use a smooth reciprocal so we never hit zero and keep it positive.
    dest_res = residual  # per-destination residual capacity
    ratio = demands / (dest_res + 1e-9)  # demand / residual capacity
    # Attractiveness multiplier: 1/(1+ratio^p) strongly penalizes tight fits.
    p = 2.0
    cap_factor = 1.0 / (1.0 + np.power(ratio, p))

    # Broadcast: edge (i,j) gets destination j's capacity factor.
    heuristic = base * cap_factor[None, :]

    # Guarantee positivity and finiteness.
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0),
                         heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
