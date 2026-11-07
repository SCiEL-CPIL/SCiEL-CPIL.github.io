# EVOLVE-BLOCK-START
import time
import numpy as np


def _nearest_neighbor_tour(d, start=0, rng=None):
    n = d.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    cur = start
    for _ in range(n - 1):
        row = d[cur].copy()
        row[visited] = np.inf
        nxt = int(np.argmin(row))
        visited[nxt] = True
        tour.append(nxt)
        cur = nxt
    return np.array(tour, dtype=int)


def _tour_length(tour, d):
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += d[tour[i], tour[(i + 1) % n]]
    return total


def _two_opt(tour, d, max_passes=60, improve_tol=1e-9):
    n = len(tour)
    tour = tour.copy()
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, n - 1):
            a, b = tour[i - 1], tour[i]
            for j in range(i + 1, n):
                c, e = tour[j], tour[(j + 1) % n]
                # current edges a-b and c-e ; new edges a-c and b-e
                delta = (d[a, c] + d[b, e]) - (d[a, b] + d[c, e])
                if delta < -improve_tol:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
    return tour


def _best_tour(d, n_restarts=6, seed=12345):
    n = d.shape[0]
    rng = np.random.default_rng(seed)
    best = None
    best_len = np.inf
    # deterministic NN from node 0
    t = _nearest_neighbor_tour(d, start=0)
    t = _two_opt(t, d)
    L = _tour_length(t, d)
    if L < best_len:
        best_len, best = L, t.copy()
    # randomized NN restarts from random starts
    for _ in range(n_restarts):
        s = int(rng.integers(0, n))
        t = _nearest_neighbor_tour(d, start=s)
        t = _two_opt(t, d)
        L = _tour_length(t, d)
        if L < best_len:
            best_len, best = L, t.copy()
    return best


def heuristics(distance_matrix):
    deadline = time.time() + 240.0  # safety margin well under 300s
    d = np.asarray(distance_matrix, dtype=float)
    n = d.shape[0]

    # Build a strong solution-based tour (NN + 2-Opt with a few restarts).
    tour = _best_tour(d, n_restarts=6, seed=7)

    # Continuous inverse-distance floor (gives gradient info everywhere).
    eps = 1e-9
    base = 1.0 / (d + eps)

    # Strong bias on the edges of the near-optimal tour (in both directions,
    # since the ACO may traverse either way; the pheromone deposition is
    # directional but the heuristic is a symmetric prior).
    tour_mask = np.zeros((n, n), dtype=float)
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        tour_mask[a, b] = 1.0
        tour_mask[b, a] = 1.0

    # Blend: a large boost on tour edges keeps the ACO near the good tour,
    # while the continuous base preserves enough exploration to refine.
    boost = 40.0
    heuristic = base + boost * tour_mask

    # Guarantee positivity and finiteness.
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)

    if time.time() > deadline:
        # Fallback: pure inverse distance (always valid).
        return np.maximum(base, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
