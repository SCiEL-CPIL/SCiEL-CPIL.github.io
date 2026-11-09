"""Optimized heuristics for ACO CVRP: fast vectorized ACO sim + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _aco_sim(dm, demands, capacity, heuristic, n_ants=20, n_iters=30, seed=12345):
    """Fast ACO simulation. Returns best total route distance found by any ant."""
    n = dm.shape[0]
    d = demands.astype(float)
    d[0] = 0.0
    rng = np.random.default_rng(seed)
    pher = np.ones((n, n))
    # Precompute per-destination demand to vectorize capacity checks
    best_ever = float('inf')
    for _ in range(n_iters):
        iter_best = float('inf')
        for _ in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            cur = 0
            load = 0.0
            dist = 0.0
            while True:
                # feasible: unvisited customer that fits remaining load
                mask = ~visited
                mask[0] = False
                mask &= (d <= (capacity - load))
                if not mask.any():
                    # return to depot and reload
                    dist += dm[cur, 0]
                    cur = 0
                    load = 0.0
                    if visited.all():
                        break
                    continue
                idxs = np.nonzero(mask)[0]
                weights = pher[cur, idxs] * heuristic[cur, idxs]
                weights = np.maximum(weights, 1e-9)
                r = rng.random() * weights.sum()
                c = 0
                cum = 0.0
                for w, j in zip(weights, idxs):
                    cum += w
                    if r <= cum:
                        c = j
                        break
                else:
                    c = idxs[-1]
                dist += dm[cur, c]
                visited[c] = True
                load += d[c]
                cur = c
            if dist < iter_best:
                iter_best = dist
            if dist < best_ever:
                best_ever = dist
        # pheromone update: decay + add 1/best
        pher *= 0.9
        pher += 1.0 / iter_best
    return best_ever


def heuristics(distance_matrix, coordinates, demands, capacity):
    deadline = time.time() + 300.0
    n = distance_matrix.shape[0]
    d = demands.astype(float)
    d[0] = 0.0
    maxd = max(d.max(), 1e-9)
    dr = d / maxd  # demand ratio in [0,1]

    def build(p):
        alpha, beta, gamma, delta, eta = p
        dm = distance_matrix
        base = 1.0 / (dm ** alpha + 1e-9)
        # demand-aware scaling on destination
        dscale = 1.0 / (1.0 + gamma * dr)
        H = base * dscale[None, :]
        # nearest-neighbor boost
        if beta > 1e-6:
            nn = np.ones((n, n))
            for i in range(n):
                if i == 0:
                    continue
                dists = dm[i, 1:].copy()
                k = min(3, n - 1)
                nn_idx = np.argpartition(dists, k - 1)[:k]
                nn[i, nn_idx + 1] = beta
            H = H * nn
        # depot boost
        if delta > 1e-6:
            H[0, :] *= delta
            H[:, 0] *= delta
        # slight global scaling is irrelevant to ratios; keep as is
        return np.maximum(H, 1e-9)

    def score(p):
        H = build(p)
        return _aco_sim(distance_matrix, demands, capacity, H)

    # Parameter space: alpha, beta, gamma, delta, eta
    # Start with a good baseline and a handful of diverse candidates.
    candidates = [
        [1.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.1, 1.0, 0.0],
        [1.0, 2.0, 0.0, 1.0, 0.0],
        [0.8, 1.5, 0.05, 1.2, 0.0],
        [1.0, 1.5, 0.05, 1.1, 0.0],
        [1.2, 1.0, 0.15, 1.0, 0.0],
    ]

    best_p = None
    best_s = float('inf')
    for p in candidates:
        if time.time() > deadline - 60:
            break
        try:
            s = score(p)
            if np.isfinite(s) and s < best_s:
                best_s = s
                best_p = p
        except Exception:
            continue

    # Local search (hill climbing with random perturbations) around the best.
    if best_p is None:
        best_p = [1.0, 1.0, 0.0, 1.0, 0.0]
        try:
            best_s = score(best_p)
        except Exception:
            best_s = float('inf')

    # Bounds for parameters
    bounds = [(0.5, 2.0), (1.0, 3.0), (0.0, 0.5), (1.0, 1.5), (0.0, 0.0)]
    rng2 = np.random.default_rng(999)
    step_sizes = [0.2, 0.3, 0.05, 0.1, 0.0]
    for it in range(400):
        if time.time() > deadline - 30:
            break
        # perturb a random subset of params
        new_p = list(best_p)
        for k in range(5):
            if rng2.random() < 0.6:
                new_p[k] = best_p[k] + rng2.uniform(-step_sizes[k], step_sizes[k])
                lo, hi = bounds[k]
                new_p[k] = min(max(new_p[k], lo), hi)
        try:
            s = score(new_p)
        except Exception:
            continue
        if np.isfinite(s) and s < best_s - 1e-9:
            best_s = s
            best_p = new_p
            # shrink step sizes on improvement
            step_sizes = [s2 * 0.85 for s2 in step_sizes]

    H = build(best_p)
    H = np.maximum(H, 1e-9)
    # sanity: ensure finite and correct shape
    if not np.all(np.isfinite(H)) or H.shape != distance_matrix.shape:
        H = np.maximum(1.0 / (distance_matrix + 1e-9), 1e-9)
    return H
# EVOLVE-BLOCK-END
