"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# ACO parameters (fixed by contract)
N_ANTS = 20
N_ITERS = 30
RHO = 0.9   # evaporation
Q = 10.0    # pheromone deposit divisor


def _run_aco(prizes, dist, max_len, heuristic):
    """Run the ACO and return the best total prize collected by any ant."""
    n = len(prizes)
    pheromone = np.ones((n, n), dtype=float)
    best_total = 0.0
    d_to_0 = dist[:, 0]

    for _ in range(N_ITERS):
        iter_best = 0.0

        for _ in range(N_ANTS):
            current = 0
            total_len = 0.0
            total_prize = 0.0  # prize at node 0 is 0
            visited_mask = np.zeros(n, dtype=bool)
            visited_mask[0] = True

            while True:
                # Candidates: unvisited nodes that keep total length within max_len
                d_ij = dist[current]
                new_len = total_len + d_ij + d_to_0
                cand_mask = (~visited_mask) & (new_len <= max_len)
                if not cand_mask.any():
                    break

                idx = np.nonzero(cand_mask)[0]
                probs = pheromone[current, idx] * heuristic[current, idx]
                probs = np.maximum(probs, 1e-9)
                s = probs.sum()
                if s <= 0 or not np.isfinite(s):
                    probs = np.ones_like(idx, dtype=float)
                else:
                    probs = probs / s

                r = np.random.random()
                cumsum = np.cumsum(probs)
                chosen = np.searchsorted(cumsum, r)
                if chosen >= len(idx):
                    chosen = len(idx) - 1

                j = idx[chosen]
                total_len += dist[current, j]
                total_prize += prizes[j]
                visited_mask[j] = True
                current = j

        pheromone *= RHO
        pheromone += iter_best / Q

        if iter_best > best_total:
            best_total = iter_best

    return best_total


def _make_heuristic(prizes, dist, params):
    """Build heuristic from a 4-dim parameter vector [a, b, c, d]."""
    a, b, c, d = params
    a = max(a, 0.0)
    b = max(b, 0.0)
    c = max(c, 0.0)
    d = max(d, 0.0)

    prize = prizes[None, :]
    d_ij = dist
    d_j0 = dist[:, 0][None, :]
    detour = d_ij + d_j0

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        h = (prize ** a) * (d_ij + 1e-9) ** (-b) * (d_j0 + 1e-9) ** (-c) * (detour + 1e-9) ** (-d)

    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    return h


def _score(params, prizes, dist, max_len):
    heuristic = _make_heuristic(prizes, dist, params)
    return _run_aco(prizes, dist, max_len, heuristic)


def _optimize(prizes, dist, max_len, time_limit):
    """Simulated annealing with restarts over the 4-dim parameter space."""
    n = len(prizes)
    best_params = np.array([1.0, 1.0, 0.5, 0.0])
    best_score = -1.0

    # Bounds for the parameter space
    lo = np.array([0.0, 0.0, 0.0, 0.0])
    hi = np.array([3.0, 3.0, 2.0, 1.5])

    rng = np.random.RandomState(12345)

    # A few random initial points to seed the search
    n_starts = 8
    for s in range(n_starts):
        start = lo + rng.rand(4) * (hi - lo)
        if s == 0:
            start = np.array([1.0, 1.0, 0.5, 0.0])
        cur_params = start.copy()
        cur_score = _score(cur_params, prizes, dist, max_len)
        if cur_score > best_score:
            best_score = cur_score
            best_params = cur_params.copy()

        if time.time() > time_limit:
            break

        # Annealing schedule
        T = 0.5
        T_min = 0.001
        cool = 0.97
        step = 0.25
        accept = 0
        while T > T_min:
            if time.time() > time_limit:
                break
            # Propose a move
            delta = rng.randn(4) * step
            new_params = cur_params + delta
            new_params = np.clip(new_params, lo, hi)
            new_score = _score(new_params, prizes, dist, max_len)
            if new_score >= cur_score or rng.rand() < np.exp((new_score - cur_score) / T):
                cur_params = new_params
                cur_score = new_score
                accept += 1
                if cur_score > best_score:
                    best_score = cur_score
                    best_params = cur_params.copy()
            T *= cool

    return best_params, best_score


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    Uses simulated annealing over a 4-dim nonlinear heuristic family.
    """
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n = len(prizes)

    prizes = prizes.copy()
    prizes[0] = 0.0

    start_time = time.time()
    time_limit = start_time + 320.0  # safety margin before 330s kill

    # Fixed seed for reproducible ACO evaluation inside optimization
    np.random.seed(42)

    if time.time() < time_limit - 5.0:
        best_params, best_score = _optimize(prizes, distance_matrix, max_len, time_limit)
    else:
        best_params = np.array([1.0, 1.0, 0.5, 0.0])

    # Build final heuristic
    heuristic = _make_heuristic(prizes, distance_matrix, best_params)

    # Ensure correct shape
    if heuristic.shape != (n, n):
        heuristic = np.ones((n, n), dtype=float) * 1e-9

    # Ensure all values are positive and finite
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
