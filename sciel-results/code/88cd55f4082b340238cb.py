import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0  # Safety margin from 330s limit


def _run_aco_fast(D, eta, n_ants=20, n_iter=30, seed=42):
    """Run ACO with given heuristic eta, return best tour length."""
    n = D.shape[0]
    rng = np.random.RandomState(seed)

    tau = np.ones((n, n))
    best_len = float('inf')

    for it in range(n_iter):
        all_paths = []
        all_lens = []

        for ant in range(n_ants):
            current = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            path = [0]
            tour_len = 0.0

            for step in range(1, n):
                # Get available nodes
                available = np.where(~visited)[0]

                # Compute probabilities: tau * eta
                probs = tau[current, available] * eta[current, available]
                total = probs.sum()

                if total <= 0:
                    next_node = available[0]
                else:
                    probs = probs / total
                    r = rng.random()
                    cumsum = np.cumsum(probs)
                    next_node = available[np.searchsorted(cumsum, r)]

                tour_len += D[current, next_node]
                current = next_node
                visited[current] = True
                path.append(current)

            # Return to start
            tour_len += D[current, 0]
            path.append(0)

            if tour_len < best_len:
                best_len = tour_len

            all_paths.append(path)
            all_lens.append(tour_len)

        # Pheromone update
        tau *= 0.9
        for ant_idx in range(n_ants):
            l = all_lens[ant_idx]
            if l <= 0:
                continue
            reward = 1.0 / l
            path = all_paths[ant_idx]
            for k in range(len(path) - 1):
                i = path[k]
                j = path[k + 1]
                tau[i, j] += reward

    return best_len


def _two_opt(tour, dist_matrix, deadline):
    """Improve a tour using 2-opt heuristic. Returns the improved tour."""
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            if time.time() > deadline:
                return tour
            for j in range(i + 1, n):
                # Check if reversing the segment between i and j improves the tour
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]

                if j == i + 1:
                    continue

                current_cost = dist_matrix[a, b] + dist_matrix[c, d]
                new_cost = dist_matrix[a, c] + dist_matrix[b, d]

                if new_cost < current_cost - 1e-10:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
    return tour


def _make_eta(D, p, topo_tour=None, topo_weight=0.0):
    """Create heuristic matrix 1/d^p, optionally boosted by topological edges."""
    n = D.shape[0]
    eta = np.ones((n, n))
    mask = D > 0
    eta[mask] = 1.0 / np.power(D[mask], p)
    eta[~mask] = 1e-9

    if topo_tour is not None and topo_weight > 0:
        # Scale base heuristic so max is 1
        max_base = np.max(eta)
        if max_base > 0:
            eta = eta / max_base
        # Boost topological edges
        for i in range(len(topo_tour)):
            u = topo_tour[i]
            v = topo_tour[(i + 1) % len(topo_tour)]
            eta[u, v] = max(eta[u, v], topo_weight)
            eta[v, u] = max(eta[v, u], topo_weight)
        # Ensure positive
        eta[eta <= 0] = 1e-9

    return eta


def _nn_tour(D, deadline):
    """Generate a nearest-neighbor tour starting from node 0."""
    n = D.shape[0]
    visited = [False] * n
    tour = [0]
    visited[0] = True
    current = 0

    for _ in range(n - 1):
        if time.time() > deadline:
            break
        min_dist = float('inf')
        next_node = -1
        for j in range(n):
            if not visited[j]:
                if D[current, j] < min_dist:
                    min_dist = D[current, j]
                    next_node = j
        if next_node == -1:
            break
        tour.append(next_node)
        visited[next_node] = True
        current = next_node

    return tour if len(tour) == n else None


def heuristics(distance_matrix):
    D = np.asarray(distance_matrix, dtype=np.float64)
    n = D.shape[0]

    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        eta = 1.0 / (D + 1e-9)
        eta[D == 0] = 1e-9
        return eta

    # Allocate time for search: up to ~90 seconds per call
    search_deadline = min(time.time() + 90.0, _DEADLINE - 5.0)

    # Generate topological tour using NN + 2-opt
    topo_tour = None
    if time.time() < search_deadline - 10.0:
        topo_tour = _nn_tour(D, search_deadline - 15.0)
        if topo_tour is not None and time.time() < search_deadline - 5.0:
            topo_tour = _two_opt(topo_tour, D, search_deadline - 5.0)

    # Initial candidates for p
    candidates = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]

    best_p = 1.0
    best_score = float('inf')

    # Evaluate each candidate
    for p in candidates:
        if time.time() > search_deadline:
            break
        try:
            eta = _make_eta(D, p, topo_tour, topo_weight=5.0)
            score = _run_aco_fast(D, eta, seed=42)
            if score < best_score:
                best_score = score
                best_p = p
        except Exception:
            continue

    # Refine around best_p with smaller steps
    if time.time() < search_deadline - 10:
        step = 0.1
        current_p = best_p
        for _ in range(15):
            if time.time() > search_deadline:
                break
            improved = False
            for delta_p in [-step, step]:
                new_p = current_p + delta_p
                if new_p <= 0.1:
                    continue
                try:
                    eta = _make_eta(D, new_p, topo_tour, topo_weight=5.0)
                    score = _run_aco_fast(D, eta, seed=42)
                    if score < best_score:
                        best_score = score
                        best_p = new_p
                        current_p = new_p
                        improved = True
                        break
                except Exception:
                    continue
            if not improved:
                step *= 0.7
            if step < 0.01:
                break

    # Final refinement with even smaller steps if time permits
    if time.time() < search_deadline - 5:
        step = 0.02
        current_p = best_p
        for _ in range(10):
            if time.time() > search_deadline:
                break
            improved = False
            for delta_p in [-step, step]:
                new_p = current_p + delta_p
                if new_p <= 0.05:
                    continue
                try:
                    eta = _make_eta(D, new_p, topo_tour, topo_weight=5.0)
                    score = _run_aco_fast(D, eta, seed=42)
                    if score < best_score:
                        best_score = score
                        best_p = new_p
                        current_p = new_p
                        improved = True
                        break
                except Exception:
                    continue
            if not improved:
                break

    # Generate final heuristic
    eta = _make_eta(D, best_p, topo_tour, topo_weight=5.0)

    # Ensure all entries are positive and finite
    eta[eta <= 0] = 1e-9
    eta[~np.isfinite(eta)] = 1e-9

    return eta
